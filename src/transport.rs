//! # Datagram Transport Abstraction
//!
//! This module provides traits for abstracting over datagram-based transports,
//! allowing the streaming and diagonal FEC modules to work with any transport
//! (UDP, QUIC, unix datagram sockets, in-memory channels, etc.).
//!
//! ## Traits
//!
//! - [`DatagramSend`]: Synchronous sending of datagrams
//! - [`DatagramRecv`]: Synchronous receiving of datagrams
//! - [`DatagramTransport`]: Combined send + receive
//!
//! ## Async Traits (requires `async` feature)
//!
//! - [`AsyncDatagramSend`]: Asynchronous sending
//! - [`AsyncDatagramRecv`]: Asynchronous receiving
//! - [`AsyncDatagramTransport`]: Combined async send + receive
//!
//! ## Example
//!
//! ```rust
//! use cm256::transport::{DatagramSend, DatagramRecv};
//! use std::io;
//!
//! // Implement for std::net::UdpSocket
//! struct UdpTransport {
//!     socket: std::net::UdpSocket,
//! }
//!
//! impl DatagramSend for UdpTransport {
//!     fn send_datagram(&self, data: &[u8]) -> io::Result<usize> {
//!         self.socket.send(data)
//!     }
//! }
//!
//! impl DatagramRecv for UdpTransport {
//!     fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize> {
//!         self.socket.recv(buf)
//!     }
//! }
//! ```

use std::io;

// =============================================================================
// Synchronous Traits
// =============================================================================

/// Trait for sending datagrams over a transport.
///
/// Implementors should send the entire datagram atomically. Partial sends
/// are not supported for datagram-based transports.
pub trait DatagramSend {
    /// Send a datagram.
    ///
    /// Returns the number of bytes sent (should equal `data.len()` for datagrams).
    fn send_datagram(&self, data: &[u8]) -> io::Result<usize>;
}

/// Trait for receiving datagrams from a transport.
///
/// Implementors should receive a complete datagram into the buffer.
pub trait DatagramRecv {
    /// Receive a datagram.
    ///
    /// Returns the number of bytes received. If the datagram is larger than
    /// the buffer, the excess bytes may be discarded (transport-dependent).
    fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize>;

    /// Try to receive a datagram without blocking.
    ///
    /// Returns `Ok(Some(n))` if a datagram was received, `Ok(None)` if no
    /// datagram is available, or `Err` on error.
    fn try_recv_datagram(&self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        match self.recv_datagram(buf) {
            Ok(n) => Ok(Some(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Combined trait for bidirectional datagram transport.
pub trait DatagramTransport: DatagramSend + DatagramRecv {}

// Blanket implementation
impl<T: DatagramSend + DatagramRecv> DatagramTransport for T {}

// =============================================================================
// Mutable variants (for transports that require &mut self)
// =============================================================================

/// Trait for sending datagrams (mutable version).
///
/// Use this for transports that require mutable access for sending.
pub trait DatagramSendMut {
    /// Send a datagram.
    fn send_datagram(&mut self, data: &[u8]) -> io::Result<usize>;
}

/// Trait for receiving datagrams (mutable version).
///
/// Use this for transports that require mutable access for receiving.
pub trait DatagramRecvMut {
    /// Receive a datagram.
    fn recv_datagram(&mut self, buf: &mut [u8]) -> io::Result<usize>;

    /// Try to receive a datagram without blocking.
    fn try_recv_datagram(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        match self.recv_datagram(buf) {
            Ok(n) => Ok(Some(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Combined trait for bidirectional datagram transport (mutable version).
pub trait DatagramTransportMut: DatagramSendMut + DatagramRecvMut {}

// Blanket implementation
impl<T: DatagramSendMut + DatagramRecvMut> DatagramTransportMut for T {}

// =============================================================================
// Blanket implementations for &mut T
// =============================================================================

impl<T: DatagramSend> DatagramSendMut for T {
    fn send_datagram(&mut self, data: &[u8]) -> io::Result<usize> {
        DatagramSend::send_datagram(self, data)
    }
}

impl<T: DatagramRecv> DatagramRecvMut for T {
    fn recv_datagram(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        DatagramRecv::recv_datagram(self, buf)
    }

    fn try_recv_datagram(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        DatagramRecv::try_recv_datagram(self, buf)
    }
}

// =============================================================================
// Standard Library Implementations
// =============================================================================

/// Connected UDP socket implementation.
///
/// Requires the socket to be connected via `connect()` before use.
impl DatagramSend for std::net::UdpSocket {
    fn send_datagram(&self, data: &[u8]) -> io::Result<usize> {
        self.send(data)
    }
}

impl DatagramRecv for std::net::UdpSocket {
    fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv(buf)
    }
}

#[cfg(unix)]
impl DatagramSend for std::os::unix::net::UnixDatagram {
    fn send_datagram(&self, data: &[u8]) -> io::Result<usize> {
        self.send(data)
    }
}

#[cfg(unix)]
impl DatagramRecv for std::os::unix::net::UnixDatagram {
    fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.recv(buf)
    }
}

// =============================================================================
// In-Memory Channel Transport (for testing)
// =============================================================================

/// In-memory datagram channel for testing.
///
/// Uses crossbeam-style bounded channels internally.
#[derive(Debug)]
pub struct MemoryChannel {
    sender: std::sync::mpsc::Sender<Vec<u8>>,
    receiver: std::sync::mpsc::Receiver<Vec<u8>>,
}

impl MemoryChannel {
    /// Create a pair of connected memory channels.
    ///
    /// Returns `(a, b)` where datagrams sent on `a` are received on `b` and vice versa.
    pub fn pair() -> (Self, Self) {
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();

        (
            Self {
                sender: tx1,
                receiver: rx2,
            },
            Self {
                sender: tx2,
                receiver: rx1,
            },
        )
    }
}

impl DatagramSend for MemoryChannel {
    fn send_datagram(&self, data: &[u8]) -> io::Result<usize> {
        self.sender
            .send(data.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))?;
        Ok(data.len())
    }
}

impl DatagramRecv for MemoryChannel {
    fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize> {
        let data = self
            .receiver
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))?;
        let len = data.len().min(buf.len());
        buf[..len].copy_from_slice(&data[..len]);
        Ok(len)
    }

    fn try_recv_datagram(&self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        match self.receiver.try_recv() {
            Ok(data) => {
                let len = data.len().min(buf.len());
                buf[..len].copy_from_slice(&data[..len]);
                Ok(Some(len))
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))
            }
        }
    }
}

// =============================================================================
// Lossy Channel (for testing FEC recovery)
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// A wrapper that simulates packet loss for testing FEC.
///
/// Drops packets based on a configured loss pattern.
#[derive(Debug)]
pub struct LossyChannel<T> {
    inner: T,
    /// Counter for sent packets
    send_counter: AtomicU64,
    /// Counter for received packets
    recv_counter: AtomicU64,
    /// Loss pattern: drop every Nth packet (0 = no loss)
    drop_every_n: u64,
    /// Random loss probability (0-100)
    random_loss_percent: u8,
}

impl<T> LossyChannel<T> {
    /// Wrap a transport with deterministic packet loss.
    ///
    /// `drop_every_n`: Drop every Nth packet (0 = no loss)
    pub fn with_pattern(inner: T, drop_every_n: u64) -> Self {
        Self {
            inner,
            send_counter: AtomicU64::new(0),
            recv_counter: AtomicU64::new(0),
            drop_every_n,
            random_loss_percent: 0,
        }
    }

    /// Wrap a transport with random packet loss.
    ///
    /// `loss_percent`: Probability of dropping each packet (0-100)
    pub fn with_random_loss(inner: T, loss_percent: u8) -> Self {
        Self {
            inner,
            send_counter: AtomicU64::new(0),
            recv_counter: AtomicU64::new(0),
            drop_every_n: 0,
            random_loss_percent: loss_percent.min(100),
        }
    }

    /// Get the underlying transport.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Check if a packet should be dropped.
    fn should_drop(&self, counter: u64) -> bool {
        // Deterministic pattern loss
        if self.drop_every_n > 0 && counter % self.drop_every_n == 0 {
            return true;
        }

        // Random loss (using counter as seed for reproducibility)
        if self.random_loss_percent > 0 {
            // Use high bits of a multiplicative hash for better distribution
            let hash = counter.wrapping_mul(0x9E3779B97F4A7C15); // Golden ratio constant
            let roll = ((hash >> 56) as u8) % 100; // Use top 8 bits, then mod 100
            if roll < self.random_loss_percent {
                return true;
            }
        }

        false
    }
}

impl<T: DatagramSend> DatagramSend for LossyChannel<T> {
    fn send_datagram(&self, data: &[u8]) -> io::Result<usize> {
        let count = self.send_counter.fetch_add(1, Ordering::Relaxed);
        if self.should_drop(count) {
            // Pretend we sent it (simulate network drop)
            return Ok(data.len());
        }
        self.inner.send_datagram(data)
    }
}

impl<T: DatagramRecv> DatagramRecv for LossyChannel<T> {
    fn recv_datagram(&self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let n = self.inner.recv_datagram(buf)?;
            let count = self.recv_counter.fetch_add(1, Ordering::Relaxed);
            if !self.should_drop(count) {
                return Ok(n);
            }
            // Dropped - try to receive next packet
        }
    }

    fn try_recv_datagram(&self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        loop {
            match self.inner.try_recv_datagram(buf)? {
                Some(n) => {
                    let count = self.recv_counter.fetch_add(1, Ordering::Relaxed);
                    if !self.should_drop(count) {
                        return Ok(Some(n));
                    }
                    // Dropped - try again
                }
                None => return Ok(None),
            }
        }
    }
}

// =============================================================================
// Async Transport Traits
// =============================================================================

use std::future::Future;
use std::pin::Pin;

/// Async trait for sending datagrams.
///
/// This trait uses explicit future types to avoid requiring `async_trait` macro.
pub trait AsyncDatagramSend {
    /// Send a datagram asynchronously.
    fn send_datagram_async<'a>(
        &'a self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>>;
}

/// Async trait for receiving datagrams.
pub trait AsyncDatagramRecv {
    /// Receive a datagram asynchronously.
    fn recv_datagram_async<'a>(
        &'a self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>>;
}

/// Combined async trait for bidirectional datagram transport.
pub trait AsyncDatagramTransport: AsyncDatagramSend + AsyncDatagramRecv {}

// Blanket implementation
impl<T: AsyncDatagramSend + AsyncDatagramRecv> AsyncDatagramTransport for T {}

/// Mutable async send trait.
pub trait AsyncDatagramSendMut {
    /// Send a datagram asynchronously.
    fn send_datagram_async<'a>(
        &'a mut self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>>;
}

/// Mutable async receive trait.
pub trait AsyncDatagramRecvMut {
    /// Receive a datagram asynchronously.
    fn recv_datagram_async<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>>;
}

/// Combined mutable async trait.
pub trait AsyncDatagramTransportMut: AsyncDatagramSendMut + AsyncDatagramRecvMut {}

impl<T: AsyncDatagramSendMut + AsyncDatagramRecvMut> AsyncDatagramTransportMut for T {}

// Blanket impl: immutable async -> mutable async
impl<T: AsyncDatagramSend> AsyncDatagramSendMut for T {
    fn send_datagram_async<'a>(
        &'a mut self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        AsyncDatagramSend::send_datagram_async(self, data)
    }
}

impl<T: AsyncDatagramRecv> AsyncDatagramRecvMut for T {
    fn recv_datagram_async<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        AsyncDatagramRecv::recv_datagram_async(self, buf)
    }
}

// =============================================================================
// Tokio Implementations
// =============================================================================

/// Async transport implementation for `tokio::net::UdpSocket`.
///
/// Requires the `tokio` feature to be enabled.
///
/// # Example
///
/// ```rust,ignore
/// use cm256::streaming::{StreamingParams, AsyncTransportEncoder};
/// use tokio::net::UdpSocket;
///
/// #[tokio::main]
/// async fn main() -> std::io::Result<()> {
///     let socket = UdpSocket::bind("0.0.0.0:0").await?;
///     socket.connect("127.0.0.1:9000").await?;
///
///     let params = StreamingParams::new(8, 2, 1200).unwrap();
///     let mut encoder = AsyncTransportEncoder::new(params, socket);
///
///     encoder.send(&[0x42; 1200]).await?;
///     Ok(())
/// }
/// ```
#[cfg(feature = "tokio")]
impl AsyncDatagramSend for tokio::net::UdpSocket {
    fn send_datagram_async<'a>(
        &'a self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move { self.send(data).await })
    }
}

#[cfg(feature = "tokio")]
impl AsyncDatagramRecv for tokio::net::UdpSocket {
    fn recv_datagram_async<'a>(
        &'a self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move { self.recv(buf).await })
    }
}

// =============================================================================
// Quinn (QUIC) Implementations
// =============================================================================

/// Async transport implementation for `quinn::Connection` (QUIC datagrams).
///
/// Requires the `quinn` feature to be enabled.
///
/// QUIC datagrams (RFC 9221) are unreliable, unordered messages - ideal for
/// real-time streaming with FEC protection.
///
/// # Example
///
/// ```rust,ignore
/// use cm256::streaming::{StreamingParams, AsyncTransportEncoder};
/// use quinn::Connection;
///
/// async fn send_with_fec(conn: Connection) -> std::io::Result<()> {
///     let params = StreamingParams::new(8, 2, 1200).unwrap();
///     let mut encoder = AsyncTransportEncoder::new(params, conn);
///
///     encoder.send(&[0x42; 1200]).await?;
///     Ok(())
/// }
/// ```
///
/// # Notes
///
/// - `send_datagram` is actually synchronous in Quinn (queues the datagram)
/// - `read_datagram` is async and waits for incoming datagrams
/// - QUIC datagrams have a maximum size based on path MTU; use
///   `Connection::max_datagram_size()` to check
#[cfg(feature = "quinn")]
impl AsyncDatagramSend for quinn::Connection {
    fn send_datagram_async<'a>(
        &'a self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            let bytes = bytes::Bytes::copy_from_slice(data);
            self.send_datagram(bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            Ok(data.len())
        })
    }
}

#[cfg(feature = "quinn")]
impl AsyncDatagramRecv for quinn::Connection {
    fn recv_datagram_async<'a>(
        &'a self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            let datagram = self
                .read_datagram()
                .await
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let len = datagram.len().min(buf.len());
            buf[..len].copy_from_slice(&datagram[..len]);
            Ok(len)
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_channel() {
        let (a, b) = MemoryChannel::pair();

        // Send from a to b
        let data = b"hello world";
        assert_eq!(a.send_datagram(data).unwrap(), data.len());

        let mut buf = [0u8; 64];
        let n = b.recv_datagram(&mut buf).unwrap();
        assert_eq!(&buf[..n], data);

        // Send from b to a
        let data2 = b"goodbye";
        assert_eq!(b.send_datagram(data2).unwrap(), data2.len());

        let n = a.recv_datagram(&mut buf).unwrap();
        assert_eq!(&buf[..n], data2);
    }

    #[test]
    fn test_try_recv() {
        let (a, b) = MemoryChannel::pair();

        let mut buf = [0u8; 64];
        assert!(matches!(b.try_recv_datagram(&mut buf), Ok(None)));

        a.send_datagram(b"test").unwrap();
        let n = b.try_recv_datagram(&mut buf).unwrap().unwrap();
        assert_eq!(&buf[..n], b"test");

        assert!(matches!(b.try_recv_datagram(&mut buf), Ok(None)));
    }

    #[test]
    fn test_lossy_channel_pattern() {
        let (a, b) = MemoryChannel::pair();
        let lossy = LossyChannel::with_pattern(a, 3); // Drop every 3rd packet

        // Send 10 packets
        for i in 0..10u8 {
            lossy.send_datagram(&[i]).unwrap();
        }

        let mut buf = [0u8; 1];
        let mut received = Vec::new();
        while let Ok(Some(_)) = b.try_recv_datagram(&mut buf) {
            received.push(buf[0]);
        }

        // Should receive all except packets 0, 3, 6, 9 (every 3rd starting from 0)
        assert_eq!(received, vec![1, 2, 4, 5, 7, 8]);
    }

    #[test]
    fn test_lossy_channel_random() {
        let (a, b) = MemoryChannel::pair();
        let lossy = LossyChannel::with_random_loss(a, 50); // 50% loss

        // Send 100 packets
        for i in 0..100u8 {
            lossy.send_datagram(&[i]).unwrap();
        }

        let mut buf = [0u8; 1];
        let mut count = 0;
        while let Ok(Some(_)) = b.try_recv_datagram(&mut buf) {
            count += 1;
        }

        // With 50% loss, should receive roughly half
        // Allow wide tolerance due to deterministic hash variation
        assert!(count >= 25 && count <= 75, "received {} packets", count);
    }
}
