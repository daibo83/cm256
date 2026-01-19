//! Tests for the reliable transport module.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::transport::MemoryChannel;

    #[test]
    fn test_config_default() {
        let config = ReliableConfig::default();
        assert_eq!(config.window_size, 64);
        assert_eq!(config.ack_every_n_packets, 8);
        assert_eq!(config.fec_delay, 8);
        assert_eq!(config.fec_parities, 1);
        assert_eq!(config.fec_step_size, 8);
        // Default overhead: 1/8 = 12.5%
    }

    #[test]
    fn test_config_presets() {
        let low_latency = ReliableConfig::low_latency();
        assert!(low_latency.ack_interval_ms < ReliableConfig::default().ack_interval_ms);

        let high_throughput = ReliableConfig::high_throughput();
        assert!(
            high_throughput.ack_every_n_packets > ReliableConfig::default().ack_every_n_packets
        );
    }

    #[test]
    fn test_encoder_decoder_creation() {
        let (tx, rx) = MemoryChannel::pair();

        let config = ReliableConfig::default();
        let encoder = ReliableEncoder::new(config.clone(), tx);
        let decoder = ReliableDecoder::new(config, rx);

        assert!(encoder.is_ok());
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_send_receive_basic() {
        let (tx, rx) = MemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = ReliableEncoder::new(config.clone(), tx).unwrap();
        let mut decoder = ReliableDecoder::new(config.clone(), rx).unwrap();

        // Send a packet
        let data = vec![0x42u8; config.symbol_bytes];
        let seq = encoder.send(&data).unwrap();
        assert_eq!(seq, 0);

        // Receive the packet
        let result = decoder.recv_no_ack().unwrap();
        match result {
            RecvResult::Source {
                seq,
                data: recv_data,
                recovered,
            } => {
                assert_eq!(seq, 0);
                assert_eq!(recv_data, data);
                assert!(!recovered);
            }
            _ => panic!("Expected Source result"),
        }
    }

    #[test]
    fn test_multiple_packets() {
        let (tx, rx) = MemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = ReliableEncoder::new(config.clone(), tx).unwrap();
        let mut decoder = ReliableDecoder::new(config.clone(), rx).unwrap();

        // Send multiple packets
        for i in 0..10u8 {
            let data = vec![i; config.symbol_bytes];
            let seq = encoder.send(&data).unwrap();
            assert_eq!(seq, i as u16);
        }

        // Receive all packets (source + parity)
        let mut received_sources = 0;
        while let Ok(result) = decoder.try_recv() {
            match result {
                RecvResult::Source { .. } => received_sources += 1,
                RecvResult::Parity => {}
                RecvResult::WouldBlock => break,
                _ => {}
            }
        }

        // Should have received at least the source packets
        assert!(received_sources >= 10);
    }

    #[test]
    fn test_ack_generation() {
        let mut arq = ReceiverArq::new();
        let stats = NetworkStats::new();

        arq.on_receive(0);
        arq.on_receive(1);
        arq.on_receive(3); // Skip 2

        let ack = arq.build_ack(&stats);

        // After receiving 0, 1, 3:
        // - Window advances to base_seq=2 (first gap)
        // - Bitmap: bit 0 (seq 2) = 0 (missing), bit 1 (seq 3) = 1 (received)
        assert_eq!(ack.base_seq, 2);
        assert_eq!(ack.is_received(2), Some(false)); // Gap
        assert_eq!(ack.is_received(3), Some(true));
    }

    #[test]
    fn test_ack_roundtrip() {
        let (tx, rx) = MemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = ReliableEncoder::new(config.clone(), tx).unwrap();
        let mut decoder = ReliableDecoder::new(config.clone(), rx).unwrap();

        // Send enough packets to trigger ACK
        for i in 0..(config.ack_every_n_packets + 1) as u8 {
            let data = vec![i; config.symbol_bytes];
            encoder.send(&data).unwrap();
        }

        // Receive packets (should trigger ACK)
        while let Ok(result) = decoder.try_recv() {
            if matches!(result, RecvResult::WouldBlock) {
                break;
            }
        }

        // Force an ACK
        decoder.force_ack().unwrap();

        // Encoder should be able to process the ACK
        // (In real scenario, we'd need bidirectional transport)
    }
}

// =============================================================================
// Async Tests
// =============================================================================

#[cfg(all(test, not(target_arch = "wasm32")))]
mod async_tests {
    use super::super::*;
    use std::future::Future;
    use std::io;
    use std::pin::Pin;
    use tokio::sync::mpsc;

    /// Async memory channel for testing async encoder/decoder.
    struct AsyncMemoryChannel {
        sender: mpsc::UnboundedSender<Vec<u8>>,
        receiver: tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<u8>>>,
    }

    impl AsyncMemoryChannel {
        fn pair() -> (Self, Self) {
            let (tx1, rx1) = mpsc::unbounded_channel();
            let (tx2, rx2) = mpsc::unbounded_channel();

            (
                Self {
                    sender: tx1,
                    receiver: tokio::sync::Mutex::new(rx2),
                },
                Self {
                    sender: tx2,
                    receiver: tokio::sync::Mutex::new(rx1),
                },
            )
        }
    }

    impl crate::transport::AsyncDatagramSendMut for AsyncMemoryChannel {
        fn send_datagram_async<'a>(
            &'a mut self,
            data: &'a [u8],
        ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
            Box::pin(async move {
                self.sender
                    .send(data.to_vec())
                    .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))?;
                Ok(data.len())
            })
        }
    }

    impl crate::transport::AsyncDatagramRecvMut for AsyncMemoryChannel {
        fn recv_datagram_async<'a>(
            &'a mut self,
            buf: &'a mut [u8],
        ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
            Box::pin(async move {
                let mut rx = self.receiver.lock().await;
                let data = rx
                    .recv()
                    .await
                    .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))?;
                let len = data.len().min(buf.len());
                buf[..len].copy_from_slice(&data[..len]);
                Ok(len)
            })
        }
    }

    #[tokio::test]
    async fn test_async_encoder_creation() {
        let (tx, _rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let encoder = AsyncReliableEncoder::new(config, tx);
        assert!(encoder.is_ok());
    }

    #[tokio::test]
    async fn test_async_decoder_creation() {
        let (_tx, rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let decoder = AsyncReliableDecoder::new(config, rx);
        assert!(decoder.is_ok());
    }

    #[tokio::test]
    async fn test_async_send_receive() {
        let (tx, rx) = AsyncMemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = AsyncReliableEncoder::new(config.clone(), tx).unwrap();
        let mut decoder = AsyncReliableDecoder::new(config.clone(), rx).unwrap();

        // Send a packet
        let data = vec![0x42u8; config.symbol_bytes];
        let seq = encoder.send(&data).await.unwrap();
        assert_eq!(seq, 0);

        // Receive the packet
        let result = decoder.recv_no_ack().await.unwrap();
        match result {
            RecvResult::Source {
                seq,
                data: recv_data,
                recovered,
            } => {
                assert_eq!(seq, 0);
                assert_eq!(recv_data, data);
                assert!(!recovered);
            }
            _ => panic!("Expected Source result"),
        }
    }

    #[tokio::test]
    async fn test_async_multiple_packets() {
        let (tx, rx) = AsyncMemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = AsyncReliableEncoder::new(config.clone(), tx).unwrap();
        let mut decoder = AsyncReliableDecoder::new(config.clone(), rx).unwrap();

        // Send multiple packets
        for i in 0..5u8 {
            let data = vec![i; config.symbol_bytes];
            let seq = encoder.send(&data).await.unwrap();
            assert_eq!(seq, i as u16);
        }

        // Receive packets (source + parity)
        let mut received_sources = 0;
        for _ in 0..20 {
            // Enough iterations to receive all
            let result = decoder.recv_no_ack().await.unwrap();
            match result {
                RecvResult::Source { seq, data, .. } => {
                    assert_eq!(data[0], seq as u8);
                    received_sources += 1;
                    if received_sources >= 5 {
                        break;
                    }
                }
                RecvResult::Parity => {}
                _ => {}
            }
        }

        assert_eq!(received_sources, 5);
    }

    #[tokio::test]
    async fn test_async_flush() {
        let (tx, _rx) = AsyncMemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut encoder = AsyncReliableEncoder::new(config.clone(), tx).unwrap();

        // Send a few packets
        for i in 0..3u8 {
            let data = vec![i; config.symbol_bytes];
            encoder.send(&data).await.unwrap();
        }

        // Flush should succeed
        let result = encoder.flush().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_encoder_accessors() {
        let (tx, _rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let encoder = AsyncReliableEncoder::new(config.clone(), tx).unwrap();

        assert_eq!(encoder.config().symbol_bytes, config.symbol_bytes);
        assert!(!encoder.arq().is_full());
        assert!(encoder.can_send());
    }

    #[tokio::test]
    async fn test_async_decoder_accessors() {
        let (_tx, rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let decoder = AsyncReliableDecoder::new(config.clone(), rx).unwrap();

        assert_eq!(decoder.config().symbol_bytes, config.symbol_bytes);
        assert!(!decoder.has_source(0));
    }
}
