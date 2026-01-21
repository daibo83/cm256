//! Tests for the simplified reliable transport module.

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_config_default() {
        let config = ReliableConfig::default();
        assert_eq!(config.fec_delay, 8);
        assert_eq!(config.fec_parities, 2);
        assert_eq!(config.fec_step_size, 8);
        assert_eq!(config.ack_every_n_packets, 8);
    }

    #[test]
    fn test_config_presets() {
        let low_latency = ReliableConfig::low_latency();
        assert!(low_latency.fec_delay < ReliableConfig::default().fec_delay);

        let high_redundancy = ReliableConfig::high_redundancy();
        assert!(high_redundancy.fec_parities > ReliableConfig::default().fec_parities);
    }

    #[test]
    fn test_nack_generation() {
        let mut arq = ReceiverArq::new();

        arq.on_receive(0);
        arq.on_receive(1);
        arq.on_receive(3); // Skip 2

        assert!(arq.has_gaps());

        let nack = arq.build_nack();
        assert!(nack.sequences.contains(&2));
    }

    #[test]
    fn test_sender_retransmit() {
        let mut arq = SenderArq::new(128, 3);

        arq.on_send(b"packet0");
        arq.on_send(b"packet1");
        arq.on_send(b"packet2");

        let nack = NackPacket::new(vec![1]);
        let to_retransmit = arq.on_nack(&nack);
        assert_eq!(to_retransmit, vec![1]);

        let data = arq.get_retransmit(1).unwrap();
        assert_eq!(data, b"packet1");
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

    /// Async memory channel for testing.
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
    async fn test_session_creation() {
        let (tx, _rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let session = SimpleSession::new(config, tx);
        assert!(session.is_ok());
    }

    #[tokio::test]
    async fn test_session_send_receive() {
        let (sender_transport, receiver_transport) = AsyncMemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut sender = SimpleSession::new(config.clone(), sender_transport).unwrap();
        let mut receiver = SimpleSession::new(config.clone(), receiver_transport).unwrap();

        // Send a packet
        let data = vec![0x42u8; config.symbol_bytes];
        let seq = sender.send(&data).await.unwrap();
        assert_eq!(seq, 0);

        // Receive the packet
        let result = receiver.recv().await.unwrap();
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
    async fn test_session_multiple_packets() {
        let (sender_transport, receiver_transport) = AsyncMemoryChannel::pair();

        // Use minimal FEC config
        let mut config = ReliableConfig::default();
        config.fec_delay = 16; // Large delay so no parities are sent for 5 packets
        config.fec_step_size = 16;

        let mut sender = SimpleSession::new(config.clone(), sender_transport).unwrap();
        let mut receiver = SimpleSession::new(config.clone(), receiver_transport).unwrap();

        // Send 5 packets
        for i in 0..5u8 {
            let data = vec![i; config.symbol_bytes];
            let seq = sender.send(&data).await.unwrap();
            assert_eq!(seq, i as u16);
        }

        // Receive 5 source packets
        for i in 0..5u8 {
            let result = receiver.recv().await.unwrap();
            match result {
                RecvResult::Source { seq, data, .. } => {
                    assert_eq!(seq, i as u16);
                    assert_eq!(data[0], i);
                }
                _ => panic!("Expected Source, got {:?}", result),
            }
        }
    }

    #[tokio::test]
    async fn test_session_flush() {
        let (tx, _rx) = AsyncMemoryChannel::pair();

        let config = ReliableConfig::default();
        let mut sender = SimpleSession::new(config.clone(), tx).unwrap();

        // Send a few packets
        for i in 0..3u8 {
            let data = vec![i; config.symbol_bytes];
            sender.send(&data).await.unwrap();
        }

        // Flush should succeed
        assert!(sender.flush().await.is_ok());
    }

    #[tokio::test]
    async fn test_session_accessors() {
        let (tx, _rx) = AsyncMemoryChannel::pair();
        let config = ReliableConfig::default();
        let session = SimpleSession::new(config.clone(), tx).unwrap();

        assert_eq!(session.config().symbol_bytes, config.symbol_bytes);
        assert_eq!(session.sender_arq().total_retransmits(), 0);
    }
}
