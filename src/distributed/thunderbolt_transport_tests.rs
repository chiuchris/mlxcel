use super::*;
use crate::distributed::transport::{Transport, TransportBackend, TransportMessage};

#[test]
fn thunderbolt_not_available() {
    assert!(!ThunderboltTransport::is_available());
}

#[test]
fn thunderbolt_config_defaults() {
    let config = ThunderboltTransportConfig::default();
    assert_eq!(config.interface, "bridge0");
    assert_eq!(config.port, 9200);
    assert!(config.use_shared_memory);
    assert_eq!(config.max_transfer_size, 1024 * 1024 * 1024);
}

#[test]
fn thunderbolt_backend_type() {
    let transport = ThunderboltTransport::new(ThunderboltTransportConfig::default());
    assert_eq!(transport.backend(), TransportBackend::Thunderbolt);
}

#[test]
fn thunderbolt_local_addr() {
    let transport = ThunderboltTransport::new(ThunderboltTransportConfig::default());
    let addr = transport.local_addr().unwrap();
    assert_eq!(addr, "bridge0:9200");
}

#[test]
fn thunderbolt_accessors() {
    let config = ThunderboltTransportConfig {
        interface: "en5".to_string(),
        port: 9300,
        use_shared_memory: false,
        max_transfer_size: 512 * 1024 * 1024,
    };
    let transport = ThunderboltTransport::new(config);
    assert_eq!(transport.interface(), "en5");
    assert_eq!(transport.port(), 9300);
    assert!(!transport.use_shared_memory());
}

#[tokio::test]
async fn thunderbolt_operations_return_unavailable() {
    let transport = ThunderboltTransport::new(ThunderboltTransportConfig::default());

    assert!(transport.connect(&["peer:1234".to_string()]).await.is_err());
    assert!(
        transport
            .send(
                "peer:1234",
                TransportMessage::Control {
                    operation: "test".to_string(),
                    payload: bytes::Bytes::new(),
                },
            )
            .await
            .is_err()
    );
    assert!(transport.recv().await.is_err());
    assert!(
        transport
            .send_stream("peer:1234", bytes::Bytes::new())
            .await
            .is_err()
    );
    assert!(transport.recv_stream().await.is_err());
    assert!(transport.rpc_call("peer:1234", b"req").await.is_err());
    assert!(transport.serve_rpc(Box::new(|_| vec![])).await.is_err());

    // shutdown always succeeds
    assert!(transport.shutdown().await.is_ok());
}
