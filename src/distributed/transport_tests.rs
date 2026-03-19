use super::*;

#[test]
fn transport_message_payload_size() {
    let tensor_msg = TransportMessage::TensorData {
        tensor_id: "layer.0.weight".to_string(),
        shape: vec![128, 64],
        data: bytes::Bytes::from(vec![0u8; 1024]),
    };
    assert_eq!(tensor_msg.payload_size(), 1024);

    let ctrl_msg = TransportMessage::Control {
        operation: "heartbeat".to_string(),
        payload: bytes::Bytes::from(vec![1u8; 256]),
    };
    assert_eq!(ctrl_msg.payload_size(), 256);
}

#[test]
fn message_kind_roundtrip() {
    for (byte, expected) in [
        (1u8, MessageKind::TensorData),
        (2, MessageKind::Control),
        (3, MessageKind::RpcRequest),
        (4, MessageKind::RpcResponse),
    ] {
        let kind = MessageKind::try_from(byte).unwrap();
        assert_eq!(kind, expected);
        assert_eq!(kind as u8, byte);
    }
}

#[test]
fn message_kind_rejects_unknown() {
    assert!(MessageKind::try_from(0u8).is_err());
    assert!(MessageKind::try_from(5u8).is_err());
    assert!(MessageKind::try_from(255u8).is_err());
}

#[test]
fn transport_backend_display() {
    assert_eq!(TransportBackend::Tcp.to_string(), "tcp");
    assert_eq!(TransportBackend::Thunderbolt.to_string(), "thunderbolt");
}
