use serde::{Serialize, Serializer};

use super::{DONE_MARKER, payload_channel, serialize_json_data};

#[derive(Serialize)]
struct TestPayload<'a> {
    token: &'a str,
}

struct FailingPayload;

impl Serialize for FailingPayload {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom("boom"))
    }
}

#[test]
fn blocking_sse_sender_sends_json_text_and_done_in_order() {
    let (sender, mut rx) = payload_channel(4);

    sender.json(&TestPayload { token: "hello" }).unwrap();
    sender.text("plain-text");
    sender.done();

    assert_eq!(
        rx.blocking_recv().unwrap().unwrap(),
        r#"{"token":"hello"}"#.to_string()
    );
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), "plain-text");
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), DONE_MARKER);
}

#[test]
fn done_marker_matches_openai_stream_terminator() {
    assert_eq!(DONE_MARKER, "[DONE]");
}

#[test]
fn serialize_json_data_returns_errors_instead_of_panicking() {
    let err = serialize_json_data(&FailingPayload)
        .unwrap_err()
        .to_string();
    assert!(err.contains("boom"));
}
