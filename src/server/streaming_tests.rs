use serde::Serialize;

use super::{DONE_MARKER, payload_channel};

#[derive(Serialize)]
struct TestPayload<'a> {
    token: &'a str,
}

#[test]
fn blocking_sse_sender_sends_json_text_and_done_in_order() {
    let (sender, mut rx) = payload_channel(4);

    sender.json(&TestPayload { token: "hello" });
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
