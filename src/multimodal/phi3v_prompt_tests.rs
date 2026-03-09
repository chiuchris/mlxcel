use super::{
    Phi3vImageTag, Phi3vPromptTokens, collect_phi3v_image_tags, ensure_phi3v_image_tags,
    prepare_phi3v_prompt_tokens,
};

#[test]
fn ensure_phi3v_image_tags_inserts_after_user_marker() {
    let prompt = "<|system|>\nS\n<|user|>\nDescribe this.";

    let tagged = ensure_phi3v_image_tags(prompt, 2);

    assert_eq!(
        tagged,
        "<|system|>\nS\n<|user|>\n<|image_1|>\n<|image_2|>\nDescribe this."
    );
}

#[test]
fn collect_phi3v_image_tags_sorts_by_position() {
    let tags = collect_phi3v_image_tags("x <|image_2|> y <|image_1|>", 2);

    assert_eq!(
        tags,
        vec![
            Phi3vImageTag {
                start: 2,
                end: 13,
                image_num: 2,
            },
            Phi3vImageTag {
                start: 16,
                end: 27,
                image_num: 1,
            },
        ]
    );
}

#[test]
fn prepare_phi3v_prompt_tokens_interleaves_negative_image_ids() {
    let prepared = prepare_phi3v_prompt_tokens(
        "before <|image_2|> middle <|image_1|> after",
        2,
        |text, add_special| {
            let mut out = Vec::new();
            if add_special {
                out.push(1000);
            }
            out.push(match text {
                "before " => 1,
                " middle " => 2,
                " after" => 3,
                other => panic!("unexpected text chunk: {other:?}"),
            });
            out
        },
        |image_num| image_num,
    );

    assert_eq!(
        prepared,
        Some(Phi3vPromptTokens {
            tokens: vec![1000, 1, -2, -2, 2, -1, 3],
            image_slots: 2,
        })
    );
}
