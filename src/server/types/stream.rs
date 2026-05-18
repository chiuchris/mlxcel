//! SSE streaming response types

use serde::Serialize;

/// Delta content for streaming
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Streaming choice
#[derive(Debug, Clone, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Chat completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl ChatCompletionChunk {
    /// Create initial chunk with role
    pub fn initial(id: String, model: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// Create content chunk
    pub fn content(id: String, model: String, content: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(content),
                },
                finish_reason: None,
            }],
        }
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason),
            }],
        }
    }
}

/// Text completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionStreamChoice {
    pub index: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl CompletionChunk {
    /// Create content chunk
    pub fn content(id: String, model: String, text: String) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text,
                finish_reason: None,
            }],
        }
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason),
            }],
        }
    }
}
