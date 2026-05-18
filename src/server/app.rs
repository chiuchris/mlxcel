//! Axum application configuration

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use super::AppState;
use super::routes;
use super::types::ErrorResponse;

/// API key authentication middleware
async fn api_key_auth(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no API key is configured, skip authentication
    let Some(expected_key) = state.config.api_key.as_ref() else {
        return next.run(request).await;
    };

    // Skip auth for health check endpoints
    let path = request.uri().path();
    if path == "/health" || path == "/" {
        return next.run(request).await;
    }

    // Check for Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());

    match auth_header {
        Some(auth) => {
            // Support "Bearer <token>" format
            let token = if let Some(stripped) = auth.strip_prefix("Bearer ") {
                stripped
            } else {
                auth
            };

            if token == expected_key {
                next.run(request).await
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse::new(
                        "Invalid API key".to_string(),
                        "invalid_api_key",
                    )),
                )
                    .into_response()
            }
        }
        None => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse::new(
                "Missing API key. Use 'Authorization: Bearer <api-key>' header.".to_string(),
                "missing_api_key",
            )),
        )
            .into_response(),
    }
}

/// Create the Axum application router
pub fn create_app(state: AppState) -> Router {
    let enable_slots = state.config.enable_slots_endpoint;
    let enable_props = state.config.enable_props_endpoint;
    let enable_metrics = state.config.enable_metrics_endpoint;

    let mut app = Router::new()
        // OpenAI API endpoints
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/models", get(routes::list_models))
        // Aliases (some clients use these)
        .route("/chat/completions", post(routes::chat_completions))
        .route("/completions", post(routes::completions))
        .route("/models", get(routes::list_models))
        // llama-server compatible endpoints
        .route("/completion", post(routes::native_completion))
        .route("/tokenize", post(routes::tokenize))
        .route("/detokenize", post(routes::detokenize));

    // Conditionally enable /props endpoint
    if enable_props {
        app = app.route("/props", get(routes::props));
    }

    // Conditionally enable /slots endpoint
    if enable_slots {
        app = app.route("/slots", get(routes::slots));
    }

    // Conditionally enable /metrics endpoint
    if enable_metrics {
        app = app.route("/metrics", get(routes::metrics));
    }

    app
        // Health check
        .route("/health", get(routes::health_check))
        .route("/", get(routes::health_check))
        // Middleware
        .layer(middleware::from_fn_with_state(state.clone(), api_key_auth))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
