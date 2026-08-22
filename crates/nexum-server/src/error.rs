//! HTTP error mapping.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// An error as the API returns it.
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub error: String,
    /// Machine-readable class, so clients can branch without parsing prose.
    pub kind: &'static str,
}

/// Errors the API can return.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    BadRequest(String),

    #[error(transparent)]
    Client(#[from] nexum_client::ClientError),
}

impl ApiError {
    fn parts(&self) -> (StatusCode, &'static str) {
        match self {
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ApiError::Client(inner) => match inner {
                // A malformed query or an unknown model is the caller's
                // mistake, not a server fault — returning 500 for these sends
                // clients into pointless retry loops.
                nexum_client::ClientError::Core(nexum_core::Error::InvalidArgument(_))
                | nexum_client::ClientError::Core(nexum_core::Error::InvalidId(_))
                | nexum_client::ClientError::Core(nexum_core::Error::UnknownNamespace(_))
                | nexum_client::ClientError::Core(nexum_core::Error::DimensionMismatch {
                    ..
                }) => (StatusCode::BAD_REQUEST, "bad_request"),
                nexum_client::ClientError::Core(nexum_core::Error::NodeNotFound(_)) => {
                    (StatusCode::NOT_FOUND, "not_found")
                }
                nexum_client::ClientError::Embed(_) => {
                    (StatusCode::BAD_GATEWAY, "embedding_provider")
                }
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = self.parts();
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        }
        (
            status,
            Json(ApiErrorBody {
                error: self.to_string(),
                kind,
            }),
        )
            .into_response()
    }
}

/// Engine errors reach the API through the client's variant, so they get the
/// same status mapping wherever they come from.
impl From<nexum_core::Error> for ApiError {
    fn from(e: nexum_core::Error) -> Self {
        ApiError::Client(nexum_client::ClientError::Core(e))
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
