//! The NexumDB HTTP API.
//!
//! This is the only way anything outside the process reaches the engine. The
//! viewer, the CLI's remote mode, and any future integration all speak it, and
//! nothing touches the storage files directly — which is what keeps a second
//! writer from corrupting a database that is built around a single one.
//!
//! JSON over HTTP rather than gRPC: the viewer is a web frontend, and a
//! browser-reachable API with no proxy layer is worth more here than protobuf's
//! wire efficiency.

pub mod error;
pub mod projection;
pub mod routes;

pub use error::{ApiError, ApiResult};

use axum::Router;
use nexum_client::Nexum;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Shared state handed to every request.
#[derive(Clone)]
pub struct ServerState {
    pub nexum: Arc<Nexum>,
}

/// How the server is configured.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub addr: SocketAddr,
    /// Allow browser origins to call the API.
    ///
    /// The desktop viewer runs from a `tauri://` or `http://localhost` origin
    /// that is not the API's own, so without this it cannot make a single
    /// request. Defaults to permissive because the server binds loopback.
    pub permissive_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            // Loopback by default: this database has no authentication, so
            // binding a public interface would expose it to the network.
            addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
            permissive_cors: true,
        }
    }
}

/// Build the router.
pub fn router(state: ServerState, permissive_cors: bool) -> Router {
    let mut router = routes::routes()
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    if permissive_cors {
        router = router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }
    router
}

/// Bind and serve until the process is asked to stop.
pub async fn serve(nexum: Arc<Nexum>, config: ServerConfig) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(config.addr).await?;
    let bound = listener.local_addr()?;
    tracing::info!(addr = %bound, path = %nexum.path().display(), "nexum serve listening");

    let app = router(ServerState { nexum }, config.permissive_cors);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
}

/// Bind, report the address, and return a future that serves.
///
/// Used by tests and by the desktop app, which need the real port when they
/// asked for port 0.
pub async fn bind(
    nexum: Arc<Nexum>,
    config: ServerConfig,
) -> std::io::Result<(SocketAddr, impl Future<Output = std::io::Result<()>>)> {
    let listener = tokio::net::TcpListener::bind(config.addr).await?;
    let bound = listener.local_addr()?;
    let app = router(ServerState { nexum }, config.permissive_cors);
    Ok((bound, async move { axum::serve(listener, app).await }))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
