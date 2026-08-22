//! The desktop shell.
//!
//! The window does not talk to storage. It starts the same HTTP server that
//! `nexum serve` runs, on a loopback port the OS picks, and the frontend
//! reaches the engine only through that — exactly as the spec requires. The
//! payoff is that there is one API surface to keep correct, and no path by
//! which the viewer could become a second writer against a single-writer
//! database.

use nexum_client::{ClientConfig, Nexum};
use nexum_embed::EmbedderConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{Manager, State};
use tokio::sync::{Mutex, oneshot};

/// Where the frontend should send API requests, plus what is open.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiInfo {
    /// Base URL, e.g. `http://127.0.0.1:53412`.
    pub base_url: String,
    /// Absolute path of the open database.
    pub database: String,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub engine_version: String,
}

/// A running server, and the handle that stops it.
struct RunningServer {
    info: ApiInfo,
    shutdown: oneshot::Sender<()>,
}

#[derive(Default)]
pub struct AppState {
    server: Mutex<Option<RunningServer>>,
}

/// What is currently open, if anything.
#[tauri::command]
async fn current_database(state: State<'_, AppState>) -> Result<Option<ApiInfo>, String> {
    Ok(state.server.lock().await.as_ref().map(|s| s.info.clone()))
}

/// Open a database and (re)start the API server for it.
///
/// Any previously open database is closed first: the engine allows one writer,
/// so leaving the old one open would make the new one fail to acquire its lock.
#[tauri::command]
async fn open_database(
    path: String,
    create: bool,
    state: State<'_, AppState>,
) -> Result<ApiInfo, String> {
    let path = PathBuf::from(path);

    if let Some(previous) = state.server.lock().await.take() {
        // Dropping the sender signals shutdown even if the receiver is gone.
        let _ = previous.shutdown.send(());
        // Give the old server a moment to release the database lock.
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    }

    let exists = path.join("config.json").exists();
    if !exists && !create {
        return Err(format!(
            "No NexumDB database at {}. Choose a database folder, or create a new one.",
            path.display()
        ));
    }

    let config = ClientConfig::default().with_embedder(load_embedder(&path));
    let nexum = if exists {
        Nexum::open(&path, config).await
    } else {
        Nexum::create(&path, config).await
    }
    .map_err(|e| format!("Could not open {}: {e}", path.display()))?;

    let model = nexum.embedder().model_id().to_string();
    let dimensions = nexum.embedder().dim();
    let nexum = Arc::new(nexum);

    // Port 0 lets the OS assign a free one, so two viewer windows — or a
    // viewer alongside `nexum serve` — never fight over a fixed port.
    let (addr, server) = nexum_server::bind(
        nexum,
        nexum_server::ServerConfig {
            addr: ([127, 0, 0, 1], 0).into(),
            permissive_cors: true,
        },
    )
    .await
    .map_err(|e| format!("Could not start the API server: {e}"))?;

    let (shutdown, stop) = oneshot::channel();
    tokio::spawn(async move {
        tokio::select! {
            result = server => {
                if let Err(e) = result {
                    tracing::error!(error = %e, "api server stopped unexpectedly");
                }
            }
            _ = stop => tracing::info!("api server shut down for a database switch"),
        }
    });

    let info = ApiInfo {
        base_url: format!("http://{addr}"),
        database: path.display().to_string(),
        embedding_model: model,
        embedding_dimensions: dimensions,
        engine_version: nexum_core::VERSION.to_string(),
    };

    *state.server.lock().await = Some(RunningServer {
        info: info.clone(),
        shutdown,
    });
    Ok(info)
}

/// Close the open database, releasing its lock.
#[tauri::command]
async fn close_database(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(previous) = state.server.lock().await.take() {
        let _ = previous.shutdown.send(());
    }
    Ok(())
}

/// Databases the user has opened before, most recent first.
#[tauri::command]
fn recent_databases(app: tauri::AppHandle) -> Vec<String> {
    read_recents(&app)
}

/// Record a database in the recents list.
#[tauri::command]
fn remember_database(app: tauri::AppHandle, path: String) -> Vec<String> {
    let mut recents = read_recents(&app);
    recents.retain(|p| p != &path);
    recents.insert(0, path);
    recents.truncate(10);

    if let Some(dir) = recents_path(&app)
        && let Some(parent) = dir.parent()
    {
        let _ = std::fs::create_dir_all(parent);
        let _ = std::fs::write(&dir, serde_json::to_vec_pretty(&recents).unwrap_or_default());
    }
    recents
}

fn recents_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|dir| dir.join("recent-databases.json"))
}

fn read_recents(app: &tauri::AppHandle) -> Vec<String> {
    let Some(path) = recents_path(app) else {
        return Vec::new();
    };
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let recents: Vec<String> = serde_json::from_slice(&bytes).unwrap_or_default();
    // Drop entries the user has since deleted or moved.
    recents
        .into_iter()
        .filter(|p| PathBuf::from(p).join("config.json").exists())
        .collect()
}

/// Read the embedder a database was built with, so queries are embedded into
/// the same vector space the documents were.
fn load_embedder(path: &std::path::Path) -> EmbedderConfig {
    std::fs::read(path.join("embedder.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "nexum=info,nexum_viewer=info".into()),
        )
        .try_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            current_database,
            open_database,
            close_database,
            recent_databases,
            remember_database,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the NexumDB viewer");
}
