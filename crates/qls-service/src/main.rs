//! qls-service — the headless Qwen Local Studio service.
//!
//! Owns the generation engine and the sd-server process, and exposes the stable local API
//! (`http://127.0.0.1:<port>/api/v1/...`) used by the desktop app, the MCP server and
//! scripts. Exactly one instance runs (the port bind is the lock). It exits by itself when
//! no client holds a lease and no work is pending, stopping sd-server unless the user
//! chose to keep it running.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qls_core::config::{AppConfig, data_dir};
use qls_core::engine::{APP_VERSION, Engine};
use qls_core::process::ServerState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub port: u16,
    pub token: String,
    pub pid: u32,
    pub version: String,
}

pub struct Lease {
    pub kind: String,
    pub last_seen: Instant,
    /// Client process id; a dead process drops its lease at once (app killed or crashed).
    pub pid: Option<u32>,
}

pub struct AppState {
    pub engine: Arc<Engine>,
    pub token: String,
    pub leases: Mutex<HashMap<String, Lease>>,
    /// When the last lease went away and how long to wait before exiting.
    pub exit_deadline: Mutex<Option<Instant>>,
    pub persistent: bool,
    /// The last client left on purpose (or died): exit right after the work stops instead
    /// of waiting out the idle grace. Cleared by any heartbeat.
    pub quick_exit: std::sync::atomic::AtomicBool,
    pub shutdown: tokio::sync::Notify,
    pub started: Instant,
}

pub const LEASE_TTL: Duration = Duration::from_secs(45);

fn load_or_create_token(dir: &std::path::Path) -> String {
    let p = dir.join("service-token");
    if let Ok(t) = std::fs::read_to_string(&p) {
        let t = t.trim().to_string();
        if t.len() >= 32 && t.chars().all(|c| c.is_ascii_hexdigit()) {
            return t;
        }
    }
    let t = format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
    let _ = std::fs::write(&p, &t);
    t
}

pub fn service_info_path() -> PathBuf {
    data_dir().join("service.json")
}

fn init_logging(dir: &std::path::Path) -> tracing_appender::non_blocking::WorkerGuard {
    let logs = dir.join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let file = tracing_appender::rolling::never(&logs, "service.log");
    let (nb, guard) = tracing_appender::non_blocking(file);
    tracing_subscriber::fmt()
        .with_writer(nb)
        .with_ansi(false)
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("QLS_LOG").unwrap_or_else(|_| "info".into()))
        .init();
    guard
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let persistent = args.iter().any(|a| a == "--persistent");
    let dir = data_dir();
    let _ = std::fs::create_dir_all(&dir);
    // Rotate the service log at startup (keep one previous).
    let log_file = dir.join("logs").join("service.log");
    if std::fs::metadata(&log_file).map(|m| m.len() > 5_000_000).unwrap_or(false) {
        let _ = std::fs::rename(&log_file, dir.join("logs").join("service.prev.log"));
    }
    let _guard = init_logging(&dir);

    let cfg = AppConfig::load();
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(cfg.service_port);

    // Single instance: binding the port is the lock. Loopback only.
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::info!("port {port} busy ({e}); another qls-service is probably running — exiting");
            return;
        }
    };

    let engine = match Engine::new(cfg, &dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("cannot start engine: {e} {:?}", e.detail);
            return;
        }
    };
    let token = load_or_create_token(&dir);
    let info = ServiceInfo { port, token: token.clone(), pid: std::process::id(), version: APP_VERSION.into() };
    let _ = std::fs::write(service_info_path(), serde_json::to_string_pretty(&info).unwrap());
    tracing::info!("qls-service {APP_VERSION} listening on http://{addr} (pid {})", info.pid);

    let state = Arc::new(AppState {
        engine: engine.clone(),
        token,
        leases: Mutex::new(HashMap::new()),
        exit_deadline: Mutex::new(Some(Instant::now() + Duration::from_secs(90))),
        persistent,
        quick_exit: std::sync::atomic::AtomicBool::new(false),
        shutdown: tokio::sync::Notify::new(),
        started: Instant::now(),
    });

    tokio::spawn(engine.clone().run_worker());
    tokio::spawn(lifecycle(state.clone()));
    tokio::spawn(server_watch(state.clone()));

    let app = api::router(state.clone());
    let st = state.clone();
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move { st.shutdown.notified().await });
    if let Err(e) = serve.await {
        tracing::error!("http server error: {e}");
    }

    // Shutdown: stop sd-server unless the user wants it kept loaded.
    let cfg = engine.config();
    if !cfg.keep_server_running {
        if engine.server.stop().await {
            tracing::info!("stopped sd-server on exit");
        }
    } else {
        tracing::info!("leaving sd-server running (keep_server_running)");
    }
    if let Ok(s) = std::fs::read_to_string(service_info_path())
        && let Ok(i) = serde_json::from_str::<ServiceInfo>(&s)
        && i.pid == std::process::id()
    {
        let _ = std::fs::remove_file(service_info_path());
    }
    tracing::info!("qls-service exited");
}

/// Exits when no client holds a lease, no job is pending, and the grace period passed.
/// Also unloads the model after long idle periods when no desktop window is attached.
async fn lifecycle(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let (n_leases, has_ui, lost) = {
            let mut leases = state.leases.lock().unwrap();
            let mut lost = Vec::new();
            leases.retain(|id, l| {
                let alive = l.last_seen.elapsed() < LEASE_TTL && l.pid.map(qls_core::process::pid_alive).unwrap_or(true);
                if !alive {
                    lost.push((id.clone(), l.kind.clone()));
                }
                alive
            });
            (leases.len(), leases.values().any(|l| l.kind == "ui"), lost)
        };
        for (id, kind) in &lost {
            tracing::info!("client {id} ({kind}) is gone");
            client_left(&state, kind, has_ui, n_leases);
        }
        let busy = state.engine.has_pending_work();
        if n_leases > 0 || busy {
            *state.exit_deadline.lock().unwrap() = None;
        } else if !state.persistent {
            // Nobody is left to use the model: free the GPU now, exit after the grace.
            let st = state.engine.server.status();
            if st.owned && st.state != ServerState::Stopped && !state.engine.config().keep_server_running {
                tracing::info!("no clients — unloading model");
                state.engine.server.stop().await;
                state.engine.emit_server();
            }
            let grace = if state.quick_exit.load(std::sync::atomic::Ordering::Relaxed) { 3 } else { 60 };
            let mut d = state.exit_deadline.lock().unwrap();
            match *d {
                None => *d = Some(Instant::now() + Duration::from_secs(grace)),
                Some(t) if Instant::now() >= t => {
                    tracing::info!("no clients and no work — shutting down");
                    state.shutdown.notify_waiters();
                    return;
                }
                _ => {}
            }
        }
        // Idle unload (MCP-only or scripted use): free VRAM after N idle minutes.
        let cfg = state.engine.config();
        let st = state.engine.server.status();
        if !has_ui
            && !busy
            && cfg.idle_unload_minutes > 0
            && st.state == ServerState::Ready
            && st.owned
            && state.engine.idle_for() > Duration::from_secs(cfg.idle_unload_minutes as u64 * 60)
        {
            tracing::info!("idle for {} min without the desktop app — unloading model", cfg.idle_unload_minutes);
            state.engine.server.stop().await;
            state.engine.emit_server();
        }
    }
}

/// A client released its lease or died. When the last desktop window goes, its unfinished
/// jobs are cancelled (MCP jobs keep running); when the last client goes, the service exits
/// as soon as the work stops.
pub fn client_left(state: &AppState, kind: &str, ui_remaining: bool, remaining: usize) {
    if kind == "ui" && !ui_remaining {
        let n = state.engine.cancel_client_jobs("ui");
        if n > 0 {
            tracing::info!("desktop app closed — cancelled {n} job(s)");
        }
    }
    if remaining == 0 {
        state.quick_exit.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Detects an owned sd-server that died between jobs, so the UI shows the right state.
async fn server_watch(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let st = state.engine.server.status();
        if st.state == ServerState::Ready && st.owned && state.engine.server.owned_alive() == Some(false) {
            state.engine.server.mark_lost(qls_core::error::AppError::new(
                qls_core::error::ErrorCode::ServerCrashed,
                "The inference server stopped unexpectedly.",
            ));
            state.engine.emit_server();
        }
    }
}
