//! Security tests for access from other devices, through the real router: who gets in on each
//! listener, the attempt limit, what a signed-in browser can and cannot reach, and sessions
//! ending with a new password or a logout. No GPU, no network port opened.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use axum::{
    Router,
    body::Body,
    extract::connect_info::MockConnectInfo,
    http::{HeaderMap, Request, StatusCode, header},
};
use qls_core::{config::AppConfig, engine::Engine};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::{AppState, api, remote::RemoteState};

const PASSWORD: &str = "home server 42";

/// Settings are saved under `QLS_CONFIG_DIR`: a temporary folder, never the user's.
fn isolate_settings() {
    static DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = tempfile::tempdir().unwrap();
        // SAFETY: set once, before any test reads it (every test calls this first).
        unsafe { std::env::set_var("QLS_CONFIG_DIR", d.path()) };
        d
    });
}

struct Fixture {
    _dir: tempfile::TempDir,
    state: Arc<AppState>,
}

fn fixture(with_password: bool) -> Fixture {
    isolate_settings();
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(AppConfig::default(), dir.path()).unwrap();
    let state = Arc::new(AppState {
        engine,
        token: "t".repeat(64),
        leases: Mutex::new(HashMap::new()),
        exit_deadline: Mutex::new(None),
        persistent: true,
        quick_exit: std::sync::atomic::AtomicBool::new(false),
        shutdown: tokio::sync::Notify::new(),
        started: Instant::now(),
        remote: RemoteState::new(dir.path(), false),
    });
    if with_password {
        state.remote.set_password(PASSWORD).unwrap();
    }
    Fixture { _dir: dir, state }
}

fn app(f: &Fixture, lan: bool, ip: &str) -> Router {
    let addr: SocketAddr = format!("{ip}:50000").parse().unwrap();
    api::router(f.state.clone(), lan).layer(MockConnectInfo(addr))
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Value,
}

impl Reply {
    fn code(&self) -> &str {
        self.body["error"]["code"].as_str().unwrap_or("")
    }
}

async fn send(app: &Router, method: &str, path: &str, headers: &[(&str, &str)], body: Option<Value>) -> Reply {
    let mut req = Request::builder().method(method).uri(path);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let req = match body {
        Some(b) => req.header(header::CONTENT_TYPE, "application/json").body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    }
    .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let bytes = axum::body::to_bytes(res.into_body(), 16 << 20).await.unwrap();
    Reply { status, headers, body: serde_json::from_slice(&bytes).unwrap_or(Value::Null) }
}

const CSRF: (&str, &str) = ("x-kokon-csrf", "1");

async fn login(app: &Router, password: &str) -> Reply {
    send(app, "POST", "/api/v1/auth/login", &[CSRF], Some(json!({ "password": password }))).await
}

/// The `Cookie` header value for a successful login.
async fn sign_in(app: &Router) -> String {
    let r = login(app, PASSWORD).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.body);
    let set = r.headers[header::SET_COOKIE].to_str().unwrap();
    for attr in ["HttpOnly", "SameSite=Strict", "Path=/"] {
        assert!(set.contains(attr), "cookie lacks {attr}: {set}");
    }
    set.split(';').next().unwrap().to_string()
}

#[tokio::test]
async fn nothing_without_credentials() {
    let f = fixture(true);
    for lan in [false, true] {
        let a = app(&f, lan, "192.168.1.50");
        for (m, p) in [("GET", "/api/v1/status"), ("GET", "/api/v1/gallery"), ("GET", "/api/v1/events"), ("POST", "/api/v1/jobs")] {
            let r = send(&a, m, p, &[], None).await;
            assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{m} {p} lan={lan}");
            assert_eq!(r.code(), "login_required");
        }
        let r = send(&a, "GET", "/api/v1/gallery/x/image?token=", &[], None).await;
        assert_eq!(r.status, StatusCode::UNAUTHORIZED);
        // A cookie that no login handed out.
        let r = send(&a, "GET", "/api/v1/status", &[("cookie", "kokon_session=0123456789abcdef")], None).await;
        assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    }
}

#[tokio::test]
async fn the_token_works_on_loopback_only() {
    let f = fixture(true);
    let auth = format!("Bearer {}", f.state.token);
    let local = app(&f, false, "127.0.0.1");
    assert_eq!(send(&local, "GET", "/api/v1/remote", &[("authorization", &auth)], None).await.status, StatusCode::OK);
    let wrong = send(&local, "GET", "/api/v1/status", &[("authorization", "Bearer nope")], None).await;
    assert_eq!((wrong.status, wrong.code()), (StatusCode::UNAUTHORIZED, "unauthorized"));
    // From a web page in a browser, even with the token.
    let page = send(&local, "GET", "/api/v1/status", &[("authorization", &auth), ("origin", "http://evil.example")], None).await;
    assert_eq!(page.status, StatusCode::UNAUTHORIZED);
    // The network listener never takes it.
    let lan = app(&f, true, "192.168.1.50");
    let r = send(&lan, "GET", "/api/v1/status", &[("authorization", &auth)], None).await;
    assert_eq!((r.status, r.code()), (StatusCode::UNAUTHORIZED, "login_required"));
    let r = send(&lan, "GET", &format!("/api/v1/status?token={}", f.state.token), &[], None).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn no_login_without_a_password() {
    let f = fixture(false);
    let a = app(&f, true, "192.168.1.50");
    let r = login(&a, "anything at all").await;
    assert_eq!((r.status, r.code()), (StatusCode::FORBIDDEN, "forbidden"));
    let s = send(&a, "GET", "/api/v1/auth/session", &[], None).await;
    assert_eq!(s.body, json!({ "signed_in": false, "password_set": false }));
}

#[tokio::test]
async fn login_needs_the_csrf_header() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.50");
    let r = send(&a, "POST", "/api/v1/auth/login", &[], Some(json!({ "password": PASSWORD }))).await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    assert!(!r.headers.contains_key(header::SET_COOKIE));
}

#[tokio::test]
async fn wrong_passwords_lock_the_address() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.66");
    for _ in 0..4 {
        let r = login(&a, "guess guess").await;
        assert_eq!((r.status, r.code()), (StatusCode::UNAUTHORIZED, "login_required"));
    }
    let r = login(&a, "guess guess").await;
    assert_eq!((r.status, r.code()), (StatusCode::TOO_MANY_REQUESTS, "too_many_attempts"));
    assert_eq!(r.body["retry_after_s"], 60);
    assert_eq!(r.headers[header::RETRY_AFTER], "60");
    // Locked: even the right password gets the same answer, and no cookie.
    let r = login(&a, PASSWORD).await;
    assert_eq!(r.status, StatusCode::TOO_MANY_REQUESTS);
    assert!(!r.headers.contains_key(header::SET_COOKIE));
    // Another device on the network is not affected.
    sign_in(&app(&f, true, "192.168.1.67")).await;
}

#[tokio::test]
async fn a_browser_gets_the_ui_api_but_no_paths_on_the_server() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.50");
    let cookie = sign_in(&a).await;
    let c = ("cookie", cookie.as_str());
    assert_eq!(send(&a, "GET", "/api/v1/status", &[c], None).await.status, StatusCode::OK);
    assert_eq!(send(&a, "GET", "/api/v1/gallery", &[c], None).await.status, StatusCode::OK);
    let s = send(&a, "GET", "/api/v1/auth/session", &[c], None).await;
    assert_eq!(s.body, json!({ "signed_in": true, "password_set": true }));
    for (m, p, body) in [
        ("POST", "/api/v1/inputs/import", json!({ "path": "/etc/passwd" })),
        ("POST", "/api/v1/models/scan", json!({ "dir": "/" })),
        ("POST", "/api/v1/models/inspect", json!({ "path": "/etc/shadow" })),
        ("POST", "/api/v1/gallery/x/export", json!({ "path": "/tmp/x.png" })),
        ("POST", "/api/v1/image-info", json!({ "path": "/etc/passwd" })),
        ("POST", "/api/v1/loras", json!({ "path": "/tmp/x.safetensors" })),
        ("POST", "/api/v1/luts/import", json!({ "path": "/tmp/x.cube" })),
        ("POST", "/api/v1/generate", json!({ "prompt": "x", "output_path": "/tmp/x.png" })),
        ("POST", "/api/v1/shutdown", json!({})),
        ("POST", "/api/v1/remote/password", json!({ "password": "taken over!" })),
        ("POST", "/api/v1/remote", json!({ "enabled": true })),
        ("GET", "/api/v1/remote", Value::Null),
    ] {
        let body = (!body.is_null()).then_some(body);
        let r = send(&a, m, p, &[c, CSRF], body).await;
        assert_eq!((r.status, r.code()), (StatusCode::FORBIDDEN, "forbidden"), "{m} {p}");
    }
    // A job may only use images already in Kokon, and saves to the gallery.
    for job in [json!({ "prompt": "x", "mode": "edit", "images": ["/etc/passwd"] }), json!({ "prompt": "x", "output_path": "/tmp/x.png" })] {
        let r = send(&a, "POST", "/api/v1/jobs", &[c, CSRF], Some(job)).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN);
        let r = send(&a, "POST", "/api/v1/plan", &[c, CSRF], Some(json!({ "prompt": "x", "mode": "edit", "images": ["C:\\Users\\me\\x.png"] }))).await;
        assert_eq!(r.status, StatusCode::FORBIDDEN);
    }
    // The password is never in the settings the API returns.
    let cfg = send(&a, "GET", "/api/v1/config", &[c], None).await;
    assert!(!cfg.body.to_string().contains("argon2"));
}

#[tokio::test]
async fn browser_changes_need_the_csrf_header() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.50");
    let cookie = sign_in(&a).await;
    let c = ("cookie", cookie.as_str());
    let beat = json!({ "client_id": "tab-1" });
    let r = send(&a, "POST", "/api/v1/session/heartbeat", &[c], Some(beat.clone())).await;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    let r = send(&a, "POST", "/api/v1/session/heartbeat", &[c, CSRF], Some(beat)).await;
    assert_eq!(r.status, StatusCode::OK);
    // A browser's lease is never the desktop window's, and never kept alive by a pid.
    let l = f.state.leases.lock().unwrap();
    assert_eq!((l["tab-1"].kind.as_str(), l["tab-1"].pid), ("web", None));
}

#[tokio::test]
async fn browser_settings_saves_keep_the_computers_settings() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.50");
    let cookie = sign_in(&a).await;
    let c = ("cookie", cookie.as_str());
    let before = f.state.engine.config();
    let mut sent = serde_json::to_value(&before).unwrap();
    sent["sdcppDir"] = json!("/tmp/elsewhere");
    sent["extraServerArgs"] = json!(["--rpc", "evil:50052"]);
    sent["remoteAccess"] = json!(true);
    sent["flashAttention"] = json!(!before.flash_attention);
    let r = send(&a, "PUT", "/api/v1/config", &[c, CSRF], Some(sent)).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.body);
    let after = f.state.engine.config();
    assert_eq!(after.sdcpp_dir, before.sdcpp_dir);
    assert_eq!(after.extra_server_args, before.extra_server_args);
    assert!(!after.remote_access);
    assert_eq!(after.flash_attention, !before.flash_attention, "an app setting does change");
}

#[tokio::test]
async fn a_new_password_or_a_logout_ends_sessions() {
    let f = fixture(true);
    let a = app(&f, true, "192.168.1.50");
    let phone = sign_in(&a).await;
    let laptop = sign_in(&a).await;
    let r = send(&a, "POST", "/api/v1/auth/logout", &[("cookie", &laptop), CSRF], None).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.headers[header::SET_COOKIE].to_str().unwrap().contains("Max-Age=0"));
    assert_eq!(send(&a, "GET", "/api/v1/status", &[("cookie", &laptop)], None).await.status, StatusCode::UNAUTHORIZED);
    assert_eq!(send(&a, "GET", "/api/v1/status", &[("cookie", &phone)], None).await.status, StatusCode::OK);
    // Changed from the desktop app (token, loopback).
    let local = app(&f, false, "127.0.0.1");
    let auth = format!("Bearer {}", f.state.token);
    let r = send(&local, "POST", "/api/v1/remote/password", &[("authorization", &auth)], Some(json!({ "password": "a brand new one" }))).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.body);
    assert_eq!(send(&a, "GET", "/api/v1/status", &[("cookie", &phone)], None).await.status, StatusCode::UNAUTHORIZED);
    assert_eq!(login(&a, PASSWORD).await.status, StatusCode::UNAUTHORIZED);
    assert_eq!(login(&a, "a brand new one").await.status, StatusCode::OK);
    let short = send(&local, "POST", "/api/v1/remote/password", &[("authorization", &auth)], Some(json!({ "password": "short" }))).await;
    assert_eq!(short.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn the_network_stays_closed_without_a_password() {
    let f = fixture(false);
    let local = app(&f, false, "127.0.0.1");
    let auth = format!("Bearer {}", f.state.token);
    let r = send(&local, "POST", "/api/v1/remote", &[("authorization", &auth)], Some(json!({ "enabled": true }))).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let st = send(&local, "GET", "/api/v1/remote", &[("authorization", &auth)], None).await.body;
    assert_eq!((st["enabled"].as_bool(), st["listening"].as_bool(), st["password_set"].as_bool()), (Some(false), Some(false), Some(false)));
    // Removing the password turns access off.
    f.state.engine.set_remote(true, f.state.engine.config().remote_port).unwrap();
    let r = send(&local, "DELETE", "/api/v1/remote/password", &[("authorization", &auth)], None).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(!f.state.engine.config().remote_access);
    assert_eq!(r.body["listening"], false);
}
