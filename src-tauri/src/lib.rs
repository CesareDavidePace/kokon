//! Desktop shell. The UI talks to qls-service over loopback HTTP; this shell only
//! ensures the service is running, keeps a "ui" lease alive while the window is open,
//! and provides native integrations (clipboard images, Explorer).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{Manager, RunEvent};
use tauri_plugin_clipboard_manager::ClipboardExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServiceInfo {
    port: u16,
    token: String,
    pid: u32,
    version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Connection {
    base_url: String,
    token: String,
    version: String,
}

#[derive(Default)]
struct Shared {
    info: Mutex<Option<ServiceInfo>>,
    heartbeat_started: Mutex<bool>,
    connect_lock: Mutex<()>,
}

fn client_id() -> String {
    format!("ui-{}", std::process::id())
}

fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("QLS_DATA_DIR") {
        return PathBuf::from(p);
    }
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("QwenLocalStudio")
}

fn http() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder().no_proxy().timeout(Duration::from_secs(5)).build().expect("http")
}

fn read_info() -> Option<ServiceInfo> {
    serde_json::from_str(&std::fs::read_to_string(data_dir().join("service.json")).ok()?).ok()
}

fn ping(port: u16) -> bool {
    http()
        .get(format!("http://127.0.0.1:{port}/api/v1/ping"))
        .timeout(Duration::from_millis(1500))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn service_exe() -> Option<PathBuf> {
    let me = std::env::current_exe().ok()?;
    let dir = me.parent()?;
    for name in ["qls-service.exe", "qls-service-x86_64-pc-windows-msvc.exe"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn spawn_service() -> Result<(), String> {
    let exe = service_exe().ok_or("qls-service.exe was not found next to the application.")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: survives independently of this window.
        cmd.creation_flags(0x0800_0000 | 0x0000_0200);
    }
    cmd.spawn().map_err(|e| format!("Could not start the Qwen Local Studio service: {e}"))?;
    Ok(())
}

fn ensure_service(shared: &Shared) -> Result<ServiceInfo, String> {
    let _g = shared.connect_lock.lock().unwrap();
    if let Some(i) = read_info()
        && ping(i.port)
    {
        return Ok(i);
    }
    spawn_service()?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(250));
        if let Some(i) = read_info()
            && ping(i.port)
        {
            return Ok(i);
        }
    }
    Err("The Qwen Local Studio service did not start. See %LOCALAPPDATA%\\QwenLocalStudio\\logs\\service.log.".into())
}

fn heartbeat(info: &ServiceInfo) -> bool {
    http()
        .post(format!("http://127.0.0.1:{}/api/v1/session/heartbeat", info.port))
        .bearer_auth(&info.token)
        .json(&serde_json::json!({ "client_id": client_id(), "kind": "ui", "pid": std::process::id() }))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn start_heartbeat(shared: Arc<Shared>) {
    {
        let mut started = shared.heartbeat_started.lock().unwrap();
        if *started {
            return;
        }
        *started = true;
    }
    std::thread::spawn(move || {
        loop {
            let info = shared.info.lock().unwrap().clone();
            if let Some(i) = info
                && !heartbeat(&i)
            {
                // Service gone (crash or manual stop): bring it back.
                if let Ok(n) = ensure_service(&shared) {
                    heartbeat(&n);
                    *shared.info.lock().unwrap() = Some(n);
                }
            }
            std::thread::sleep(Duration::from_secs(15));
        }
    });
}

#[tauri::command]
async fn connect(state: tauri::State<'_, Arc<Shared>>) -> Result<Connection, String> {
    let shared = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let info = ensure_service(&shared)?;
        heartbeat(&info);
        *shared.info.lock().unwrap() = Some(info.clone());
        start_heartbeat(shared.clone());
        Ok(Connection { base_url: format!("http://127.0.0.1:{}/api/v1", info.port), token: info.token, version: info.version })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn copy_image(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let img = tauri::async_runtime::spawn_blocking(move || image::open(&path).map(|i| i.into_rgba8()))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("Could not read the image: {e}"))?;
    let (w, h) = img.dimensions();
    app.clipboard().write_image(&tauri::image::Image::new_owned(img.into_raw(), w, h)).map_err(|e| e.to_string())
}

/// Returns the clipboard image as PNG bytes (empty if the clipboard holds no image).
#[tauri::command]
async fn read_clipboard_image(app: tauri::AppHandle) -> Result<tauri::ipc::Response, String> {
    let Ok(img) = app.clipboard().read_image() else { return Ok(tauri::ipc::Response::new(Vec::<u8>::new())) };
    let (w, h, rgba) = (img.width(), img.height(), img.rgba().to_vec());
    let png = tauri::async_runtime::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let buf = image::RgbaImage::from_raw(w, h, rgba).ok_or("bad clipboard image")?;
        let mut out = Vec::new();
        image::DynamicImage::ImageRgba8(buf).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).map_err(|e| e.to_string())?;
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(tauri::ipc::Response::new(png))
}

#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    tauri_plugin_opener::reveal_item_in_dir(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_folder(app: tauri::AppHandle, path: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err("Folder not found.".into());
    }
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

#[tauri::command]
fn data_folder() -> String {
    data_dir().to_string_lossy().into_owned()
}

fn release(shared: &Shared) {
    if let Some(i) = shared.info.lock().unwrap().clone() {
        let _ = http()
            .post(format!("http://127.0.0.1:{}/api/v1/session/release", i.port))
            .bearer_auth(&i.token)
            .timeout(Duration::from_secs(2))
            .json(&serde_json::json!({ "client_id": client_id(), "kind": "ui" }))
            .send();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shared = Arc::new(Shared::default());
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(shared.clone())
        .invoke_handler(tauri::generate_handler![connect, copy_image, read_clipboard_image, reveal, open_folder, data_folder])
        .build(tauri::generate_context!())
        .expect("error while building Qwen Local Studio");
    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {
            release(&shared);
        }
    });
}
