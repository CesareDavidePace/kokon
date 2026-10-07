//! Environment inspection: GPU, sd-server version/devices, first-launch readiness checks
//! and the copyable diagnostics report.

use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use crate::config::AppConfig;
use crate::process::{parse_device_list, run_server_probe};

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuInfo {
    pub name: String,
    pub memory_total_mib: u64,
    pub memory_used_mib: u64,
    pub memory_free_mib: u64,
    pub driver_version: String,
}

/// Queries NVIDIA GPUs through nvidia-smi (ships with the driver). None if unavailable.
pub fn query_gpus() -> Option<Vec<GpuInfo>> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args(["--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let gpus: Vec<GpuInfo> = text.lines().filter_map(parse_gpu_line).collect();
    (!gpus.is_empty()).then_some(gpus)
}

pub fn parse_gpu_line(line: &str) -> Option<GpuInfo> {
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 5 {
        return None;
    }
    Some(GpuInfo {
        name: f[0].to_string(),
        memory_total_mib: f[1].parse().ok()?,
        memory_used_mib: f[2].parse().ok()?,
        memory_free_mib: f[3].parse().ok()?,
        driver_version: f[4].to_string(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub id: &'static str,
    pub label: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupReport {
    pub ready: bool,
    pub checks: Vec<Check>,
    pub devices: Vec<(String, String)>,
    pub backend_version: Option<String>,
}

pub fn backend_version(exe: &Path) -> Option<String> {
    run_server_probe(exe, &["--help"], Duration::from_secs(20)).and_then(|t| crate::sdcpp::parse_version_banner(&t))
}

pub fn list_devices(exe: &Path) -> Vec<(String, String)> {
    run_server_probe(exe, &["--list-devices"], Duration::from_secs(30)).map(|t| parse_device_list(&t)).unwrap_or_default()
}

fn fmt_size(bytes: u64) -> String {
    if bytes >= 1 << 30 { format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64) } else { format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64) }
}

/// First-launch / settings checks. `server_ok` is the live health result if known.
pub fn setup_checks(cfg: &AppConfig, server_ok: Option<bool>) -> SetupReport {
    let mut checks = Vec::new();
    let dir_ok = cfg.sdcpp_dir.is_dir();
    checks.push(Check { id: "sdcpp_dir", label: "stable-diffusion.cpp folder".into(), ok: dir_ok, detail: cfg.sdcpp_dir.display().to_string() });
    let exe = cfg.server_exe();
    let exe_ok = exe.is_file();
    checks.push(Check { id: "server_exe", label: "sd-server.exe".into(), ok: exe_ok, detail: exe.display().to_string() });
    for (label, path, _required) in cfg.profile().files() {
        let meta = std::fs::metadata(path).ok().filter(|m| m.is_file());
        checks.push(Check {
            id: "model_file",
            label: label.to_string(),
            ok: meta.is_some(),
            detail: match &meta {
                Some(m) => format!("{} ({})", path.display(), fmt_size(m.len())),
                None => format!("Missing: {}", path.display()),
            },
        });
    }
    let devices = if exe_ok { list_devices(&exe) } else { vec![] };
    let wanted = cfg.device.to_ascii_lowercase();
    let dev = devices.iter().find(|(n, _)| n.to_ascii_lowercase() == wanted);
    let gpu_ok = dev.is_some();
    checks.push(Check {
        id: "device",
        label: "GPU device".into(),
        ok: gpu_ok,
        detail: match dev {
            Some((n, d)) => format!("{n} — {d}"),
            None if devices.is_empty() => "No devices reported by sd-server".into(),
            None => format!("{} not found. Available: {}", cfg.device, devices.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")),
        },
    });
    if let Some(running) = server_ok {
        // Informational: the server is started on demand, so "not running" is not a problem.
        checks.push(Check {
            id: "server",
            label: "Inference server".into(),
            ok: true,
            detail: if running { "Running".into() } else { "Not running yet — starts automatically".into() },
        });
    }
    checks.push(upscaler_check(cfg));
    let backend_version = if exe_ok { backend_version(&exe) } else { None };
    // The server and the ESRGAN model are optional: neither blocks starting.
    let ready = checks.iter().filter(|c| c.id != "server" && c.id != "upscaler").all(|c| c.ok);
    SetupReport { ready, checks, devices, backend_version }
}

/// ESRGAN model for sharper enlargements. Optional: without it Lanczos is used. The app
/// never downloads models, so the check explains where to get it.
pub fn upscaler_check(cfg: &AppConfig) -> Check {
    let dir = cfg.upscalers_dir();
    match cfg.upscaler_file() {
        Some(p) => {
            let size = std::fs::metadata(&p).map(|m| fmt_size(m.len())).unwrap_or_default();
            Check { id: "upscaler", label: "ESRGAN upscaler (optional)".into(), ok: true, detail: format!("{} ({size})", p.display()) }
        }
        None => Check {
            id: "upscaler",
            label: "ESRGAN upscaler (optional)".into(),
            ok: false,
            detail: {
                let name = cfg.upscaler_model.trim();
                let get = crate::config::upscaler_download_url(name)
                    .map(|u| format!(" Download {u} (64 MB) into that folder for sharper results."))
                    .unwrap_or_default();
                format!("Not found: {}\\{name}.pth — enlargements use Lanczos.{get}", dir.display())
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_line_parsing() {
        let g = parse_gpu_line("NVIDIA GeForce RTX 5070 Ti, 16303, 2312, 13991, 617.14").unwrap();
        assert_eq!(g.name, "NVIDIA GeForce RTX 5070 Ti");
        assert_eq!((g.memory_total_mib, g.memory_used_mib, g.memory_free_mib), (16303, 2312, 13991));
        assert_eq!(g.driver_version, "617.14");
        assert!(parse_gpu_line("garbage").is_none());
    }

    #[test]
    fn checks_flag_missing_files() {
        let mut cfg = AppConfig::default();
        cfg.sdcpp_dir = std::env::temp_dir().join("qls-does-not-exist");
        cfg.profiles = vec![crate::config::ModelProfile::qwen_image_21_q4(&cfg.sdcpp_dir)];
        let r = setup_checks(&cfg, None);
        assert!(!r.ready);
        assert!(r.checks.iter().any(|c| c.id == "server_exe" && !c.ok));
        assert_eq!(r.checks.iter().filter(|c| c.id == "model_file").count(), 4);
        let up = r.checks.iter().find(|c| c.id == "upscaler").unwrap();
        assert!(!up.ok && up.detail.contains("4x-UltraSharp.pth") && up.detail.contains("huggingface.co/uwg/upscaler"));
    }

    #[test]
    fn upscaler_check_is_optional() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = AppConfig::default();
        cfg.upscalers_dir = Some(dir.path().to_path_buf());
        assert!(!upscaler_check(&cfg).ok);
        std::fs::write(dir.path().join("4x-UltraSharp.pth"), vec![0u8; 2048]).unwrap();
        let c = upscaler_check(&cfg);
        assert!(c.ok && c.detail.contains("4x-UltraSharp.pth"));
    }
}
