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
    /// "nvidia", "amd", "intel", "apple" or "other"; picks the engine (`GpuBackend::for_vendor`).
    pub vendor: String,
    /// Apple Silicon: `memory_total_mib` is unified memory shared with the system.
    pub unified: bool,
    pub memory_total_mib: u64,
    pub memory_used_mib: u64,
    pub memory_free_mib: u64,
    pub driver_version: String,
}

impl GpuInfo {
    /// Memory the planner may use for the diffusion model, as on a discrete card. Apple
    /// Silicon shares it with the system and the text encoder
    /// (`planner::unified_memory_planning_mib`).
    pub fn planning_mib(&self) -> u64 {
        if self.unified {
            use crate::catalog::{TEXT_ENCODER_Q4_K_M, VISION_PROJECTOR};
            let te = (TEXT_ENCODER_Q4_K_M.size + VISION_PROJECTOR.size) / (1024 * 1024);
            crate::planner::unified_memory_planning_mib(self.memory_total_mib, te)
        } else {
            self.memory_total_mib
        }
    }
}

/// The GPUs to plan for: NVIDIA through nvidia-smi (live memory use), otherwise any
/// graphics card Windows knows (AMD, Intel), or the Apple Silicon GPU. None if there is none.
pub fn query_gpus() -> Option<Vec<GpuInfo>> {
    query_nvidia().or_else(query_dxgi).or_else(query_apple)
}

/// Card and engine device names refer to the same GPU ("NVIDIA GeForce RTX 3090" from
/// nvidia-smi, DXGI or ggml; one may carry a suffix).
fn same_gpu(card: &str, device: &str) -> bool {
    let (a, b) = (card.trim().to_ascii_lowercase(), device.trim().to_ascii_lowercase());
    !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
}

/// Engine devices (`--list-devices`) of this engine's kind, in their order.
fn engine_gpus<'a>(devices: &'a [(String, String)], prefix: &str) -> impl Iterator<Item = &'a (String, String)> {
    let prefix = prefix.to_string();
    devices.iter().filter(move |(n, _)| n.to_ascii_lowercase().starts_with(&prefix))
}

/// The engine device (e.g. "cuda1") of card `card` in `cards`. Matched by name, not by
/// position: nvidia-smi, DXGI (sorted by VRAM) and the engine each order cards their own
/// way. The k-th card of a name maps to the k-th engine device of that name.
pub fn device_for_card(devices: &[(String, String)], cards: &[GpuInfo], card: usize, prefix: &str) -> Option<String> {
    let name = &cards.get(card)?.name;
    let k = cards[..card].iter().filter(|c| c.name.eq_ignore_ascii_case(name)).count();
    engine_gpus(devices, prefix).filter(|(_, d)| same_gpu(name, d)).nth(k).map(|(n, _)| n.to_ascii_lowercase())
}

/// The card in `cards` behind engine device `device` (inverse of `device_for_card`).
pub fn card_for_device(devices: &[(String, String)], cards: &[GpuInfo], device: &str, prefix: &str) -> Option<usize> {
    let gpus: Vec<_> = engine_gpus(devices, prefix).collect();
    let i = gpus.iter().position(|(n, _)| n.eq_ignore_ascii_case(device))?;
    let desc = &gpus[i].1;
    let k = gpus[..i].iter().filter(|(_, d)| d.eq_ignore_ascii_case(desc)).count();
    cards.iter().enumerate().filter(|(_, c)| same_gpu(&c.name, desc)).nth(k).map(|(j, _)| j)
}

/// Apple Silicon: chip name and unified memory from sysctl (they never change, so read once).
#[cfg(target_os = "macos")]
fn query_apple() -> Option<Vec<GpuInfo>> {
    static INFO: std::sync::OnceLock<Option<GpuInfo>> = std::sync::OnceLock::new();
    INFO.get_or_init(|| {
        let read = |key: &str| {
            let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", key]).output().ok()?;
            out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        };
        if read("hw.optional.arm64").as_deref() != Some("1") {
            return None; // Intel Mac: not supported
        }
        let total = read("hw.memsize")?.parse::<u64>().ok()? / (1024 * 1024);
        let name = read("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".into());
        Some(GpuInfo { name, vendor: "apple".into(), unified: true, memory_total_mib: total, memory_used_mib: 0, memory_free_mib: total, driver_version: String::new() })
    })
    .clone()
    .map(|g| vec![g])
}

#[cfg(not(target_os = "macos"))]
fn query_apple() -> Option<Vec<GpuInfo>> {
    None
}

/// NVIDIA GPUs through nvidia-smi (ships with the driver). None if unavailable.
fn query_nvidia() -> Option<Vec<GpuInfo>> {
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
        vendor: "nvidia".into(),
        unified: false,
        memory_total_mib: f[1].parse().ok()?,
        memory_used_mib: f[2].parse().ok()?,
        memory_free_mib: f[3].parse().ok()?,
        driver_version: f[4].to_string(),
    })
}

/// PCI vendor id → vendor name used across the app.
pub fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        _ => "other",
    }
}

/// Graphics adapters through DXGI, largest dedicated VRAM first. Skips software adapters
/// (Microsoft Basic Render Driver). DXGI has no system-wide usage figure, so used = 0.
/// Integrated GPUs report only their small reserved memory and so fall under the
/// smallest preset's warning.
#[cfg(windows)]
fn query_dxgi() -> Option<Vec<GpuInfo>> {
    use windows::Win32::Graphics::Dxgi::*;
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    let mut gpus = Vec::new();
    for i in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(i) }) else { break };
        let Ok(d) = (unsafe { adapter.GetDesc1() }) else { continue };
        if d.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 || d.VendorId == 0x1414 {
            continue;
        }
        let len = d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len());
        let total = d.DedicatedVideoMemory as u64 / (1024 * 1024);
        gpus.push(GpuInfo {
            name: String::from_utf16_lossy(&d.Description[..len]).trim().to_string(),
            vendor: vendor_name(d.VendorId).into(),
            unified: false,
            memory_total_mib: total,
            memory_used_mib: 0,
            memory_free_mib: total,
            driver_version: String::new(),
        });
    }
    gpus.sort_by(|a, b| b.memory_total_mib.cmp(&a.memory_total_mib));
    (!gpus.is_empty()).then_some(gpus)
}

#[cfg(not(windows))]
fn query_dxgi() -> Option<Vec<GpuInfo>> {
    None
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
    checks.push(Check { id: "server_exe", label: "stable-diffusion.cpp server".into(), ok: exe_ok, detail: exe.display().to_string() });
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
    let wanted = cfg.effective_device();
    let dev = devices.iter().find(|(n, _)| n.to_ascii_lowercase() == wanted);
    let gpu_ok = dev.is_some();
    checks.push(Check {
        id: "device",
        label: "GPU device".into(),
        ok: gpu_ok,
        detail: match dev {
            Some((n, d)) => format!("{n} — {d}"),
            None if devices.is_empty() => "No devices reported by sd-server".into(),
            None => format!("{wanted} not found. Available: {}", devices.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")),
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
    if exe_ok {
        checks.push(engine_release_check(cfg));
    }
    let backend_version = if exe_ok { backend_version(&exe) } else { None };
    // Informational: the server starts on demand; an engine of an older release still
    // generates (enlargements fall back to Lanczos), and Setup offers the update.
    let ready = checks.iter().filter(|c| c.id != "server" && c.id != "engine_release").all(|c| c.ok);
    SetupReport { ready, checks, devices, backend_version }
}

/// The upscaler model (MoSR): enlarges results and runs Upscale. Shipped with the app and
/// otherwise downloaded by Setup.
pub fn upscaler_check(cfg: &AppConfig) -> Check {
    let label = "Upscaler".to_string();
    let chosen = cfg.upscaler_model.trim();
    if chosen == crate::config::LANCZOS {
        return Check { id: "upscaler", label, ok: true, detail: "Lanczos (plain resize, no model)".into() };
    }
    match cfg.upscaler() {
        Some((name, p)) => {
            let size = std::fs::metadata(&p).map(|m| fmt_size(m.len())).unwrap_or_default();
            let fallback = if name != chosen { format!("{chosen} not found in {}; using ", cfg.upscalers_dir().display()) } else { String::new() };
            Check { id: "upscaler", label, ok: true, detail: format!("{fallback}{} ({size})", p.display()) }
        }
        None => Check { id: "upscaler", label, ok: false, detail: format!("Missing: {} — install it from Setup.", crate::catalog::UPSCALER.path(&cfg.sdcpp_dir).display()) },
    }
}

/// Whether the installed engine is the catalog's release (needed for the upscaler).
pub fn engine_release_check(cfg: &AppConfig) -> Check {
    let engines: Vec<_> = crate::catalog::engine_downloads(cfg.gpu_backend).into_iter().filter(|d| d.engine_tag.is_some()).collect();
    let tags: Vec<Option<String>> = engines.iter().map(|d| d.installed_tag(&cfg.sdcpp_dir)).collect();
    let want = crate::catalog::engine_tag();
    let ok = tags.iter().all(|t| t.as_deref() == Some(want));
    Check {
        id: "engine_release",
        label: "Engine release".into(),
        ok,
        detail: if ok {
            want.into()
        } else {
            format!("An older engine is installed; Setup updates it to {want}.")
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
        assert_eq!(g.vendor, "nvidia");
        assert_eq!(g.planning_mib(), 16303);
        let mac = GpuInfo { name: "Apple M2 Pro".into(), vendor: "apple".into(), unified: true, memory_total_mib: 16384, ..Default::default() };
        assert!(mac.planning_mib() < 6 * 1024, "16 GB Mac plans like a small card");
        assert_eq!(crate::catalog::recommended_preset(Some(mac.planning_mib())).id, "12gb");
    }

    #[test]
    fn cards_map_to_engine_devices_by_name() {
        let card = |n: &str, mib| GpuInfo { name: n.into(), memory_total_mib: mib, ..Default::default() };
        // DXGI sorts by VRAM; the engine lists the 3060 first.
        let cards = vec![card("NVIDIA GeForce RTX 3090", 24576), card("NVIDIA GeForce RTX 3060", 12288), card("NVIDIA GeForce RTX 3060", 12288)];
        let devs: Vec<(String, String)> = [("CUDA0", "NVIDIA GeForce RTX 3060"), ("CUDA1", "NVIDIA GeForce RTX 3090"), ("CUDA2", "NVIDIA GeForce RTX 3060"), ("CPU", "AMD Ryzen 9")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        assert_eq!(device_for_card(&devs, &cards, 0, "cuda").as_deref(), Some("cuda1"));
        assert_eq!(device_for_card(&devs, &cards, 1, "cuda").as_deref(), Some("cuda0"));
        assert_eq!(device_for_card(&devs, &cards, 2, "cuda").as_deref(), Some("cuda2"));
        assert_eq!(device_for_card(&devs, &cards, 0, "vulkan"), None, "another engine's devices");
        for i in 0..3 {
            let dev = device_for_card(&devs, &cards, i, "cuda").unwrap();
            assert_eq!(card_for_device(&devs, &cards, &dev, "cuda"), Some(i));
        }
        assert_eq!(card_for_device(&devs, &cards, "cuda7", "cuda"), None);
    }

    #[test]
    fn dxgi_reports_real_adapters() {
        assert_eq!((vendor_name(0x10DE), vendor_name(0x1002), vendor_name(0x8086), vendor_name(0x5143)), ("nvidia", "amd", "intel", "other"));
        // Machines without a GPU (CI) report none; otherwise every entry is a real card.
        for g in query_dxgi().unwrap_or_default() {
            println!("DXGI: {} ({}) {} MiB", g.name, g.vendor, g.memory_total_mib);
            assert!(!g.name.is_empty() && !g.name.contains("Basic Render"));
        }
    }

    #[test]
    fn checks_flag_missing_files() {
        let mut cfg = AppConfig::default();
        cfg.sdcpp_dir = std::env::temp_dir().join("qls-does-not-exist");
        cfg.upscaler_model = crate::config::UPSCALER_CHOICES[0].0.into(); // the PC default (Macs: Lanczos)
        cfg.profiles = vec![crate::config::ModelProfile::official(&cfg.sdcpp_dir, cfg.preset())];
        let r = setup_checks(&cfg, None);
        assert!(!r.ready);
        assert!(r.checks.iter().any(|c| c.id == "server_exe" && !c.ok));
        assert_eq!(r.checks.iter().filter(|c| c.id == "model_file").count(), 4);
        let up = r.checks.iter().find(|c| c.id == "upscaler").unwrap();
        assert!(!up.ok && up.detail.contains("4xNomos2_hq_mosr.pth"));
        assert!(!r.checks.iter().any(|c| c.id == "engine_release"), "no engine, nothing to compare");
    }

    #[test]
    fn upscaler_and_engine_release_checks() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = AppConfig::default();
        cfg.sdcpp_dir = dir.path().to_path_buf();
        cfg.gpu_backend = crate::config::GpuBackend::Cuda;
        cfg.upscaler_model = crate::config::UPSCALER_CHOICES[0].0.into(); // the PC default (Macs: Lanczos)
        assert!(!upscaler_check(&cfg).ok, "required");
        let up = crate::catalog::UPSCALER;
        std::fs::create_dir_all(up.path(dir.path()).parent().unwrap()).unwrap();
        std::fs::File::create(up.path(dir.path())).unwrap().set_len(up.size).unwrap();
        let c = upscaler_check(&cfg);
        assert!(c.ok && c.detail.contains("4xNomos2_hq_mosr.pth"));
        let _guard = crate::catalog::ENGINE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let e = engine_release_check(&cfg);
        assert!(!e.ok && e.detail.contains(crate::catalog::ENGINE_TAG));
        crate::catalog::ENGINE_CUDA.write_tag(dir.path()).unwrap();
        assert!(engine_release_check(&cfg).ok);
    }
}
