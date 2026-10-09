//! Persistent configuration and model profiles. Stored as JSON in
//! `%APPDATA%\Kokon\config.json`; every field has a default so older or
//! partial files keep loading.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, ErrorCode};

pub const APP_NAME: &str = "Kokon";
/// Folder names before the app was renamed Kokon (2026-10-08); moved on first start.
const LEGACY_CONFIG_DIR: &str = "Qwen Local Studio";
const LEGACY_DATA_DIR: &str = "QwenLocalStudio";
/// Distinct from the installer's program folder `%LOCALAPPDATA%\Kokon`, so uninstalling
/// never touches the gallery or the models.
pub const DATA_DIR_NAME: &str = "KokonData";
pub const DEFAULT_SERVER_PORT: u16 = 39150;
pub const DEFAULT_SERVICE_PORT: u16 = 39151;
/// ESRGAN models offered in Settings: (file stem, label, download URL). Chosen by visual A/B on
/// a Qwen edit (2026-10-06): both keep fine texture that RealESRGAN_x4plus smooths away.
pub const UPSCALER_CHOICES: &[(&str, &str, &str)] = &[
    ("4x-UltraSharp", "UltraSharp", "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth"),
    ("4x_NMKD-Superscale-SP_178000_G", "Superscale-SP", "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth"),
];
pub const DEFAULT_UPSCALER: &str = UPSCALER_CHOICES[0].0;
/// Default before schema 2; configs still on it are moved to [`DEFAULT_UPSCALER`].
const OLD_DEFAULT_UPSCALER: &str = "RealESRGAN_x4plus";
const SCHEMA_VERSION: u32 = 6;
/// Profile id of the official Qwen-Image 2.1 weights (the default since schema 5).
pub const OFFICIAL_PROFILE_ID: &str = "qwen-image-2.1";

/// Where to download `name`, if it is one of [`UPSCALER_CHOICES`].
pub fn upscaler_download_url(name: &str) -> Option<&'static str> {
    UPSCALER_CHOICES.iter().find(|(n, _, _)| *n == name).map(|(_, _, u)| *u)
}
/// Extensions sd-server accepts for model-backed upscalers (common.cpp, commit 3f8527a).
pub const UPSCALER_EXTENSIONS: &[&str] = &["pth", "safetensors", "gguf", "pt"];

/// Where the text encoder's weights live. Compute always runs on the selected GPU except
/// for `Cpu`, which also computes on the CPU (slow for image editing, measured ~35×).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TextEncoderPlacement {
    /// Weights in system RAM, compute on GPU (`--params-backend te=cpu`). Recommended.
    #[default]
    Ram,
    /// Weights and compute on GPU (fastest prompt encoding, slower sampling on 16 GB).
    Gpu,
    /// Weights and compute on CPU (`--backend te=cpu`). Lowest VRAM, slow editing.
    Cpu,
    /// Let stable-diffusion.cpp's auto-fit decide.
    Auto,
    /// Every model's weights reloaded from disk when needed (`--params-backend disk`).
    /// Least memory, slower; for Macs with 16 GB of unified memory.
    Disk,
}

/// Which stable-diffusion.cpp build runs the model. Each build lives in its own folder, so
/// switching back and forth needs no re-download.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    /// NVIDIA (CUDA 12 build). The verified path.
    Cuda,
    /// AMD, Intel and NVIDIA through Vulkan. Works on NVIDIA; AMD/Intel unverified.
    Vulkan,
    /// Apple Silicon (Metal build, macOS 26+). Beta.
    Metal,
}

impl Default for GpuBackend {
    /// The engine of this platform when nothing is configured.
    fn default() -> Self {
        if cfg!(target_os = "macos") { GpuBackend::Metal } else { GpuBackend::Cuda }
    }
}

impl GpuBackend {
    /// ggml device-name prefix (`cuda0`, `vulkan0`).
    pub fn device_prefix(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "cuda",
            GpuBackend::Vulkan => "vulkan",
            // ggml registers the Mac GPU as "MTL0" (`--list-devices`, seen on an M2 Pro).
            GpuBackend::Metal => "mtl",
        }
    }

    /// Folder of this build, relative to the stable-diffusion.cpp folder ("" = the folder
    /// itself, where the CUDA build has always been).
    pub fn engine_subdir(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "",
            GpuBackend::Vulkan => "engine-vulkan",
            GpuBackend::Metal => "engine-metal",
        }
    }

    /// Engine for a GPU vendor as reported by `diagnostics::query_gpus`.
    pub fn for_vendor(vendor: &str) -> Self {
        match vendor {
            "nvidia" => GpuBackend::Cuda,
            "apple" => GpuBackend::Metal,
            _ => GpuBackend::Vulkan,
        }
    }
}

/// Effort of a Detailed upscale (low-denoise re-render at ~2K before the upscaler model).
/// Generation itself always runs in one pass (two-pass refine removed 2026-10-06).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RefineMode {
    /// Kept for old configs; behaves like Fast.
    Off,
    /// Fewer refine steps + EasyCache.
    #[default]
    Fast,
    /// More refine steps, no cache.
    Best,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecommendedSettings {
    pub sampler: String,
    pub scheduler: Option<String>,
    /// 1.0 = no classifier-free guidance: one model pass per step, negative prompt unused.
    pub cfg: f32,
    /// CFG for modes that start from an image (edit, multi-reference, inpaint, outpaint,
    /// Detailed upscale).
    pub cfg_edit: f32,
    pub steps_fast: u32,
    pub steps_balanced: u32,
    pub steps_quality: u32,
    pub steps_ultra: u32,
    /// Pixel area used by Auto resolution at each quality level.
    pub area_fast: u64,
    pub area_balanced: u64,
    pub area_quality: u64,
    pub area_ultra: u64,
    pub align: u32,
    pub max_side: u32,
    pub max_references: u32,
    pub negative_prompt: String,
    /// Detailed upscale: largest re-render area and effective steps per effort.
    pub hires_max_area: u64,
    pub hires_steps_fast: u32,
    pub hires_steps_best: u32,
    /// EasyCache reuse threshold of a Fast Detailed upscale.
    pub easycache_threshold: f32,
}

impl Default for RecommendedSettings {
    fn default() -> Self {
        // Official Qwen-Image 2.1 settings (QwenLM/Qwen-Image-2.1 README, diffusers
        // QwenImage21Pipeline): no CFG ("meant to be sampled without guidance"), 40 steps,
        // native 2K buckets; flow-matching Euler, whose resolution-dependent schedule
        // sd-server selects automatically. Only Ultra runs the official 40 steps; the other
        // presets use 20 (judged enough by the user, 2026-10-09) and differ in size.
        Self {
            sampler: "euler".into(),
            scheduler: None,
            cfg: 1.0,
            cfg_edit: 1.0,
            steps_fast: 20,
            steps_balanced: 20,
            steps_quality: 20,
            steps_ultra: 40,
            area_fast: 1024 * 1024,
            area_balanced: 1536 * 1536,
            area_quality: 2048 * 2048,
            area_ultra: 2048 * 2048,
            align: 32,
            max_side: 4096,
            max_references: 10,
            negative_prompt: String::new(),
            hires_max_area: 2048 * 1152,
            hires_steps_fast: 6,
            hires_steps_best: 8,
            easycache_threshold: 0.2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileCapabilities {
    pub text_to_image: bool,
    pub edit: bool,
    pub multi_reference: bool,
    pub inpaint: bool,
    pub outpaint: bool,
    pub alpha: bool,
}

impl Default for ProfileCapabilities {
    fn default() -> Self {
        Self { text_to_image: true, edit: true, multi_reference: true, inpaint: true, outpaint: true, alpha: true }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelProfile {
    pub id: String,
    pub display_name: String,
    pub diffusion_model: PathBuf,
    pub vae: PathBuf,
    pub text_encoder: PathBuf,
    pub vision_projector: Option<PathBuf>,
    /// Extra sd-server arguments specific to this model.
    pub backend_args: Vec<String>,
    pub recommended_settings: RecommendedSettings,
    pub capabilities: ProfileCapabilities,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self::official(&default_sdcpp_dir(), crate::catalog::preset(crate::catalog::DEFAULT_PRESET).unwrap())
    }
}

impl ModelProfile {
    /// The official Qwen-Image 2.1 weights a hardware preset downloads (Unsloth GGUF of
    /// Qwen/Qwen-Image-2.1, Qwen's own Qwen3-VL GGUF, the 2.1 VAE).
    pub fn official(sdcpp_dir: &Path, preset: &crate::catalog::Preset) -> Self {
        use crate::catalog::{VAE, VISION_PROJECTOR};
        let quant = preset.diffusion.dest.rsplit('-').next().unwrap_or("").trim_end_matches(".gguf");
        Self {
            id: OFFICIAL_PROFILE_ID.into(),
            display_name: format!("Qwen-Image 2.1 {quant} (official)"),
            diffusion_model: preset.diffusion.path(sdcpp_dir),
            vae: VAE.path(sdcpp_dir),
            text_encoder: preset.text_encoder.path(sdcpp_dir),
            vision_projector: Some(VISION_PROJECTOR.path(sdcpp_dir)),
            backend_args: vec![],
            recommended_settings: RecommendedSettings::default(),
            capabilities: ProfileCapabilities::default(),
        }
    }

    /// (label, path, required) for every model file.
    pub fn files(&self) -> Vec<(&'static str, &Path, bool)> {
        let mut v = vec![
            ("Diffusion model", self.diffusion_model.as_path(), true),
            ("VAE", self.vae.as_path(), true),
            ("Text encoder (Qwen3-VL)", self.text_encoder.as_path(), true),
        ];
        if let Some(p) = &self.vision_projector {
            v.push(("Vision projector (mmproj)", p.as_path(), true));
        }
        v
    }

    pub fn model_name(&self) -> String {
        self.diffusion_model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| self.display_name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppConfig {
    pub schema_version: u32,
    pub sdcpp_dir: PathBuf,
    /// stable-diffusion.cpp build that runs the model.
    pub gpu_backend: GpuBackend,
    pub server_port: u16,
    pub service_port: u16,
    pub active_profile: String,
    pub profiles: Vec<ModelProfile>,
    /// ggml device name for diffusion + VAE, e.g. "cuda0" (see `sd-server --list-devices`).
    /// A name of another engine (e.g. "cuda0" with Vulkan) means device 0 of the active one.
    pub device: String,
    pub text_encoder_placement: TextEncoderPlacement,
    /// VRAM kept free, in GiB (passed as a negative `--max-vram`).
    pub vram_reserve_gib: f32,
    pub flash_attention: bool,
    /// Native CUDA SageAttention in the diffusion model (`--sage-attn`); ~1.5x faster
    /// steps at 2K on an RTX 5070 Ti with near-identical output. CUDA only (SM80+);
    /// ignored by other engines.
    pub sage_attention: bool,
    /// VAE tile size in pixels (`--vae-tiling --vae-tile-size N`); 0 disables tiling.
    /// Untiled 2K decodes need ~18 GB, OOM and retry with 256px tiles (~22 s); 768px
    /// tiles decode 2K in ~4 s on a 16 GB card.
    pub vae_tile_size: u32,
    /// Storage type of the Qwen 2.1 prefix cache (text + reference K/V reused across steps,
    /// `--model-args qwen_image_2_1_prefix_cache_type=...`). "auto" picks FP32 whenever
    /// SageAttention is on: ~8 GiB for a 2 MP reference, which never fits on 16 GB, so every
    /// edit fell back to the full sequence. "q8_0" (~2.2 GiB at 2 MP) fits; "f16" halves FP32;
    /// "off" disables the cache.
    pub prefix_cache_type: String,
    /// Load all weights when the server starts (`--eager-load`) instead of on the first
    /// step of the first job (~7 s of per-segment loading).
    pub eager_load: bool,
    /// Folder with ESRGAN models (`--hires-upscalers-dir`); None = `<sdcpp>\models\upscalers`.
    pub upscalers_dir: Option<PathBuf>,
    /// ESRGAN model name (file stem) used to enlarge results and for upscales.
    pub upscaler_model: String,
    /// ESRGAN tile size in pixels; 0 = backend default (128).
    pub upscale_tile_size: u32,
    /// Default effort of a Detailed upscale.
    pub refine_mode: RefineMode,
    pub extra_server_args: Vec<String>,
    pub output_dir: PathBuf,
    pub keep_server_running: bool,
    /// When no desktop window is attached (e.g. MCP only), unload the model after this
    /// many idle minutes. 0 disables.
    pub idle_unload_minutes: u32,
    pub setup_complete: bool,
    /// Hardware preset chosen in the setup wizard (`catalog::PRESETS`).
    pub hardware_preset: String,
    /// Plan as if the GPU had this much VRAM (MiB); 0 = use the detected size. For testing
    /// smaller cards on a larger one, together with a VRAM ballast (scripts/dev).
    pub vram_override_mib: u64,
    /// Opaque UI preferences (last quality, resolution mode, aspect...).
    pub ui: serde_json::Value,
}

impl Default for AppConfig {
    fn default() -> Self {
        let sdcpp = default_sdcpp_dir();
        Self {
            schema_version: SCHEMA_VERSION,
            profiles: vec![ModelProfile::official(&sdcpp, crate::catalog::preset(crate::catalog::DEFAULT_PRESET).unwrap())],
            active_profile: OFFICIAL_PROFILE_ID.into(),
            sdcpp_dir: sdcpp,
            gpu_backend: GpuBackend::default(),
            server_port: DEFAULT_SERVER_PORT,
            service_port: DEFAULT_SERVICE_PORT,
            device: "cuda0".into(),
            text_encoder_placement: TextEncoderPlacement::Ram,
            vram_reserve_gib: 1.0,
            flash_attention: true,
            sage_attention: true,
            vae_tile_size: 768,
            prefix_cache_type: "q8_0".into(),
            eager_load: true,
            upscalers_dir: None,
            upscaler_model: DEFAULT_UPSCALER.into(),
            upscale_tile_size: 0,
            refine_mode: RefineMode::Fast,
            extra_server_args: vec![],
            output_dir: default_output_dir(),
            keep_server_running: false,
            idle_unload_minutes: 15,
            setup_complete: false,
            hardware_preset: crate::catalog::DEFAULT_PRESET.into(),
            vram_override_mib: 0,
            ui: serde_json::json!({}),
        }
    }
}

pub fn default_output_dir() -> PathBuf {
    dirs::picture_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join("Pictures")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
}

/// Where the setup wizard installs stable-diffusion.cpp and the models on a new install:
/// `%LOCALAPPDATA%\KokonData\sdcpp`.
pub fn default_sdcpp_dir() -> PathBuf {
    data_dir().join("sdcpp")
}

/// `%APPDATA%\Kokon` (roaming) — configuration.
pub fn config_dir() -> PathBuf {
    if let Ok(p) = std::env::var("QLS_CONFIG_DIR") {
        return PathBuf::from(p);
    }
    migrate_dir(&dirs::config_dir().unwrap_or_else(std::env::temp_dir), LEGACY_CONFIG_DIR, APP_NAME)
}

/// `%LOCALAPPDATA%\KokonData` — gallery database, inputs, thumbnails, logs, models
/// installed by the setup screen.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("QLS_DATA_DIR") {
        return PathBuf::from(p);
    }
    migrate_dir(&dirs::data_local_dir().unwrap_or_else(std::env::temp_dir), LEGACY_DATA_DIR, DATA_DIR_NAME)
}

/// The data folder of the app's previous name (for relocating stored paths).
fn legacy_data_dir() -> PathBuf {
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join(LEGACY_DATA_DIR)
}

/// `base\new`, moving `base\old` there first if only the old folder exists (a rename on the
/// same drive). If the move fails, e.g. while a service of the old version still holds its
/// files, the old folder stays in use and the move is retried on the next start.
pub fn migrate_dir(base: &Path, old: &str, new: &str) -> PathBuf {
    let new_path = base.join(new);
    let old_path = base.join(old);
    // Unit tests must never move the real folders of this PC (tests call it with temp dirs).
    if cfg!(test) && !base.starts_with(std::env::temp_dir()) {
        return new_path;
    }
    if new_path.exists() || !old_path.is_dir() {
        return new_path;
    }
    match std::fs::rename(&old_path, &new_path) {
        Ok(()) => {
            tracing::info!("moved {} to {}", old_path.display(), new_path.display());
            new_path
        }
        Err(e) => {
            tracing::warn!("could not move {} to {} ({e}); using the old folder", old_path.display(), new_path.display());
            old_path
        }
    }
}

/// `p` inside the old data folder after that folder was moved: the same path in the new one.
fn relocate(p: &mut PathBuf, old: &Path, new: &Path) {
    if old.exists() || old == new {
        return;
    }
    if let Ok(rest) = p.strip_prefix(old) {
        *p = new.join(rest);
    }
}

impl AppConfig {
    pub fn path() -> PathBuf {
        config_dir().join("config.json")
    }

    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            // Windows PowerShell 5.1 writes UTF-8 with a BOM; accept it.
            Ok(s) => match serde_json::from_str::<AppConfig>(s.trim_start_matches('\u{feff}')) {
                Ok(mut c) => {
                    if c.profiles.is_empty() {
                        c.profiles.push(ModelProfile::official(&c.sdcpp_dir, c.preset()));
                        c.active_profile = OFFICIAL_PROFILE_ID.into();
                    }
                    // A file without profiles got the default one, built for the default
                    // folder and preset: build it for the configured folder and preset.
                    if c.profiles.len() == 1 && c.profiles[0] == ModelProfile::default() {
                        c.profiles[0] = ModelProfile::official(&c.sdcpp_dir, c.preset());
                    }
                    if c.schema_version < 2 && c.upscaler_model.trim() == OLD_DEFAULT_UPSCALER {
                        c.upscaler_model = DEFAULT_UPSCALER.into();
                    }
                    if c.schema_version < 3 {
                        // Presets saved with the old defaults move to the new ones; edited values stay.
                        for p in &mut c.profiles {
                            let r = &mut p.recommended_settings;
                            if r.steps_balanced == 30 {
                                r.steps_balanced = 20;
                            }
                            if r.steps_quality == 40 {
                                r.steps_quality = 20;
                            }
                            if r.area_quality == 1536 * 1536 {
                                r.area_quality = 2_000_000;
                            }
                        }
                    }
                    if c.schema_version < 4 {
                        // Old defaults came from sd.cpp's example (CFG 6, edits 4, 20 steps);
                        // move them to the official settings. Edited values stay.
                        for p in &mut c.profiles {
                            let r = &mut p.recommended_settings;
                            if r.cfg == 6.0 {
                                r.cfg = 1.0;
                            }
                            if r.cfg_edit == 4.0 {
                                r.cfg_edit = 1.0;
                            }
                            if r.steps_balanced == 20 {
                                r.steps_balanced = 40;
                            }
                            if r.steps_quality == 20 {
                                r.steps_quality = 40;
                            }
                            if r.area_quality == 2_000_000 {
                                r.area_quality = 2048 * 2048;
                            }
                        }
                    }
                    if c.schema_version < 6 {
                        // 20 steps for Balanced and Quality, Balanced at 1.5K (Ultra takes the
                        // official 40 steps). Edited values stay.
                        for p in &mut c.profiles {
                            let r = &mut p.recommended_settings;
                            if r.steps_balanced == 40 {
                                r.steps_balanced = 20;
                            }
                            if r.steps_quality == 40 {
                                r.steps_quality = 20;
                            }
                            if r.area_balanced == 1024 * 1024 {
                                r.area_balanced = 1536 * 1536;
                            }
                        }
                    }
                    // Paths into the data folder of the app's old name (models installed by the
                    // setup screen) follow the folder to its new name.
                    if std::env::var("QLS_DATA_DIR").is_err() {
                        c.relocate_paths(&legacy_data_dir(), &data_dir());
                    }
                    if c.schema_version < 5 && !c.profiles.iter().any(|p| p.id == OFFICIAL_PROFILE_ID) {
                        // The official weights become available as a profile; the active one
                        // stays until the user downloads them (setup wizard) or switches.
                        c.profiles.push(ModelProfile::official(&c.sdcpp_dir, c.preset()));
                    }
                    c.schema_version = c.schema_version.max(SCHEMA_VERSION);
                    c
                }
                Err(e) => {
                    tracing::warn!("config at {} is invalid ({e}); using defaults", path.display());
                    let _ = std::fs::copy(path, path.with_extension("json.invalid"));
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), AppError> {
        self.save_to(&Self::path())
    }

    /// Atomic write: temp file + rename.
    pub fn save_to(&self, path: &Path) -> Result<(), AppError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| AppError::io("create the settings folder", dir, e))?;
        }
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self).map_err(AppError::internal)?;
        std::fs::write(&tmp, json).map_err(|e| AppError::io("write settings", &tmp, e))?;
        std::fs::rename(&tmp, path).map_err(|e| AppError::io("save settings", path, e))?;
        Ok(())
    }

    pub fn profile(&self) -> &ModelProfile {
        self.profiles.iter().find(|p| p.id == self.active_profile).unwrap_or(&self.profiles[0])
    }

    pub fn relocate_paths(&mut self, old: &Path, new: &Path) {
        relocate(&mut self.sdcpp_dir, old, new);
        if let Some(d) = self.upscalers_dir.as_mut() {
            relocate(d, old, new);
        }
        for p in &mut self.profiles {
            for f in [&mut p.diffusion_model, &mut p.vae, &mut p.text_encoder] {
                relocate(f, old, new);
            }
            if let Some(v) = p.vision_projector.as_mut() {
                relocate(v, old, new);
            }
        }
    }

    pub fn preset(&self) -> &'static crate::catalog::Preset {
        crate::catalog::preset(&self.hardware_preset).unwrap_or_else(|| crate::catalog::preset(crate::catalog::DEFAULT_PRESET).unwrap())
    }

    /// Switches to the official weights of `preset` (replacing an older official profile)
    /// and its runtime settings. Used when the setup wizard finishes.
    pub fn apply_preset(&mut self, preset: &crate::catalog::Preset) {
        let mut profile = ModelProfile::official(&self.sdcpp_dir, preset);
        if let Some(old) = self.profiles.iter().find(|p| p.id == OFFICIAL_PROFILE_ID) {
            // Keep presets the user tuned on the official profile.
            profile.recommended_settings = old.recommended_settings.clone();
            profile.backend_args = old.backend_args.clone();
        }
        self.profiles.retain(|p| p.id != OFFICIAL_PROFILE_ID);
        self.profiles.insert(0, profile);
        self.active_profile = OFFICIAL_PROFILE_ID.into();
        self.hardware_preset = preset.id.into();
        self.text_encoder_placement = preset.text_encoder_placement;
    }

    /// VRAM the planner should assume: the override when set, otherwise the detected size.
    pub fn planning_vram_mib(&self, detected: Option<u64>) -> Option<u64> {
        if self.vram_override_mib > 0 { Some(self.vram_override_mib) } else { detected }
    }

    /// Folder of the active stable-diffusion.cpp build.
    pub fn engine_dir(&self) -> PathBuf {
        match self.gpu_backend.engine_subdir() {
            "" => self.sdcpp_dir.clone(),
            sub => self.sdcpp_dir.join(sub),
        }
    }

    pub fn server_exe(&self) -> PathBuf {
        self.engine_dir().join(if cfg!(windows) { "sd-server.exe" } else { "sd-server" })
    }

    /// The configured device when it belongs to the active engine, else its first GPU.
    pub fn effective_device(&self) -> String {
        let dev = self.device.trim().to_ascii_lowercase();
        let prefix = self.gpu_backend.device_prefix();
        match dev.strip_prefix(prefix) {
            Some(i) if !i.is_empty() && i.chars().all(|c| c.is_ascii_digit()) => dev,
            // Engines number their devices independently: fall back to the first one.
            _ => format!("{prefix}0"),
        }
    }

    /// SageAttention is requested and the engine supports it.
    pub fn uses_sage_attention(&self) -> bool {
        self.sage_attention && self.gpu_backend == GpuBackend::Cuda
    }

    pub fn upscalers_dir(&self) -> PathBuf {
        match &self.upscalers_dir {
            Some(d) if !d.as_os_str().is_empty() => d.clone(),
            _ => self.sdcpp_dir.join("models").join("upscalers"),
        }
    }

    /// The configured ESRGAN model file, if present. sd-server resolves names the same
    /// way: `<dir>/<name>.<ext>`, top level only.
    pub fn upscaler_file(&self) -> Option<PathBuf> {
        let name = self.upscaler_model.trim();
        if name.is_empty() {
            return None;
        }
        let dir = self.upscalers_dir();
        UPSCALER_EXTENSIONS.iter().map(|e| dir.join(format!("{name}.{e}"))).find(|p| p.is_file())
    }

    pub fn validate(&self) -> Result<(), AppError> {
        if self.server_port == self.service_port {
            return Err(AppError::new(ErrorCode::InvalidConfig, "The inference server and the app service cannot use the same port."));
        }
        if self.server_port < 1024 || self.service_port < 1024 {
            return Err(AppError::new(ErrorCode::InvalidConfig, "Ports must be 1024 or higher."));
        }
        if !(0.0..=64.0).contains(&self.vram_reserve_gib) {
            return Err(AppError::new(ErrorCode::InvalidConfig, "VRAM reserve must be between 0 and 64 GiB."));
        }
        if self.device.trim().is_empty() || !self.device.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(AppError::new(ErrorCode::InvalidConfig, "Device must be a backend name such as cuda0."));
        }
        if self.prefix_cache_type.is_empty() || !self.prefix_cache_type.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(AppError::new(ErrorCode::InvalidConfig, "Prefix cache type must be auto, off, or a weight type such as q8_0 or f16."));
        }
        if self.profiles.is_empty() {
            return Err(AppError::new(ErrorCode::InvalidConfig, "No model profile configured."));
        }
        if self.vram_override_mib != 0 && !(4096..=196_608).contains(&self.vram_override_mib) {
            return Err(AppError::new(ErrorCode::InvalidConfig, "Simulated VRAM must be between 4096 and 196608 MiB (or 0 to detect it)."));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let c = AppConfig::default();
        let p = c.profile();
        assert_eq!(p.id, OFFICIAL_PROFILE_ID);
        assert_eq!(p.display_name, "Qwen-Image 2.1 Q4_K_M (official)");
        assert!(p.diffusion_model.ends_with("qwen-image-2.1-Q4_K_M.gguf"));
        assert!(p.diffusion_model.starts_with(default_sdcpp_dir()));
        assert_eq!(c.hardware_preset, "16gb");
        assert_eq!(c.planning_vram_mib(Some(16303)), Some(16303));
        assert_eq!(c.device, "cuda0");
        assert_eq!(c.vram_reserve_gib, 1.0);
        assert!(c.flash_attention);
        assert!(!c.keep_server_running);
        assert!(c.validate().is_ok());
        assert_eq!(p.files().len(), 4);
        assert_eq!(c.refine_mode, RefineMode::Fast);
        assert_eq!(c.upscaler_model, "4x-UltraSharp");
        assert_eq!(c.upscalers_dir(), default_sdcpp_dir().join("models").join("upscalers"));
    }

    #[test]
    fn engine_choice_sets_folder_device_and_flags() {
        let mut c = AppConfig::default();
        c.gpu_backend = GpuBackend::Cuda;
        assert_eq!(c.engine_dir(), c.sdcpp_dir);
        assert_eq!(c.effective_device(), "cuda0");
        assert!(c.uses_sage_attention());
        c.gpu_backend = GpuBackend::Vulkan;
        assert_eq!(c.engine_dir(), c.sdcpp_dir.join("engine-vulkan"));
        assert!(c.server_exe().starts_with(c.sdcpp_dir.join("engine-vulkan")));
        assert_eq!(c.effective_device(), "vulkan0", "a CUDA device name maps to the first Vulkan GPU");
        c.device = "Vulkan1".into();
        assert_eq!(c.effective_device(), "vulkan1");
        c.device = "vulkan".into();
        assert_eq!(c.effective_device(), "vulkan0");
        assert!(!c.uses_sage_attention(), "SageAttention is CUDA only");
        assert_eq!(GpuBackend::for_vendor("nvidia"), GpuBackend::Cuda);
        assert_eq!(GpuBackend::for_vendor("amd"), GpuBackend::Vulkan);
        assert_eq!(GpuBackend::for_vendor("apple"), GpuBackend::Metal);
        c.gpu_backend = GpuBackend::Metal;
        c.device = "cuda0".into();
        assert_eq!(c.effective_device(), "mtl0", "matches MTL0 in --list-devices");
        assert_eq!(c.engine_dir(), c.sdcpp_dir.join("engine-metal"));
        assert!(!c.uses_sage_attention());
        // Configs written before the field existed keep this platform's engine.
        let old: AppConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(old.gpu_backend, if cfg!(target_os = "macos") { GpuBackend::Metal } else { GpuBackend::Cuda });
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains(r#""gpuBackend":"metal""#));
    }

    #[test]
    fn official_profile_is_added_but_not_forced() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let old = r#"{"schemaVersion":4,"sdcppDir":"D:\\sd","activeProfile":"uc","profiles":[{"id":"uc","displayName":"UC","diffusionModel":"D:\\sd\\uc.gguf"}]}"#;
        std::fs::write(&p, old).unwrap();
        let c = AppConfig::load_from(&p);
        assert_eq!(c.active_profile, "uc", "the user's choice stays active");
        let off = c.profiles.iter().find(|p| p.id == OFFICIAL_PROFILE_ID).unwrap();
        assert_eq!(off.diffusion_model, PathBuf::from(r"D:\sd\models\qwen21\qwen-image-2.1-Q4_K_M.gguf"));
        // Saved again, nothing is added twice.
        c.save_to(&p).unwrap();
        assert_eq!(AppConfig::load_from(&p).profiles.len(), 2);
        // A file without profiles uses its own folder and preset.
        std::fs::write(&p, r#"{"sdcppDir":"D:\\sd","hardwarePreset":"24gb"}"#).unwrap();
        let c = AppConfig::load_from(&p);
        assert_eq!(c.profile().diffusion_model, PathBuf::from(r"D:\sd\models\qwen21\qwen-image-2.1-Q8_0.gguf"));
    }

    #[test]
    fn old_folders_move_to_the_new_name() {
        let base = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base.path().join("QwenLocalStudio").join("inputs")).unwrap();
        std::fs::write(base.path().join("QwenLocalStudio").join("gallery.db"), b"db").unwrap();
        let d = migrate_dir(base.path(), "QwenLocalStudio", "KokonData");
        assert_eq!(d, base.path().join("KokonData"));
        assert!(d.join("gallery.db").is_file() && !base.path().join("QwenLocalStudio").exists());
        // Once moved, nothing happens again; a fresh install just gets the new name.
        assert_eq!(migrate_dir(base.path(), "QwenLocalStudio", "KokonData"), d);
        assert_eq!(migrate_dir(base.path(), "Missing", "New"), base.path().join("New"));
        // Model paths inside the moved folder follow it; others stay.
        let (old, new) = (base.path().join("QwenLocalStudio"), d.clone());
        let mut c = AppConfig::default();
        c.sdcpp_dir = old.join("sdcpp");
        c.apply_preset(crate::catalog::preset("16gb").unwrap());
        c.profiles.push(ModelProfile { id: "mine".into(), diffusion_model: PathBuf::from(r"C:\AI\sdcpp\x.gguf"), ..ModelProfile::default() });
        c.relocate_paths(&old, &new);
        assert_eq!(c.sdcpp_dir, new.join("sdcpp"));
        assert!(c.profile().diffusion_model.starts_with(new.join("sdcpp")));
        assert_eq!(c.profiles[1].diffusion_model, PathBuf::from(r"C:\AI\sdcpp\x.gguf"));
    }

    #[test]
    fn presets_switch_weights_and_keep_tuning() {
        let mut c = AppConfig::default();
        c.profiles[0].recommended_settings.steps_fast = 12;
        c.apply_preset(crate::catalog::preset("32gb").unwrap());
        let p = c.profile();
        assert!(p.diffusion_model.ends_with("qwen-image-2.1-Q8_0.gguf"));
        assert!(p.text_encoder.ends_with("Qwen3VL-8B-Instruct-Q8_0.gguf"));
        assert_eq!(p.display_name, "Qwen-Image 2.1 Q8_0 (official)");
        assert_eq!(p.recommended_settings.steps_fast, 12);
        assert_eq!((c.hardware_preset.as_str(), c.profiles.len()), ("32gb", 1));
        c.vram_override_mib = 12_288;
        assert_eq!(c.planning_vram_mib(Some(16303)), Some(12_288));
        assert!(c.validate().is_ok());
        c.vram_override_mib = 100;
        assert!(c.validate().is_err());
    }

    #[test]
    fn old_default_upscaler_is_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        std::fs::write(&p, r#"{"schemaVersion":1,"upscalerModel":"RealESRGAN_x4plus"}"#).unwrap();
        let c = AppConfig::load_from(&p);
        assert_eq!(c.upscaler_model, DEFAULT_UPSCALER);
        assert_eq!(c.schema_version, SCHEMA_VERSION);
        // A choice made after the change is kept.
        std::fs::write(&p, r#"{"schemaVersion":2,"upscalerModel":"RealESRGAN_x4plus"}"#).unwrap();
        assert_eq!(AppConfig::load_from(&p).upscaler_model, "RealESRGAN_x4plus");
        std::fs::write(&p, r#"{"schemaVersion":1,"upscalerModel":"4x_NMKD-Superscale-SP_178000_G"}"#).unwrap();
        assert_eq!(AppConfig::load_from(&p).upscaler_model, "4x_NMKD-Superscale-SP_178000_G");
    }

    #[test]
    fn old_quality_presets_are_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let old = r#"{"schemaVersion":2,"profiles":[{"id":"a","recommendedSettings":{"steps_balanced":30,"steps_quality":40,"area_quality":2359296}}]}"#;
        std::fs::write(&p, old).unwrap();
        let r = AppConfig::load_from(&p).profiles[0].recommended_settings.clone();
        // Schema 3, 4 and 6 run in sequence and end on the current presets.
        assert_eq!((r.steps_balanced, r.steps_quality, r.area_quality, r.cfg_edit), (20, 20, 2048 * 2048, 1.0));
        // Values the user changed are kept.
        let custom = r#"{"schemaVersion":2,"profiles":[{"id":"a","recommendedSettings":{"steps_balanced":25,"steps_quality":50}}]}"#;
        std::fs::write(&p, custom).unwrap();
        let r = AppConfig::load_from(&p).profiles[0].recommended_settings.clone();
        assert_eq!((r.steps_balanced, r.steps_quality), (25, 50));
    }

    #[test]
    fn sdcpp_example_settings_move_to_official() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let old = r#"{"schemaVersion":3,"profiles":[{"id":"a","recommendedSettings":{"cfg":6.0,"cfg_edit":4.0,"steps_fast":20,"steps_balanced":20,"steps_quality":20,"area_quality":2000000}}]}"#;
        std::fs::write(&p, old).unwrap();
        let r = AppConfig::load_from(&p).profiles[0].recommended_settings.clone();
        assert_eq!((r.cfg, r.cfg_edit), (1.0, 1.0));
        assert_eq!((r.steps_fast, r.steps_balanced, r.steps_quality, r.area_quality), (20, 20, 20, 2048 * 2048));
        assert_eq!(r, RecommendedSettings::default());
        let custom = r#"{"schemaVersion":3,"profiles":[{"id":"a","recommendedSettings":{"cfg":3.0,"cfg_edit":2.5,"steps_balanced":28}}]}"#;
        std::fs::write(&p, custom).unwrap();
        let r = AppConfig::load_from(&p).profiles[0].recommended_settings.clone();
        assert_eq!((r.cfg, r.cfg_edit, r.steps_balanced), (3.0, 2.5, 28));
    }

    #[test]
    fn upscaler_file_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = AppConfig::default();
        c.upscalers_dir = Some(dir.path().to_path_buf());
        assert!(c.upscaler_file().is_none());
        std::fs::write(dir.path().join("4x-UltraSharp.pth"), b"x").unwrap();
        assert_eq!(c.upscaler_file().unwrap(), dir.path().join("4x-UltraSharp.pth"));
        c.upscaler_model = "  ".into();
        assert!(c.upscaler_file().is_none());
        // Older config files without the new fields keep loading with defaults.
        let old: AppConfig = serde_json::from_str(r#"{"serverPort": 41000, "vaeTileSize": 512}"#).unwrap();
        assert_eq!(old.refine_mode, RefineMode::Fast);
        assert_eq!(old.profile().recommended_settings.hires_max_area, 2048 * 1152);
    }

    #[test]
    fn roundtrip_and_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let mut c = AppConfig::default();
        c.server_port = 40000;
        c.ui = serde_json::json!({"quality": "fast"});
        c.save_to(&path).unwrap();
        assert_eq!(AppConfig::load_from(&path), c);
        std::fs::write(&path, r#"{"serverPort": 41000}"#).unwrap();
        let p = AppConfig::load_from(&path);
        assert_eq!(p.server_port, 41000);
        assert_eq!(p.profiles.len(), 1);
        std::fs::write(&path, "\u{feff}{\"serverPort\": 42000}").unwrap();
        assert_eq!(AppConfig::load_from(&path).server_port, 42000, "UTF-8 BOM accepted");
        std::fs::write(&path, "{ broken").unwrap();
        assert_eq!(AppConfig::load_from(&path), AppConfig::default());
        assert!(path.with_extension("json.invalid").exists());
    }

    #[test]
    fn validation() {
        let mut c = AppConfig::default();
        c.service_port = c.server_port;
        assert!(c.validate().is_err());
        let mut c = AppConfig::default();
        c.device = "cuda0; rm".into();
        assert!(c.validate().is_err());
    }
}
