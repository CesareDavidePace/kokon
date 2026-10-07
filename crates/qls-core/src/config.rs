//! Persistent configuration and model profiles. Stored as JSON in
//! `%APPDATA%\Qwen Local Studio\config.json`; every field has a default so older or
//! partial files keep loading.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, ErrorCode};

pub const APP_NAME: &str = "Qwen Local Studio";
pub const DEFAULT_SDCPP_DIR: &str = r"C:\AI\sdcpp";
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
const SCHEMA_VERSION: u32 = 4;

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
    /// Pixel area used by Auto resolution at each quality level.
    pub area_fast: u64,
    pub area_balanced: u64,
    pub area_quality: u64,
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
        // sd-server selects automatically. Fast halves the steps for speed (not official).
        Self {
            sampler: "euler".into(),
            scheduler: None,
            cfg: 1.0,
            cfg_edit: 1.0,
            steps_fast: 20,
            steps_balanced: 40,
            steps_quality: 40,
            area_fast: 1024 * 1024,
            area_balanced: 1024 * 1024,
            area_quality: 2048 * 2048,
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
        Self::qwen_image_21_q4(Path::new(DEFAULT_SDCPP_DIR))
    }
}

impl ModelProfile {
    pub fn qwen_image_21_q4(sdcpp_dir: &Path) -> Self {
        let m = sdcpp_dir.join("models").join("qwen21");
        Self {
            id: "qwen-image-2.1-uc-q4km".into(),
            display_name: "Qwen Image 2.1 UC Q4_K_M".into(),
            diffusion_model: m.join("qwen-image-2.1-UC-Q4_K_M.gguf"),
            vae: m.join("qwen_image_2.1_vae_bf16.safetensors"),
            text_encoder: m.join("Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
            vision_projector: Some(m.join("mmproj-Qwen3VL-8B-Instruct-F16.gguf")),
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
    pub server_port: u16,
    pub service_port: u16,
    pub active_profile: String,
    pub profiles: Vec<ModelProfile>,
    /// ggml device name for diffusion + VAE, e.g. "cuda0" (see `sd-server --list-devices`).
    pub device: String,
    pub text_encoder_placement: TextEncoderPlacement,
    /// VRAM kept free, in GiB (passed as a negative `--max-vram`).
    pub vram_reserve_gib: f32,
    pub flash_attention: bool,
    /// Native CUDA SageAttention in the diffusion model (`--sage-attn`); ~1.5x faster
    /// steps at 2K on an RTX 5070 Ti with near-identical output.
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
    /// Opaque UI preferences (last quality, resolution mode, aspect...).
    pub ui: serde_json::Value,
}

impl Default for AppConfig {
    fn default() -> Self {
        let sdcpp = PathBuf::from(DEFAULT_SDCPP_DIR);
        Self {
            schema_version: SCHEMA_VERSION,
            profiles: vec![ModelProfile::qwen_image_21_q4(&sdcpp)],
            active_profile: "qwen-image-2.1-uc-q4km".into(),
            sdcpp_dir: sdcpp,
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

/// `%APPDATA%\Qwen Local Studio` (roaming) — configuration.
pub fn config_dir() -> PathBuf {
    if let Ok(p) = std::env::var("QLS_CONFIG_DIR") {
        return PathBuf::from(p);
    }
    dirs::config_dir().unwrap_or_else(std::env::temp_dir).join(APP_NAME)
}

/// `%LOCALAPPDATA%\QwenLocalStudio` — gallery database, inputs, thumbnails, logs.
/// Deliberately distinct from the installer's `%LOCALAPPDATA%\Qwen Local Studio`.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("QLS_DATA_DIR") {
        return PathBuf::from(p);
    }
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("QwenLocalStudio")
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
                        c.profiles.push(ModelProfile::qwen_image_21_q4(&c.sdcpp_dir));
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

    pub fn server_exe(&self) -> PathBuf {
        self.sdcpp_dir.join("sd-server.exe")
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
        assert_eq!(p.display_name, "Qwen Image 2.1 UC Q4_K_M");
        assert!(p.diffusion_model.ends_with("qwen-image-2.1-UC-Q4_K_M.gguf"));
        assert_eq!(c.device, "cuda0");
        assert_eq!(c.vram_reserve_gib, 1.0);
        assert!(c.flash_attention);
        assert!(!c.keep_server_running);
        assert!(c.validate().is_ok());
        assert_eq!(p.files().len(), 4);
        assert_eq!(c.refine_mode, RefineMode::Fast);
        assert_eq!(c.upscaler_model, "4x-UltraSharp");
        assert_eq!(c.upscalers_dir(), PathBuf::from(r"C:\AI\sdcpp\models\upscalers"));
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
        // Schema 3 then schema 4 run in sequence and end on the official settings.
        assert_eq!((r.steps_balanced, r.steps_quality, r.area_quality, r.cfg_edit), (40, 40, 2048 * 2048, 1.0));
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
        assert_eq!((r.steps_fast, r.steps_balanced, r.steps_quality, r.area_quality), (20, 40, 40, 2048 * 2048));
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
