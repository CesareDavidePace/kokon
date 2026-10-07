//! Inference adapter for stable-diffusion.cpp's native async API (`/sdcpp/v1/*`).
//!
//! This is the only module that knows the sd-server wire format. Schemas were verified
//! against examples/server (api.md, routes_sdcpp.cpp, async_jobs.h) at commit 3f8527a:
//!   GET  /sdcpp/v1/capabilities
//!   POST /sdcpp/v1/img_gen            -> 202 {id, kind, status, created, poll_url}
//!   GET  /sdcpp/v1/jobs/{id}          -> {status: queued|generating|completed|failed|cancelled, ...}
//!   POST /sdcpp/v1/jobs/{id}/cancel   -> 200, or 409 while generating (not interruptible)
//!   POST /sdcpp/v1/upscale            -> 200 {images, upscaler, scale, width, height} (sync ESRGAN)
//!
//! Request objects are merged field by field over the server's launch defaults
//! (`SDGenerationParams::from_json_str`), so omitted fields keep the CLI values.

use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, ErrorCode};

// ---------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Guidance {
    pub txt_cfg: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub img_cfg: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct SampleParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<String>,
    pub sample_steps: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_shift: Option<f32>,
    pub guidance: Guidance,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct VaeTilingParams {
    pub enabled: bool,
    /// Image pixels; 0 = backend default (256).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile_size_w: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile_size_h: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_overlap: Option<f32>,
}

/// Second pass of `img_gen` (`hires`): the first result is upscaled to the target size
/// (`upscaler` = built-in filter or an ESRGAN model stem from `--hires-upscalers-dir`)
/// and re-sampled with `denoising_strength`. Reference latents and the inpaint mask are
/// carried into the second pass (src/pipeline/image.cpp, commit 3f8527a).
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct HiresParams {
    pub enabled: bool,
    pub upscaler: String,
    pub target_width: u32,
    pub target_height: u32,
    /// Effective refine steps (the server scales the schedule by the denoise strength).
    pub steps: u32,
    pub denoising_strength: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscale_tile_size: Option<u32>,
}

/// Body of `POST /sdcpp/v1/img_gen`. Images are base64 PNG strings.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct ImgGenRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: i64,
    pub batch_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_image: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ref_images: Vec<String>,
    /// Our images already arrive at exact working geometry, so input transforms are
    /// disabled (`mode=none`) to guarantee no crop/stretch happens server-side.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub image_preprocess: Vec<String>,
    pub sample_params: SampleParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vae_tiling_params: Option<VaeTilingParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hires: Option<HiresParams>,
    /// e.g. "easycache"; applies to both passes. Omitted = server default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_mode: Option<String>,
    /// e.g. "threshold=0.2".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_option: Option<String>,
    pub output_format: String,
    /// We write our own metadata after post-processing.
    pub embed_image_metadata: bool,
}

pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn b64_decode(s: &str) -> Result<Vec<u8>, AppError> {
    let s = s.split_once(";base64,").map(|(_, d)| d).unwrap_or(s);
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| AppError::new(ErrorCode::GenerationFailed, "The inference server returned an unreadable image.").detail(e.to_string()))
}

// ---------------------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CapModel {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub stem: String,
    #[serde(default)]
    pub path: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CapLimits {
    #[serde(default)]
    pub min_width: u32,
    #[serde(default)]
    pub max_width: u32,
    #[serde(default)]
    pub min_height: u32,
    #[serde(default)]
    pub max_height: u32,
    #[serde(default)]
    pub max_batch_count: u32,
    #[serde(default)]
    pub max_queue_size: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct CapUpscaler {
    #[serde(default)]
    pub name: String,
    /// True for a model-backed upscaler (false for built-in filters).
    #[serde(default)]
    pub model: bool,
    /// Usable with `POST /sdcpp/v1/upscale` (RGB ESRGAN models).
    #[serde(default)]
    pub image_upscale: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Capabilities {
    #[serde(default)]
    pub model: CapModel,
    #[serde(default)]
    pub supported_modes: Vec<String>,
    #[serde(default)]
    pub samplers: Vec<String>,
    #[serde(default)]
    pub schedulers: Vec<String>,
    #[serde(default)]
    pub output_formats: Vec<String>,
    #[serde(default)]
    pub limits: CapLimits,
    #[serde(default)]
    pub features_by_mode: serde_json::Value,
    #[serde(default)]
    pub defaults_by_mode: serde_json::Value,
    #[serde(default)]
    pub upscalers: Vec<CapUpscaler>,
    /// Whether any RGB ESRGAN model is available for `/sdcpp/v1/upscale`.
    #[serde(default)]
    pub upscale: bool,
}

impl Capabilities {
    pub fn img_feature(&self, name: &str) -> bool {
        self.features_by_mode.get("img_gen").and_then(|f| f.get(name)).and_then(|v| v.as_bool()).unwrap_or(false)
    }
    pub fn supports_img_gen(&self) -> bool {
        self.supported_modes.iter().any(|m| m == "img_gen")
    }
    /// True when `name` is a model-backed upscaler the server can load (hires and,
    /// if `image` is set, the standalone upscale endpoint).
    pub fn has_upscaler(&self, name: &str, image: bool) -> bool {
        self.upscalers.iter().any(|u| u.model && u.name == name && (!image || u.image_upscale))
    }
    /// True when the loaded diffusion model is the file we expect (path compare is
    /// case/slash-insensitive, since Windows paths come back with forward slashes).
    pub fn model_matches(&self, expected: &std::path::Path) -> bool {
        let norm = |s: &str| s.replace('\\', "/").to_ascii_lowercase();
        let want = norm(&expected.to_string_lossy());
        let got = norm(&self.model.path);
        if !got.is_empty() {
            return got == want || want.ends_with(&got) || got.ends_with(&want);
        }
        let file = expected.file_name().map(|f| f.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        self.model.name.to_ascii_lowercase() == file
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobAccepted {
    pub id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendJobStatus {
    Queued,
    Generating,
    Completed,
    Failed,
    Cancelled,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct JobError {
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobImage {
    #[serde(default)]
    pub index: u32,
    pub b64_json: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct JobResult {
    #[serde(default)]
    pub images: Vec<JobImage>,
    #[serde(default)]
    pub output_format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendJob {
    pub id: String,
    pub status: BackendJobStatus,
    #[serde(default)]
    pub queue_position: u32,
    #[serde(default)]
    pub result: Option<JobResult>,
    #[serde(default)]
    pub error: Option<JobError>,
}

/// Body of `POST /sdcpp/v1/upscale`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UpscaleRequest {
    pub image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscaler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile_size: Option<u32>,
    pub output_format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpscaleResponse {
    #[serde(default)]
    pub images: Vec<JobImage>,
    #[serde(default)]
    pub upscaler: String,
    #[serde(default)]
    pub scale: u32,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelOutcome {
    Cancelled,
    /// The backend cannot interrupt a job that is already generating.
    NotInterruptible,
    AlreadyFinished,
    NotFound,
}

// ---------------------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct SdcppClient {
    base: String,
    http: reqwest::Client,
}

fn unavailable(e: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::ServerUnavailable, "The inference server is not reachable.").detail(e.to_string())
}

impl SdcppClient {
    pub fn new(port: u16) -> Self {
        let http = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(3))
            .build()
            .expect("http client");
        Self { base: format!("http://127.0.0.1:{port}"), http }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    pub async fn capabilities(&self, timeout: Duration) -> Result<Capabilities, AppError> {
        let r = self.http.get(format!("{}/sdcpp/v1/capabilities", self.base)).timeout(timeout).send().await.map_err(unavailable)?;
        if !r.status().is_success() {
            return Err(AppError::new(ErrorCode::ServerUnavailable, "The inference server did not report its capabilities.")
                .detail(format!("HTTP {}", r.status())));
        }
        r.json::<Capabilities>().await.map_err(|e| {
            AppError::new(ErrorCode::Unsupported, "The program on the inference port does not look like stable-diffusion.cpp.").detail(e.to_string())
        })
    }

    pub async fn submit(&self, req: &ImgGenRequest) -> Result<JobAccepted, AppError> {
        let body = serde_json::to_vec(req).map_err(AppError::internal)?;
        let r = self
            .http
            .post(format!("{}/sdcpp/v1/img_gen", self.base))
            .header("Content-Type", "application/json")
            .body(body)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(unavailable)?;
        let status = r.status().as_u16();
        let text = r.text().await.unwrap_or_default();
        match status {
            200 | 202 => serde_json::from_str(&text)
                .map_err(|e| AppError::new(ErrorCode::Unsupported, "Unexpected reply from the inference server.").detail(format!("{e}: {text}"))),
            429 => Err(AppError::new(ErrorCode::QueueFull, "The inference server queue is full.").detail(text)),
            400 => Err(AppError::new(ErrorCode::MalformedRequest, "The inference server rejected the request.").detail(text)),
            _ => Err(AppError::new(ErrorCode::GenerationFailed, "The inference server returned an error.").detail(format!("HTTP {status}: {text}"))),
        }
    }

    /// Synchronous ESRGAN pass. It waits for the server's generation lock, so the
    /// timeout covers a job that might still be finishing.
    pub async fn upscale(&self, req: &UpscaleRequest) -> Result<UpscaleResponse, AppError> {
        let body = serde_json::to_vec(req).map_err(AppError::internal)?;
        let r = self
            .http
            .post(format!("{}/sdcpp/v1/upscale", self.base))
            .header("Content-Type", "application/json")
            .body(body)
            .timeout(Duration::from_secs(600))
            .send()
            .await
            .map_err(unavailable)?;
        let status = r.status().as_u16();
        let text = r.text().await.unwrap_or_default();
        let server_error = || serde_json::from_str::<serde_json::Value>(&text).ok().and_then(|v| v["error"].as_str().map(str::to_owned)).unwrap_or_else(|| text.clone());
        match status {
            200 => {
                let resp: UpscaleResponse = serde_json::from_str(&text)
                    .map_err(|e| AppError::new(ErrorCode::Unsupported, "Unexpected reply from the upscaler.").detail(format!("{e}: {}", &text[..text.len().min(300)])))?;
                if resp.images.is_empty() {
                    return Err(AppError::new(ErrorCode::GenerationFailed, "The upscaler returned no image."));
                }
                Ok(resp)
            }
            404 => Err(AppError::new(ErrorCode::Unsupported, "This sd-server build has no upscale endpoint.").detail(text)),
            400 => Err(AppError::new(ErrorCode::MalformedRequest, "The upscaler rejected the image.").detail(server_error())),
            _ => Err(AppError::new(ErrorCode::GenerationFailed, "The upscaler failed.").detail(format!("HTTP {status}: {}", server_error()))),
        }
    }

    pub async fn job(&self, id: &str) -> Result<BackendJob, AppError> {
        let r = self
            .http
            .get(format!("{}/sdcpp/v1/jobs/{}", self.base, id))
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(unavailable)?;
        match r.status().as_u16() {
            200 => r.json().await.map_err(|e| AppError::new(ErrorCode::Unsupported, "Unexpected job status format.").detail(e.to_string())),
            404 | 410 => Err(AppError::new(ErrorCode::NotFound, "The inference job is no longer known to the server.")),
            s => Err(AppError::new(ErrorCode::GenerationFailed, "Could not read job status.").detail(format!("HTTP {s}"))),
        }
    }

    pub async fn cancel(&self, id: &str) -> Result<CancelOutcome, AppError> {
        let r = self
            .http
            .post(format!("{}/sdcpp/v1/jobs/{}/cancel", self.base, id))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(unavailable)?;
        let status = r.status().as_u16();
        Ok(match status {
            200 => {
                let job: Option<BackendJob> = r.json().await.ok();
                match job.map(|j| j.status) {
                    Some(BackendJobStatus::Cancelled) | None => CancelOutcome::Cancelled,
                    _ => CancelOutcome::AlreadyFinished,
                }
            }
            409 => CancelOutcome::NotInterruptible,
            404 | 410 => CancelOutcome::NotFound,
            _ => CancelOutcome::NotInterruptible,
        })
    }
}

/// Parses `stable-diffusion.cpp version X, commit abc123` from `--help` output.
pub fn parse_version_banner(text: &str) -> Option<String> {
    let line = text.lines().find(|l| l.contains("stable-diffusion.cpp version"))?;
    Some(line.trim().trim_start_matches("stable-diffusion.cpp").trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization_matches_native_schema() {
        let req = ImgGenRequest {
            prompt: "change the red car to a black Porsche".into(),
            negative_prompt: String::new(),
            width: 1920,
            height: 1088,
            seed: 42,
            batch_count: 1,
            strength: None,
            init_image: None,
            mask_image: None,
            ref_images: vec!["AAA".into(), "BBB".into()],
            image_preprocess: vec!["target=ref,mode=none".into()],
            sample_params: SampleParams {
                sample_method: Some("euler".into()),
                scheduler: None,
                sample_steps: 30,
                flow_shift: None,
                guidance: Guidance { txt_cfg: 6.0, img_cfg: None },
            },
            vae_tiling_params: None,
            hires: None,
            cache_mode: None,
            cache_option: None,
            output_format: "png".into(),
            embed_image_metadata: false,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["width"], 1920);
        assert_eq!(v["height"], 1088);
        assert_eq!(v["sample_params"]["sample_method"], "euler");
        assert_eq!(v["sample_params"]["sample_steps"], 30);
        assert_eq!(v["sample_params"]["guidance"]["txt_cfg"], 6.0);
        assert!(v["sample_params"].get("scheduler").is_none(), "omitted -> backend default");
        assert!(v["sample_params"]["guidance"].get("img_cfg").is_none());
        assert_eq!(v["ref_images"][0], "AAA");
        assert_eq!(v["ref_images"][1], "BBB", "reference order preserved");
        assert!(v.get("init_image").is_none());
        assert!(v.get("mask_image").is_none());
        assert!(v.get("strength").is_none());
        assert!(v.get("vae_tiling_params").is_none());
        assert_eq!(v["embed_image_metadata"], false);
        assert_eq!(v["image_preprocess"][0], "target=ref,mode=none");
        assert!(v.get("hires").is_none() && v.get("cache_mode").is_none() && v.get("cache_option").is_none(), "server defaults apply");
    }

    #[test]
    fn hires_cache_and_tiling_serialization() {
        let req = ImgGenRequest {
            prompt: "a cabin".into(),
            width: 1376,
            height: 768,
            hires: Some(HiresParams {
                enabled: true,
                upscaler: "RealESRGAN_x4plus".into(),
                target_width: 2048,
                target_height: 1152,
                steps: 6,
                denoising_strength: 0.35,
                upscale_tile_size: None,
            }),
            cache_mode: Some("easycache".into()),
            cache_option: Some("threshold=0.2".into()),
            vae_tiling_params: Some(VaeTilingParams { enabled: false, ..Default::default() }),
            ..Default::default()
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["hires"]["enabled"], true);
        assert_eq!(v["hires"]["upscaler"], "RealESRGAN_x4plus");
        assert_eq!((v["hires"]["target_width"].as_u64(), v["hires"]["target_height"].as_u64()), (Some(2048), Some(1152)));
        assert_eq!(v["hires"]["steps"], 6);
        assert!((v["hires"]["denoising_strength"].as_f64().unwrap() - 0.35).abs() < 1e-6);
        assert!(v["hires"].get("upscale_tile_size").is_none());
        assert!(v["hires"].get("scale").is_none(), "target size wins over scale");
        assert_eq!(v["cache_mode"], "easycache");
        assert_eq!(v["cache_option"], "threshold=0.2");
        // Only `enabled` is sent, so the server keeps its configured tile size.
        assert_eq!(v["vae_tiling_params"], serde_json::json!({"enabled": false}));
    }

    #[test]
    fn upscale_request_and_response() {
        let r = UpscaleRequest { image: "AAA".into(), upscaler: Some("RealESRGAN_x4plus".into()), tile_size: None, output_format: "png".into() };
        assert_eq!(serde_json::to_value(&r).unwrap(), serde_json::json!({"image": "AAA", "upscaler": "RealESRGAN_x4plus", "output_format": "png"}));
        let resp: UpscaleResponse = serde_json::from_str(
            r#"{"images":[{"index":0,"b64_json":"iVBOR"}],"upscaler":"RealESRGAN_x4plus","scale":4,"repeats":1,"width":5504,"height":3072,"output_format":"png"}"#,
        )
        .unwrap();
        assert_eq!((resp.scale, resp.width, resp.height), (4, 5504, 3072));
        assert_eq!(resp.images[0].b64_json, "iVBOR");
    }

    #[test]
    fn job_status_parsing() {
        let done = r#"{"completed":1791295347,"created":1791295301,"error":null,"id":"job_1","kind":"img_gen","queue_position":0,
            "result":{"images":[{"b64_json":"iVBORw0K","index":0}],"output_format":"png"},"started":1791295302,"status":"completed"}"#;
        let j: BackendJob = serde_json::from_str(done).unwrap();
        assert_eq!(j.status, BackendJobStatus::Completed);
        assert_eq!(j.result.unwrap().images[0].b64_json, "iVBORw0K");
        let failed = r#"{"id":"job_2","status":"failed","error":{"code":"generation_failed","message":"generate_image returned empty results"}}"#;
        let j: BackendJob = serde_json::from_str(failed).unwrap();
        assert_eq!(j.status, BackendJobStatus::Failed);
        assert_eq!(j.error.unwrap().code, "generation_failed");
        let weird: BackendJob = serde_json::from_str(r#"{"id":"x","status":"paused"}"#).unwrap();
        assert_eq!(weird.status, BackendJobStatus::Unknown);
    }

    #[test]
    fn capabilities_parsing_and_model_match() {
        let caps: Capabilities = serde_json::from_str(r#"{"model":{"name":"qwen-image-2.1-UC-Q4_K_M.gguf","path":"C:/AI/sdcpp/models/qwen21/qwen-image-2.1-UC-Q4_K_M.gguf","stem":"x"},
            "supported_modes":["img_gen"],"features_by_mode":{"img_gen":{"cancel_generating":false,"ref_images":true}},
            "limits":{"max_width":4096,"max_height":4096,"min_width":64,"min_height":64,"max_batch_count":8,"max_queue_size":64}}"#).unwrap();
        assert!(caps.supports_img_gen());
        assert!(caps.img_feature("ref_images"));
        assert!(!caps.img_feature("cancel_generating"));
        assert!(caps.model_matches(std::path::Path::new(r"C:\AI\sdcpp\models\qwen21\qwen-image-2.1-UC-Q4_K_M.gguf")));
        assert!(!caps.model_matches(std::path::Path::new(r"C:\AI\sdcpp\models\other.gguf")));
        assert_eq!(caps.limits.max_width, 4096);
        assert!(!caps.upscale && caps.upscalers.is_empty(), "older servers: no upscaler fields");

        let caps: Capabilities = serde_json::from_str(
            r#"{"upscale":true,"upscalers":[{"name":"Lanczos","model":false,"image_upscale":false},
                {"name":"RealESRGAN_x4plus","model":true,"image_upscale":true},{"name":"ltx_latent","model":true,"image_upscale":false}]}"#,
        )
        .unwrap();
        assert!(caps.upscale);
        assert!(caps.has_upscaler("RealESRGAN_x4plus", true));
        assert!(caps.has_upscaler("ltx_latent", false) && !caps.has_upscaler("ltx_latent", true));
        assert!(!caps.has_upscaler("Lanczos", false), "built-in filters are not models");
    }

    #[test]
    fn base64_roundtrip_and_data_urls() {
        let data = vec![0u8, 1, 2, 250];
        assert_eq!(b64_decode(&b64(&data)).unwrap(), data);
        assert_eq!(b64_decode(&format!("data:image/png;base64,{}", b64(&data))).unwrap(), data);
        assert!(b64_decode("!!!").is_err());
    }

    #[test]
    fn version_banner() {
        assert_eq!(parse_version_banner("stable-diffusion.cpp version unknown, commit 3f8527a\nUsage: ...").unwrap(), "version unknown, commit 3f8527a");
        assert!(parse_version_banner("nothing").is_none());
    }
}
