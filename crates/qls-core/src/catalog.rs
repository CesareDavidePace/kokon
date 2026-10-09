//! What the first-run wizard installs: the stable-diffusion.cpp build the app was verified
//! with, the official Qwen-Image 2.1 weights, and the VRAM presets that choose between them.
//!
//! Every file is pinned by URL, size and SHA-256 (from the Hugging Face / GitHub APIs,
//! 2026-10-08). If a link moves, the wizard reports it and asks the user to download the
//! file by hand from `page` and to open an issue at [`ISSUES_URL`].

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::{GpuBackend, TextEncoderPlacement};

/// Where users report broken download links. Set before publishing the repository.
pub const ISSUES_URL: &str = "https://github.com/GabrieleLozupone/kokon/issues/new";

/// Folder under the stable-diffusion.cpp folder that holds the Qwen weights.
pub const MODEL_SUBDIR: &str = "models/qwen21";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    /// stable-diffusion.cpp binaries (a zip extracted into the folder).
    Engine,
    Model,
    /// Not needed to run (the ESRGAN upscaler; Lanczos is used without it).
    Optional,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Download {
    pub id: &'static str,
    pub label: &'static str,
    pub kind: FileKind,
    pub url: &'static str,
    /// Page where the file can be downloaded by hand when `url` stops working.
    pub page: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// Destination relative to the stable-diffusion.cpp folder (for a zip: the archive path,
    /// extracted into its own folder and then deleted).
    pub dest: &'static str,
    /// Files a zip provides (relative to the stable-diffusion.cpp folder); their presence
    /// means it is installed.
    pub provides: &'static [&'static str],
}

/// `base` + a catalog path written with `/`, as native components.
pub fn join_rel(base: &Path, rel: &str) -> PathBuf {
    rel.split('/').filter(|c| !c.is_empty()).fold(base.to_path_buf(), |p, c| p.join(c))
}

impl Download {
    pub fn is_zip(&self) -> bool {
        !self.provides.is_empty()
    }

    pub fn path(&self, sdcpp_dir: &Path) -> PathBuf {
        join_rel(sdcpp_dir, self.dest)
    }

    /// Installed: a zip's files exist, or the file exists with the expected size. The hash is
    /// checked after every download; a size match is enough for files already on disk.
    pub fn is_present(&self, sdcpp_dir: &Path) -> bool {
        if self.is_zip() {
            return self.provides.iter().all(|f| join_rel(sdcpp_dir, f).is_file());
        }
        std::fs::metadata(self.path(sdcpp_dir)).map(|m| m.is_file() && m.len() == self.size).unwrap_or(false)
    }
}

const SDCPP_RELEASE: &str = "https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-929-3f8527a";

/// stable-diffusion.cpp commit 3f8527a (2026-09-27): the build every backend fact in the
/// README was verified on. CUDA 12 build; it runs on RTX 20xx to RTX 50xx.
pub const ENGINE_CUDA: Download = Download {
    id: "engine_cuda",
    label: "stable-diffusion.cpp (CUDA 12)",
    kind: FileKind::Engine,
    url: "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-929-3f8527a/sd-master-3f8527a-bin-win-cuda12-x64.zip",
    page: SDCPP_RELEASE,
    size: 337_915_440,
    sha256: "217d6dead9abd3f827fc338268555cc179234e7e6330ef21ecb1c985e19d2dc7",
    dest: "sd-master-3f8527a-bin-win-cuda12-x64.zip",
    provides: &["sd-server.exe", "stable-diffusion.dll", "ggml-cuda.dll"],
};

pub const CUDA_RUNTIME: Download = Download {
    id: "cuda_runtime",
    label: "CUDA 12 runtime (cuBLAS)",
    kind: FileKind::Engine,
    url: "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-929-3f8527a/cudart-sd-bin-win-cu12-x64.zip",
    page: SDCPP_RELEASE,
    size: 563_452_046,
    sha256: "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d",
    dest: "cudart-sd-bin-win-cu12-x64.zip",
    provides: &["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"],
};

/// Same commit, Vulkan build: AMD, Intel and NVIDIA cards through the GPU driver's Vulkan
/// runtime (nothing else to install). Kept in its own folder so it can sit next to the
/// CUDA build. SHA-256 from GitHub's asset digest, checked against a download 2026-10-09.
pub const ENGINE_VULKAN: Download = Download {
    id: "engine_vulkan",
    label: "stable-diffusion.cpp (Vulkan)",
    kind: FileKind::Engine,
    url: "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-929-3f8527a/sd-master-3f8527a-bin-win-vulkan-x64.zip",
    page: SDCPP_RELEASE,
    size: 30_067_605,
    sha256: "60e6850d650417409f18c2170ab5e27335db96da70cd3d1ad1e930bcffc2fd35",
    dest: "engine-vulkan/sd-master-3f8527a-bin-win-vulkan-x64.zip",
    provides: &["engine-vulkan/sd-server.exe", "engine-vulkan/stable-diffusion.dll", "engine-vulkan/ggml-vulkan.dll"],
};

/// Same commit, macOS build: universal binary with the Metal shaders embedded, needs
/// macOS 26. Ad-hoc signed; downloaded by the app (not a browser) it gets no quarantine flag.
pub const ENGINE_METAL: Download = Download {
    id: "engine_metal",
    label: "stable-diffusion.cpp (Metal)",
    kind: FileKind::Engine,
    url: "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-929-3f8527a/sd-master-3f8527a-bin-Darwin-macOS-26.6.2-arm64.zip",
    page: SDCPP_RELEASE,
    size: 35_049_086,
    sha256: "1c8ee6c8e413e3335b1223bbc657ea5d86dae1819f8426a98b84c265587eb192",
    dest: "engine-metal/sd-master-3f8527a-bin-Darwin-macOS-26.6.2-arm64.zip",
    provides: &["engine-metal/sd-server", "engine-metal/libstable-diffusion.dylib"],
};

/// The engine files of a backend, in download order.
pub fn engine_downloads(backend: GpuBackend) -> Vec<Download> {
    match backend {
        GpuBackend::Cuda => vec![ENGINE_CUDA, CUDA_RUNTIME],
        GpuBackend::Vulkan => vec![ENGINE_VULKAN],
        GpuBackend::Metal => vec![ENGINE_METAL],
    }
}

const UNSLOTH: &str = "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/tree/main";
const QWEN_TE: &str = "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main";

/// Official Qwen-Image 2.1 weights, 4-bit (Unsloth GGUF of Qwen/Qwen-Image-2.1).
pub const DIFFUSION_Q4_K_M: Download = Download {
    id: "diffusion_q4_k_m",
    label: "Qwen-Image 2.1 Q4_K_M",
    kind: FileKind::Model,
    url: "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf",
    page: UNSLOTH,
    size: 4_199_565_024,
    sha256: "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe",
    dest: "models/qwen21/qwen-image-2.1-Q4_K_M.gguf",
    provides: &[],
};

/// Official weights, 8-bit: closest to the original, for cards with room to spare.
pub const DIFFUSION_Q8_0: Download = Download {
    id: "diffusion_q8_0",
    label: "Qwen-Image 2.1 Q8_0",
    kind: FileKind::Model,
    url: "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q8_0.gguf",
    page: UNSLOTH,
    size: 7_640_860_384,
    sha256: "c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5",
    dest: "models/qwen21/qwen-image-2.1-Q8_0.gguf",
    provides: &[],
};

pub const VAE: Download = Download {
    id: "vae",
    label: "Qwen-Image 2.1 VAE",
    kind: FileKind::Model,
    url: "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
    page: "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae",
    size: 675_509_688,
    sha256: "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
    dest: "models/qwen21/qwen_image_2.1_vae_bf16.safetensors",
    provides: &[],
};

/// Text encoder from Qwen's own GGUF repository (the one sd.cpp's docs point to).
pub const TEXT_ENCODER_Q4_K_M: Download = Download {
    id: "text_encoder_q4_k_m",
    label: "Qwen3-VL 8B text encoder Q4_K_M",
    kind: FileKind::Model,
    url: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    page: QWEN_TE,
    size: 5_027_784_800,
    sha256: "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2",
    dest: "models/qwen21/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    provides: &[],
};

pub const TEXT_ENCODER_Q8_0: Download = Download {
    id: "text_encoder_q8_0",
    label: "Qwen3-VL 8B text encoder Q8_0",
    kind: FileKind::Model,
    url: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q8_0.gguf",
    page: QWEN_TE,
    size: 8_709_519_456,
    sha256: "0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0",
    dest: "models/qwen21/Qwen3VL-8B-Instruct-Q8_0.gguf",
    provides: &[],
};

/// Vision projector: lets the text encoder see reference images (needed for editing).
pub const VISION_PROJECTOR: Download = Download {
    id: "vision_projector",
    label: "Qwen3-VL vision projector",
    kind: FileKind::Model,
    url: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    page: QWEN_TE,
    size: 1_159_029_824,
    sha256: "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38",
    dest: "models/qwen21/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    provides: &[],
};

/// Default ESRGAN model (CC BY-NC-SA 4.0). Must match `config::UPSCALER_CHOICES[0]`.
pub const UPSCALER: Download = Download {
    id: "upscaler",
    label: "4x-UltraSharp upscaler",
    kind: FileKind::Optional,
    url: "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
    page: "https://huggingface.co/uwg/upscaler/tree/main/ESRGAN",
    size: 66_961_958,
    sha256: "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
    dest: "models/upscalers/4x-UltraSharp.pth",
    provides: &[],
};

/// Hardware presets: which weights to download and how to run them, by VRAM.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    /// Smallest reported VRAM (MiB) this preset is recommended for.
    pub min_vram_mib: u64,
    /// Name on a Mac: the unified memory sizes it is chosen for (through
    /// `planner::unified_memory_planning_mib`), since "12 GB" would read as RAM there.
    pub unified_label: &'static str,
    /// `summary` for a Mac (no "system RAM": it is the same memory).
    pub unified_summary: &'static str,
    pub diffusion: Download,
    pub text_encoder: Download,
    pub text_encoder_placement: TextEncoderPlacement,
    /// Measured on real hardware of this class (otherwise only derived).
    pub verified: bool,
    pub summary: &'static str,
}

/// Ordered by VRAM. Cards report a bit less than their nominal size (16 GB → 16303 MiB),
/// so thresholds sit between classes.
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "12gb",
        label: "12 GB",
        unified_label: "16–24 GB Mac",
        unified_summary: "4-bit image model and text encoder; images of about 1–1.5 megapixels",
        min_vram_mib: 0,
        diffusion: DIFFUSION_Q4_K_M,
        text_encoder: TEXT_ENCODER_Q4_K_M,
        text_encoder_placement: TextEncoderPlacement::Ram,
        verified: true,
        summary: "4-bit model; text encoder in system RAM; smaller working sizes",
    },
    Preset {
        id: "16gb",
        label: "16 GB",
        unified_label: "32–36 GB Mac",
        unified_summary: "4-bit image model and text encoder; native 2K for Quality",
        min_vram_mib: 14 * 1024,
        diffusion: DIFFUSION_Q4_K_M,
        text_encoder: TEXT_ENCODER_Q4_K_M,
        text_encoder_placement: TextEncoderPlacement::Ram,
        verified: true,
        summary: "4-bit model; text encoder in system RAM; native 2K for Quality",
    },
    Preset {
        id: "24gb",
        label: "24 GB",
        unified_label: "48 GB Mac",
        unified_summary: "8-bit image model (closest to the original); larger edits fit",
        min_vram_mib: 20 * 1024,
        diffusion: DIFFUSION_Q8_0,
        text_encoder: TEXT_ENCODER_Q4_K_M,
        text_encoder_placement: TextEncoderPlacement::Ram,
        verified: false,
        summary: "8-bit model (closest to the original); larger edits fit",
    },
    Preset {
        id: "32gb",
        label: "32 GB+",
        unified_label: "64 GB+ Mac",
        unified_summary: "8-bit image model and 8-bit text encoder",
        min_vram_mib: 28 * 1024,
        diffusion: DIFFUSION_Q8_0,
        text_encoder: TEXT_ENCODER_Q8_0,
        text_encoder_placement: TextEncoderPlacement::Ram,
        verified: false,
        summary: "8-bit model and 8-bit text encoder",
    },
];

pub const DEFAULT_PRESET: &str = "16gb";

pub fn preset(id: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.id == id)
}

/// The preset for a card with `vram_mib` (None: unknown GPU → the 16 GB default).
pub fn recommended_preset(vram_mib: Option<u64>) -> &'static Preset {
    match vram_mib {
        Some(v) => PRESETS.iter().rev().find(|p| v >= p.min_vram_mib).unwrap_or(&PRESETS[0]),
        None => preset(DEFAULT_PRESET).unwrap(),
    }
}

/// Everything a preset needs on `backend`, in download order (small, essential files first).
pub fn downloads_for(p: &Preset, backend: GpuBackend, with_upscaler: bool) -> Vec<Download> {
    let mut v = engine_downloads(backend);
    v.extend([VAE, VISION_PROJECTOR, p.text_encoder, p.diffusion]);
    if with_upscaler {
        v.push(UPSCALER);
    }
    v
}

/// Folder a zip is extracted into: the folder of its `dest`.
pub fn zip_dir(d: &Download, sdcpp_dir: &Path) -> PathBuf {
    d.path(sdcpp_dir).parent().map(Path::to_path_buf).unwrap_or_else(|| sdcpp_dir.to_path_buf())
}

/// Message shown when a download keeps failing (moved link, changed file, server down).
pub fn manual_download_hint(d: &Download, sdcpp_dir: &Path) -> String {
    let target = if d.is_zip() { format!("extract it into {}", zip_dir(d, sdcpp_dir).display()) } else { format!("save it as {}", d.path(sdcpp_dir).display()) };
    format!(
        "Download {} by hand from {} and {target}, then press Check again. Please also open an issue at {ISSUES_URL} so the link can be fixed.",
        d.label, d.page
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_follow_vram() {
        assert_eq!(recommended_preset(Some(12_288)).id, "12gb");
        assert_eq!(recommended_preset(Some(11_264)).id, "12gb");
        assert_eq!(recommended_preset(Some(8_192)).id, "12gb", "smaller cards get the lightest preset");
        assert_eq!(recommended_preset(Some(16_303)).id, "16gb");
        assert_eq!(recommended_preset(Some(24_564)).id, "24gb");
        assert_eq!(recommended_preset(Some(32_607)).id, "32gb");
        assert_eq!(recommended_preset(Some(49_140)).id, "32gb");
        assert_eq!(recommended_preset(None).id, "16gb");
    }

    #[test]
    fn catalog_is_well_formed() {
        let all = [ENGINE_CUDA, CUDA_RUNTIME, ENGINE_VULKAN, ENGINE_METAL, DIFFUSION_Q4_K_M, DIFFUSION_Q8_0, VAE, TEXT_ENCODER_Q4_K_M, TEXT_ENCODER_Q8_0, VISION_PROJECTOR, UPSCALER];
        for d in all {
            assert!(d.url.starts_with("https://"), "{}", d.id);
            assert_eq!(d.sha256.len(), 64, "{}", d.id);
            assert!(d.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), "{}", d.id);
            assert!(d.size > 0);
            assert!(!d.dest.contains('\\') && d.provides.iter().all(|f| !f.contains('\\')), "portable paths: {}", d.id);
            assert_eq!(d.is_zip(), d.kind == FileKind::Engine, "{}", d.id);
        }
        // The default upscaler matches the one Settings offers first.
        assert!(UPSCALER.dest.ends_with(&format!("{}.pth", crate::config::DEFAULT_UPSCALER)));
        for p in PRESETS {
            assert_eq!(downloads_for(p, GpuBackend::Cuda, true).len(), 7);
            let vk = downloads_for(p, GpuBackend::Vulkan, false);
            assert_eq!(vk.len(), 5);
            assert!(vk.iter().all(|d| d.id != "engine_cuda" && d.id != "cuda_runtime"));
        }
    }

    #[test]
    fn presence_checks() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!VAE.is_present(dir.path()));
        std::fs::create_dir_all(join_rel(dir.path(), MODEL_SUBDIR)).unwrap();
        std::fs::write(VAE.path(dir.path()), b"short").unwrap();
        assert!(!VAE.is_present(dir.path()), "a partial file is not installed");
        for f in ENGINE_CUDA.provides {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        assert!(ENGINE_CUDA.is_present(dir.path()));
        let hint = manual_download_hint(&ENGINE_CUDA, dir.path());
        assert!(hint.contains("extract it into") && hint.contains(ISSUES_URL));
        // The Vulkan build lives in its own folder: the CUDA files don't count for it.
        assert!(!ENGINE_VULKAN.is_present(dir.path()));
        std::fs::create_dir_all(dir.path().join("engine-vulkan")).unwrap();
        for f in ENGINE_VULKAN.provides {
            std::fs::write(join_rel(dir.path(), f), b"x").unwrap();
        }
        assert!(ENGINE_VULKAN.is_present(dir.path()));
        let mut cfg = crate::config::AppConfig::default();
        cfg.sdcpp_dir = dir.path().to_path_buf();
        cfg.gpu_backend = GpuBackend::Vulkan;
        if cfg!(windows) {
            // The Vulkan build is a Windows download (sd-server.exe).
            assert_eq!(cfg.server_exe(), join_rel(dir.path(), ENGINE_VULKAN.provides[0]), "installer and launcher agree on the path");
        }
        cfg.gpu_backend = GpuBackend::Metal;
        if !cfg!(windows) {
            assert_eq!(cfg.server_exe(), join_rel(dir.path(), ENGINE_METAL.provides[0]));
        }
        assert!(manual_download_hint(&ENGINE_VULKAN, dir.path()).contains("engine-vulkan"));
    }
}
