//! Executes a [`Plan`] at the pixel level: builds the backend request (exact working
//! geometry, binary masks) and post-processes results (crop alignment padding, restore
//! dimensions, composite inpaint/outpaint so untouched pixels stay bit-identical).
//! Enlargements go through ESRGAN when the engine has it: [`crop_content`], then
//! [`enlarge_target`] / [`esrgan_plan`] + [`Stitcher`], then [`finish`].

use image::{GrayImage, Luma, Rgba, RgbaImage};

use crate::error::{AppError, AppResult, ErrorCode};
use crate::geometry::{Rect, Size};
use crate::imaging::{self, LoadedImage};
use crate::planner::{Plan, ResolvedMode};
use crate::sdcpp::{Guidance, ImgGenRequest, SampleParams, VaeTilingParams, b64};

/// Server-side facts that shape the request but are not part of the plan.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// `--vae-tile-size` the server was launched with (0 = no server-side tiling).
    pub server_vae_tile_size: u32,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self { server_vae_tile_size: 0 }
    }
}

/// Tile size used when the server has no tiling configured (2K decode: ~4 s vs ~22 s at 256 px).
pub const DEFAULT_VAE_TILE: u32 = 768;

/// Per-request VAE tiling. The request is merged over the server's launch defaults, so
/// when the server already tiles (`--vae-tiling --vae-tile-size N`) nothing is sent —
/// sending `tile_size 0` would silently select 256 px tiles. Only an explicit "off"
/// overrides it.
pub fn vae_tiling_params(tiling: bool, server_tile: u32) -> Option<VaeTilingParams> {
    match (tiling, server_tile > 0) {
        (true, true) | (false, false) => None,
        (false, true) => Some(VaeTilingParams { enabled: false, ..Default::default() }),
        (true, false) => Some(VaeTilingParams { enabled: true, tile_size_w: Some(DEFAULT_VAE_TILE), tile_size_h: Some(DEFAULT_VAE_TILE), target_overlap: Some(0.125) }),
    }
}

pub struct PreparedJob {
    pub request: ImgGenRequest,
    pub post: PostProcess,
}

pub enum PostKind {
    Plain,
    Inpaint { source: RgbaImage, region: Rect, blend: GrayImage },
    Outpaint { source: RgbaImage, feather: u32 },
}

pub struct PostProcess {
    pub kind: PostKind,
    /// Qwen-Image 2.1 always decodes an alpha channel; on opaque content ~1% of pixels
    /// come back at alpha 247–254. When transparency is not intended, force opaque.
    pub keep_alpha: bool,
    pub working: Size,
    pub content: Rect,
    pub output: Size,
    pub mode: ResolvedMode,
    pub outpaint: Option<crate::geometry::OutpaintPlan>,
}

fn png_b64(img: &RgbaImage) -> AppResult<String> {
    Ok(b64(&imaging::encode_png(img, &[], None, true)?))
}

fn gray_png_b64(m: &GrayImage) -> AppResult<String> {
    let mut out = Vec::new();
    image::DynamicImage::ImageLuma8(m.clone())
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| AppError::internal(e))?;
    Ok(b64(&out))
}

/// Places `img` (resized to the canvas content) on the plan's padded canvas.
fn to_canvas(img: &RgbaImage, c: &crate::geometry::WorkingCanvas) -> RgbaImage {
    let resized = imaging::resize(img, c.content.width, c.content.height);
    imaging::pad_edge(&resized, c.canvas, c.content_offset)
}

fn mean_color(img: &RgbaImage) -> Rgba<u8> {
    let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
    let step = ((img.width() as u64 * img.height() as u64) / 65_536).max(1) as usize;
    for p in img.pixels().step_by(step) {
        r += p[0] as u64;
        g += p[1] as u64;
        b += p[2] as u64;
        n += 1;
    }
    let n = n.max(1);
    Rgba([(r / n) as u8, (g / n) as u8, (b / n) as u8, 255])
}

fn is_binary(m: &GrayImage) -> bool {
    m.pixels().all(|p| p[0] == 0 || p[0] == 255)
}

const RGBA_PREFIX: &str = "This is an RGBA image with transparency.";
const RGBA_SUFFIX: &str = "The image has alpha channel and the background is transparent.";

/// True when the prompt asks for transparency (e.g. "transparent background", "remove the background").
pub fn wants_transparency(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    ["transparent", "transparency", "alpha channel", "rgba", "no background", "without background", "remove the background", "remove background", "cut out", "cutout"]
        .iter()
        .any(|k| p.contains(k))
}

/// Prompt sent to the model. Transparency requests use the prompt format recommended for
/// Qwen-Image 2.1 RGBA output; outpainting gets an instruction prefix so short prompts
/// like "extend left" still describe the task.
pub fn effective_prompt(plan: &Plan, prompt: &str) -> String {
    let p = prompt.trim();
    if wants_transparency(p) && !p.starts_with(RGBA_PREFIX) {
        let body = p.trim_end_matches('.');
        return format!("{RGBA_PREFIX} {body}. {RGBA_SUFFIX}");
    }
    match plan.mode {
        ResolvedMode::Outpaint => format!(
            "Extend the image naturally into the empty border areas, continuing the scene with matching perspective, lighting and style. {}",
            prompt.trim()
        ),
        _ => prompt.trim().to_string(),
    }
}

pub fn prepare(plan: &Plan, prompt: &str, seed: i64, images: &[LoadedImage], mask: Option<&GrayImage>, opts: &BackendOptions) -> AppResult<PreparedJob> {
    let first = plan.working;
    let mut req = ImgGenRequest {
        prompt: effective_prompt(plan, prompt),
        negative_prompt: plan.negative_prompt.clone(),
        width: first.width,
        height: first.height,
        seed,
        batch_count: plan.batch_count,
        strength: None,
        init_image: None,
        mask_image: None,
        ref_images: Vec::new(),
        image_preprocess: Vec::new(),
        sample_params: SampleParams {
            sample_method: Some(plan.sampler.clone()),
            scheduler: plan.scheduler.clone(),
            sample_steps: plan.steps,
            flow_shift: plan.flow_shift,
            guidance: Guidance { txt_cfg: plan.cfg, img_cfg: None },
        },
        vae_tiling_params: vae_tiling_params(plan.vae_tiling, opts.server_vae_tile_size),
        hires: None,
        cache_mode: plan.cache_threshold.map(|_| "easycache".to_string()),
        cache_option: plan.cache_threshold.map(|t| format!("threshold={t}")),
        output_format: "png".into(),
        embed_image_metadata: false,
    };
    let mut kind = PostKind::Plain;
    let keep_alpha = wants_transparency(prompt) || images.iter().any(|i| i.has_transparency);
    let need_images = |n: usize| -> AppResult<()> {
        if images.len() < n {
            Err(AppError::new(ErrorCode::InvalidInput, "An input image is missing."))
        } else {
            Ok(())
        }
    };

    match plan.mode {
        ResolvedMode::Generate => {}
        ResolvedMode::Edit | ResolvedMode::MultiReference => {
            need_images(plan.references.len())?;
            for r in &plan.references {
                let img = &images[r.index].pixels;
                req.ref_images.push(png_b64(&to_canvas(img, &r.canvas))?);
            }
            req.image_preprocess.push("target=ref,mode=none".into());
            if let Some(s) = plan.strength {
                req.strength = Some(s);
            }
        }
        ResolvedMode::Inpaint => {
            need_images(1)?;
            let src = &images[0].pixels;
            let src_size = Size::new(src.width(), src.height());
            let user_mask = mask.ok_or_else(|| AppError::invalid("A mask is required for inpainting."))?;
            let user_mask = imaging::resize_mask(user_mask, src_size.width, src_size.height);
            let region = plan.inpaint_region.unwrap_or(Rect::new(0, 0, src_size.width, src_size.height));
            let wc = plan.references[0].canvas;
            let crop = imaging::crop(src, region);
            let canvas = to_canvas(&crop, &wc);
            let canvas_b64 = png_b64(&canvas)?;
            drop(canvas);

            // Blend mask (source resolution): soft edges so the patch merges invisibly.
            let mut blend = imaging::crop_mask(&user_mask, region);
            if is_binary(&blend) {
                let r = ((region.width.max(region.height) as f64) * 0.006).round().max(3.0) as u32;
                blend = imaging::feather(&imaging::dilate(&blend, r / 2), r);
            }
            // Backend mask (working resolution): binary, grown to cover the soft edge.
            let m_work = imaging::resize_mask(&blend, wc.content.width, wc.content.height);
            let grow = ((wc.scale * 4.0).round() as u32).clamp(4, 24);
            let m_work = imaging::backend_mask(&m_work, grow);
            let mut m_canvas = GrayImage::from_pixel(wc.canvas.width, wc.canvas.height, Luma([0]));
            image::imageops::replace(&mut m_canvas, &m_work, wc.content_offset.0 as i64, wc.content_offset.1 as i64);

            req.init_image = Some(canvas_b64.clone());
            req.ref_images.push(canvas_b64);
            req.mask_image = Some(gray_png_b64(&m_canvas)?);
            req.strength = plan.strength;
            req.image_preprocess = vec!["target=init,mode=none".into(), "target=mask,mode=none,filter=nearest-exact".into(), "target=ref,mode=none".into()];
            kind = PostKind::Inpaint { source: src.clone(), region, blend };
        }
        ResolvedMode::Outpaint => {
            need_images(1)?;
            let src = &images[0].pixels;
            let op = plan.outpaint.ok_or_else(|| AppError::internal("missing outpaint plan"))?;
            let r = op.source_rect_working;
            let scaled = imaging::resize(src, r.width, r.height);
            // Unknown areas get the image's mean color: neutral, no edge streaks.
            let canvas = imaging::pad_solid(&scaled, op.working.canvas, (r.x, r.y), mean_color(src));
            let mask = imaging::outpaint_mask(op.working.canvas, r, op.overlap_working);
            req.init_image = Some(png_b64(&canvas)?);
            req.mask_image = Some(gray_png_b64(&mask)?);
            req.ref_images.push(png_b64(&to_canvas(src, &plan.references[0].canvas))?);
            req.strength = plan.strength;
            req.image_preprocess = vec!["target=init,mode=none".into(), "target=mask,mode=none,filter=nearest-exact".into(), "target=ref,mode=none".into()];
            let feather_final = ((op.overlap_working.left.max(op.overlap_working.right).max(op.overlap_working.top).max(op.overlap_working.bottom)) as f64
                / op.working.scale)
                .round() as u32;
            kind = PostKind::Outpaint { source: src.clone(), feather: feather_final.max(2) };
        }
        ResolvedMode::Upscale => {
            // Detailed upscale: plain img2img over the (already enlarged) image.
            need_images(1)?;
            if plan.steps == 0 {
                return Err(AppError::internal("a quick upscale has no diffusion request"));
            }
            let canvas = to_canvas(&images[0].pixels, &plan.references[0].canvas);
            req.init_image = Some(png_b64(&canvas)?);
            req.strength = plan.strength;
            req.image_preprocess = vec!["target=init,mode=none".into()];
        }
    }

    Ok(PreparedJob {
        request: req,
        post: PostProcess { kind, keep_alpha, working: plan.working, content: plan.content, output: plan.output, mode: plan.mode, outpaint: plan.outpaint },
    })
}

/// Turns one backend output image into the delivered image (Lanczos for any resize).
pub fn postprocess(raw: RgbaImage, post: &PostProcess) -> AppResult<RgbaImage> {
    finish(crop_content(raw, post), post)
}

/// Step 1: the meaningful part of a backend image (alignment padding removed).
pub fn crop_content(raw: RgbaImage, post: &PostProcess) -> RgbaImage {
    let raw = if raw.width() != post.working.width || raw.height() != post.working.height {
        tracing::warn!("backend returned {}x{}, expected {:?}", raw.width(), raw.height(), post.working);
        imaging::resize(&raw, post.working.width, post.working.height)
    } else {
        raw
    };
    let mut content = imaging::crop(&raw, post.content);
    drop(raw);
    if !post.keep_alpha {
        for p in content.pixels_mut() {
            p[3] = 255;
        }
    }
    content
}

/// Size the content is resized to in [`finish`], when ESRGAN may help. Inpaint patches
/// are generated near native size and keep the plain path.
pub fn enlarge_target(post: &PostProcess) -> Option<Size> {
    match &post.kind {
        PostKind::Plain => Some(post.output),
        PostKind::Outpaint { .. } => post.outpaint.map(|o| o.final_size),
        PostKind::Inpaint { .. } => None,
    }
}

/// Step 3: resize (now usually *down* from an ESRGAN result) and composite. `content`
/// is the output of [`crop_content`], optionally enlarged in between.
pub fn finish(content: RgbaImage, post: &PostProcess) -> AppResult<RgbaImage> {
    Ok(match &post.kind {
        PostKind::Plain => imaging::resize(&content, post.output.width, post.output.height),
        PostKind::Inpaint { source, region, blend } => {
            let patch = imaging::resize(&content, region.width, region.height);
            let mut out = source.clone();
            imaging::composite(&mut out, &patch, blend, (region.x, region.y));
            out
        }
        PostKind::Outpaint { source, feather } => {
            let op = post.outpaint.as_ref().ok_or_else(|| AppError::internal("missing outpaint plan"))?;
            let mut out = imaging::resize(&content, op.final_size.width, op.final_size.height);
            let paste = imaging::paste_back_mask(Size::new(source.width(), source.height()), op.expansion, *feather);
            imaging::composite(&mut out, source, &paste, (op.source_rect_final.x, op.source_rect_final.y));
            out
        }
    })
}

// ---------------------------------------------------------------------------------------
// ESRGAN enlargement (POST /sdcpp/v1/upscale)
// ---------------------------------------------------------------------------------------

/// Results per side of one `/sdcpp/v1/upscale` call (server limit, routes_sdcpp.cpp).
pub const ESRGAN_MAX_OUTPUT_SIDE: u32 = 8192;
/// Context shared by neighbouring tiles; ESRGAN's receptive field is far smaller.
pub const ESRGAN_TILE_OVERLAP: u32 = 32;
/// Cap on the intermediate x4 image (~400 MB RGBA); larger inputs are shrunk first.
pub const ESRGAN_MAX_OUTPUT_AREA: u64 = 100_000_000;
/// Below this linear factor Lanczos is as good and much cheaper.
pub const ESRGAN_MIN_FACTOR: f64 = 1.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Region sent to the upscaler (input coordinates).
    pub rect: Rect,
    /// Part of the tile kept in the stitched result; cores cover the image without gaps.
    pub core: Rect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EsrganPlan {
    /// Size the content is resized to before upscaling (usually unchanged).
    pub input: Size,
    pub tiles: Vec<Tile>,
}

fn axis_tiles(len: u32, max: u32, overlap: u32) -> Vec<(u32, u32, u32, u32)> {
    if len <= max {
        return vec![(0, len, 0, len)];
    }
    let n = (len - overlap).div_ceil(max - overlap);
    let tile = (len + (n - 1) * overlap).div_ceil(n).min(max);
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let start = (i * (tile - overlap)).min(len - tile);
        let end = start + tile;
        let core_start = if i == 0 { 0 } else { start + overlap / 2 };
        let core_end = if i == n - 1 { len } else { end - overlap / 2 };
        out.push((start, end, core_start, core_end));
    }
    out
}

/// Overlapping tiles no larger than `max` per side.
pub fn tile_grid(size: Size, max: u32, overlap: u32) -> Vec<Tile> {
    let xs = axis_tiles(size.width, max, overlap);
    let ys = axis_tiles(size.height, max, overlap);
    let mut tiles = Vec::with_capacity(xs.len() * ys.len());
    for &(y0, y1, cy0, cy1) in &ys {
        for &(x0, x1, cx0, cx1) in &xs {
            tiles.push(Tile { rect: Rect::new(x0, y0, x1 - x0, y1 - y0), core: Rect::new(cx0, cy0, cx1 - cx0, cy1 - cy0) });
        }
    }
    tiles
}

/// How to enlarge `content` to `target` with an x`scale` ESRGAN model; None when the
/// factor is small enough for Lanczos.
pub fn esrgan_plan(content: Size, target: Size, scale: u32) -> Option<EsrganPlan> {
    let factor = (target.width as f64 / content.width as f64).max(target.height as f64 / content.height as f64);
    if factor <= ESRGAN_MIN_FACTOR || scale < 2 {
        return None;
    }
    let s2 = (scale * scale) as u64;
    let input = if content.area() * s2 > ESRGAN_MAX_OUTPUT_AREA {
        let k = (ESRGAN_MAX_OUTPUT_AREA as f64 / (content.area() * s2) as f64).sqrt();
        Size::new(((content.width as f64 * k).floor() as u32).max(1), ((content.height as f64 * k).floor() as u32).max(1))
    } else {
        content
    };
    Some(EsrganPlan { input, tiles: tile_grid(input, ESRGAN_MAX_OUTPUT_SIDE / scale, ESRGAN_TILE_OVERLAP) })
}

/// Assembles upscaled tiles. The model's real factor comes from the first result.
pub struct Stitcher {
    input: Size,
    out: Option<RgbaImage>,
    scale: u32,
}

impl Stitcher {
    pub fn new(input: Size) -> Self {
        Self { input, out: None, scale: 0 }
    }

    pub fn add(&mut self, tile: &Tile, up: &RgbaImage) -> AppResult<()> {
        let scale = up.width() / tile.rect.width.max(1);
        if scale < 1 || up.width() != tile.rect.width * scale || up.height() != tile.rect.height * scale {
            return Err(AppError::new(ErrorCode::GenerationFailed, "The upscaler returned an unexpected size.")
                .detail(format!("tile {:?} -> {}x{}", tile.rect, up.width(), up.height())));
        }
        if self.out.is_none() {
            self.scale = scale;
            self.out = Some(RgbaImage::new(self.input.width * scale, self.input.height * scale));
        } else if scale != self.scale {
            return Err(AppError::new(ErrorCode::GenerationFailed, "The upscaler changed its scale between tiles."));
        }
        let c = tile.core;
        let piece = image::imageops::crop_imm(up, (c.x - tile.rect.x) * scale, (c.y - tile.rect.y) * scale, c.width * scale, c.height * scale).to_image();
        image::imageops::replace(self.out.as_mut().unwrap(), &piece, (c.x * scale) as i64, (c.y * scale) as i64);
        Ok(())
    }

    /// The stitched image. ESRGAN models are RGB-only, so `alpha_from`'s alpha channel
    /// (if it has any transparency) is resized onto the result.
    pub fn finish(self, alpha_from: &RgbaImage) -> AppResult<RgbaImage> {
        let mut out = self.out.ok_or_else(|| AppError::internal("no upscaled tiles"))?;
        if alpha_from.pixels().any(|p| p[3] != 255) {
            let a = image::imageops::resize(alpha_from, out.width(), out.height(), image::imageops::FilterType::Lanczos3);
            for (o, s) in out.pixels_mut().zip(a.pixels()) {
                o[3] = s[3];
            }
        } else {
            for p in out.pixels_mut() {
                p[3] = 255;
            }
        }
        Ok(out)
    }
}

/// PNG (base64) of one tile for the upscale endpoint (alpha is dropped by the server).
pub fn tile_png_b64(input: &RgbaImage, tile: &Tile) -> AppResult<String> {
    png_b64(&imaging::crop(input, tile.rect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecommendedSettings;
    use crate::geometry::ExpansionUnits;
    use crate::config::RefineMode;
    use crate::planner::{GenerationRequest, Mode, OutpaintSpec, PlanInputs, Quality, ResolutionMode, UpscaleMethod, UpscaleSpec, plan};
    use crate::sdcpp::b64_decode;

    fn env(sizes: Vec<Size>) -> PlanInputs {
        PlanInputs { image_sizes: sizes, vram_total_mib: Some(16303), upscaler: Some("RealESRGAN_x4plus".into()), ..Default::default() }
    }
    fn esrgan_opts() -> BackendOptions {
        BackendOptions { server_vae_tile_size: 768 }
    }

    #[test]
    fn vae_tiling_never_overrides_the_server_tile_size() {
        assert_eq!(vae_tiling_params(true, 768), None, "server already tiles with its own size");
        assert_eq!(vae_tiling_params(false, 0), None);
        assert_eq!(vae_tiling_params(false, 768), Some(VaeTilingParams { enabled: false, ..Default::default() }));
        let p = vae_tiling_params(true, 0).unwrap();
        assert_eq!((p.enabled, p.tile_size_w, p.tile_size_h, p.target_overlap), (true, Some(768), Some(768), Some(0.125)));
        // Regression: a planned-tiled 2K job sent tile_size 0, i.e. 256 px tiles.
        let mut r = req("a cabin");
        r.aspect_ratio = Some("16:9".into());
        r.resolution = ResolutionMode::K2;
        let i = PlanInputs { server_vae_tiling: true, ..env(vec![]) };
        let p = plan(&r, &i, &RecommendedSettings::default()).unwrap();
        assert!(p.vae_tiling);
        let j = prepare(&p, &r.prompt, 1, &[], None, &esrgan_opts()).unwrap();
        assert!(serde_json::to_value(&j.request).unwrap().get("vae_tiling_params").is_none());
    }

    #[test]
    fn quality_generate_request_is_one_pass() {
        let mut r = req("a cozy mountain cabin");
        r.aspect_ratio = Some("16:9".into());
        r.quality = Quality::Quality;
        r.advanced.refine = Some(RefineMode::Best);
        let p = plan(&r, &env(vec![]), &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 42, &[], None, &esrgan_opts()).unwrap();
        let v = serde_json::to_value(&j.request).unwrap();
        assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(p.working.width as u64), Some(p.working.height as u64)));
        assert!(v.get("hires").map(|h| h.is_null()).unwrap_or(true), "no second pass");
        assert!(v.get("cache_mode").map(|c| c.is_null()).unwrap_or(true));
        assert_eq!(v["sample_params"]["sample_steps"], 40);
        assert_eq!(v["sample_params"]["guidance"]["txt_cfg"], 1.0, "official: no CFG");
        assert!(p.working.area() > 2_500_000, "Quality renders toward native 2K: {:?}", p.working);
    }

    #[test]
    fn large_edit_is_one_pass_then_enlarged() {
        let img = photo(2560, 1440);
        let mut r = req("make it a dark version");
        r.resolution = ResolutionMode::Source;
        let p = plan(&r, &env(vec![img.size()]), &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 3, std::slice::from_ref(&img), None, &esrgan_opts()).unwrap();
        assert!(j.request.hires.is_none());
        assert_eq!((j.request.width, j.request.height), (p.working.width, p.working.height));
        assert_eq!(decode(&j.request.ref_images[0]).dimensions(), (p.working.width, p.working.height), "reference matches the canvas");
        // The single-pass result is enlarged to the 2560x1440 source size with ESRGAN.
        let content = crop_content(fake_backend(p.working.width, p.working.height), &j.post);
        let target = enlarge_target(&j.post).unwrap();
        assert_eq!(target, Size::new(2560, 1440));
        let ep = esrgan_plan(Size::new(content.width(), content.height()), target, 4).unwrap();
        assert_eq!(ep.tiles.len(), 1, "fits one upscale call");
        let out = finish(content, &j.post).unwrap();
        assert_eq!(out.dimensions(), (2560, 1440));
    }

    #[test]
    fn detailed_upscale_is_img2img_at_2k() {
        let img = photo(1376, 768);
        let r = GenerationRequest {
            mode: Mode::Upscale,
            upscale: Some(UpscaleSpec { method: UpscaleMethod::Detailed, width: Some(5120), height: Some(2880), scale: None }),
            preserve_dimensions: true,
            ..Default::default()
        };
        let p = plan(&r, &env(vec![img.size()]), &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, crate::planner::UPSCALE_PROMPT, 9, std::slice::from_ref(&img), None, &esrgan_opts()).unwrap();
        assert_eq!((j.request.width, j.request.height), (2048, 1152));
        assert_eq!(decode(j.request.init_image.as_ref().unwrap()).dimensions(), (2048, 1152));
        assert!(j.request.ref_images.is_empty() && j.request.mask_image.is_none() && j.request.hires.is_none());
        assert_eq!(j.request.strength, Some(0.3));
        assert_eq!(j.request.image_preprocess, vec!["target=init,mode=none".to_string()]);
        assert_eq!(enlarge_target(&j.post), Some(Size::new(5120, 2880)));

        let quick = GenerationRequest { upscale: Some(UpscaleSpec::default()), ..r };
        let p = plan(&quick, &env(vec![img.size()]), &RecommendedSettings::default()).unwrap();
        assert!(prepare(&p, "", 9, std::slice::from_ref(&img), None, &esrgan_opts()).is_err(), "quick upscales never reach img_gen");
    }

    #[test]
    fn enlarge_targets_by_kind() {
        let img = photo(800, 600);
        let mut r = req("more beach");
        r.outpaint = Some(OutpaintSpec { left: 0.0, right: 50.0, top: 0.0, bottom: 0.0, units: ExpansionUnits::Percent });
        let p = plan(&r, &env(vec![img.size()]), &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 1, std::slice::from_ref(&img), None, &esrgan_opts()).unwrap();
        assert_eq!(enlarge_target(&j.post), Some(Size::new(1200, 600)));
        let mut m = GrayImage::new(800, 600);
        m.put_pixel(400, 300, Luma([255]));
        let mut r = req("remove it");
        r.mask = Some("m".into());
        let i = PlanInputs { mask: Some((Size::new(800, 600), imaging::mask_bbox(&m, 127), 0.01)), ..env(vec![img.size()]) };
        let p = plan(&r, &i, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 1, std::slice::from_ref(&img), Some(&m), &esrgan_opts()).unwrap();
        assert_eq!(enlarge_target(&j.post), None, "inpaint keeps its own path");
    }

    #[test]
    fn esrgan_vs_lanczos_choice() {
        let c = Size::new(1376, 768);
        assert!(esrgan_plan(c, Size::new(1500, 837), 4).is_none(), "1.09x: Lanczos");
        assert!(esrgan_plan(c, Size::new(1376, 768), 4).is_none());
        assert!(esrgan_plan(c, Size::new(1000, 558), 4).is_none(), "shrinking: Lanczos");
        let ep = esrgan_plan(c, Size::new(5120, 2880), 4).unwrap();
        assert_eq!(ep.input, c);
        assert_eq!(ep.tiles, vec![Tile { rect: Rect::new(0, 0, 1376, 768), core: Rect::new(0, 0, 1376, 768) }]);
        assert!(esrgan_plan(c, Size::new(5120, 2880), 1).is_none(), "not an enlarging model");
    }

    #[test]
    fn large_inputs_are_tiled_and_capped() {
        let ep = esrgan_plan(Size::new(3000, 1700), Size::new(6000, 3400), 4).unwrap();
        assert_eq!(ep.input, Size::new(3000, 1700), "an 82 MP intermediate is under the cap");
        assert_eq!(ep.tiles.len(), 2);
        for t in &ep.tiles {
            assert!(t.rect.width <= 2048 && t.rect.height <= 2048);
        }
        // Cores tile the image exactly once.
        let mut hits = vec![0u8; 3000 * 1700];
        for t in &ep.tiles {
            assert!(t.core.x >= t.rect.x && t.core.right() <= t.rect.right() && t.core.y >= t.rect.y && t.core.bottom() <= t.rect.bottom());
            for y in t.core.y..t.core.bottom() {
                for x in t.core.x..t.core.right() {
                    hits[(y * 3000 + x) as usize] += 1;
                }
            }
        }
        assert!(hits.iter().all(|h| *h == 1));
        let ep = esrgan_plan(Size::new(4000, 3000), Size::new(8000, 6000), 4).unwrap();
        assert!(ep.input.area() * 16 <= ESRGAN_MAX_OUTPUT_AREA, "{:?}", ep.input);
        assert!(ep.input.width * 4 >= 8000 && ep.input.height * 4 >= 6000, "still covers the target");
        assert!(tile_grid(Size::new(5000, 300), 2048, 32).iter().all(|t| t.rect.width <= 2048));
    }

    #[test]
    fn stitched_tiles_match_a_whole_image_upscale() {
        // A nearest-neighbour "model" is exact per pixel, so any seam would show.
        let x4 = |img: &RgbaImage| image::imageops::resize(img, img.width() * 4, img.height() * 4, image::imageops::FilterType::Nearest);
        let src = photo(300, 170).pixels;
        let tiles = tile_grid(Size::new(300, 170), 128, 16);
        assert!(tiles.len() > 4);
        let mut st = Stitcher::new(Size::new(300, 170));
        for t in &tiles {
            let png = tile_png_b64(&src, t).unwrap();
            st.add(t, &x4(&decode(&png))).unwrap();
        }
        let out = st.finish(&src).unwrap();
        assert_eq!(out, x4(&src));
        let mut bad = Stitcher::new(Size::new(300, 170));
        assert!(bad.add(&tiles[0], &RgbaImage::new(7, 7)).is_err());
    }

    #[test]
    fn stitched_result_keeps_source_alpha() {
        let mut src = photo(64, 64).pixels;
        for y in 0..32 {
            for x in 0..64 {
                src.get_pixel_mut(x, y)[3] = 0;
            }
        }
        let t = tile_grid(Size::new(64, 64), 2048, 32)[0];
        let mut st = Stitcher::new(Size::new(64, 64));
        let mut up = image::imageops::resize(&src, 256, 256, image::imageops::FilterType::Nearest);
        for p in up.pixels_mut() {
            p[3] = 255; // ESRGAN output is RGB
        }
        st.add(&t, &up).unwrap();
        let out = st.finish(&src).unwrap();
        assert_eq!(out.get_pixel(10, 10)[3], 0);
        assert_eq!(out.get_pixel(10, 250)[3], 255);
    }

    fn photo(w: u32, h: u32) -> LoadedImage {
        let pixels = RgbaImage::from_fn(w, h, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8, 255]));
        LoadedImage { pixels, format: "png", exif_orientation: 1, icc_profile: None, has_transparency: false, text: vec![] }
    }
    fn req(p: &str) -> GenerationRequest {
        GenerationRequest { prompt: p.into(), preserve_dimensions: true, ..Default::default() }
    }
    fn decode(b: &str) -> RgbaImage {
        imaging::decode_image(&b64_decode(b).unwrap()).unwrap().pixels
    }
    fn fake_backend(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_pixel(w, h, Rgba([9, 99, 199, 255]))
    }

    #[test]
    fn edit_request_and_restore_1920x1080() {
        let img = photo(1920, 1080);
        let mut r = req("make it night");
        r.resolution = ResolutionMode::Source;
        r.advanced.refine = Some(RefineMode::Off);
        let p = plan(&r, &PlanInputs { image_sizes: vec![img.size()], vram_total_mib: Some(16303), ..Default::default() }, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 7, std::slice::from_ref(&img), None, &BackendOptions::default()).unwrap();
        assert_eq!((j.request.width, j.request.height), (1920, 1088));
        assert_eq!(j.request.ref_images.len(), 1);
        assert!(j.request.init_image.is_none() && j.request.mask_image.is_none());
        let sent = decode(&j.request.ref_images[0]);
        assert_eq!(sent.dimensions(), (1920, 1088));
        // Original content is intact inside the padded reference.
        assert_eq!(imaging::crop(&sent, Rect::new(0, 4, 1920, 1080)), img.pixels);
        let out = postprocess(fake_backend(1920, 1088), &j.post).unwrap();
        assert_eq!(out.dimensions(), (1920, 1080));
    }

    #[test]
    fn multi_reference_order_preserved_in_request() {
        let imgs = vec![photo(640, 960), photo(800, 800), photo(1200, 600)];
        let r = req("person from image 1, clothes from image 2, light from image 3");
        let p = plan(&r, &PlanInputs { image_sizes: imgs.iter().map(|i| i.size()).collect(), vram_total_mib: Some(16303), ..Default::default() }, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 1, &imgs, None, &BackendOptions::default()).unwrap();
        assert_eq!(j.request.ref_images.len(), 3);
        let dims: Vec<(u32, u32)> = j.request.ref_images.iter().map(|b| decode(b).dimensions()).collect();
        assert!(dims[0].1 > dims[0].0, "first is portrait");
        assert_eq!(dims[1].0, dims[1].1, "second is square");
        assert!(dims[2].0 > dims[2].1, "third is landscape");
        for (w, h) in dims {
            assert!(w % 32 == 0 && h % 32 == 0);
            assert!((w as u64 * h as u64) <= (j.request.width as u64 * j.request.height as u64));
        }
    }

    #[test]
    fn inpaint_preserves_unmasked_pixels_exactly() {
        let img = photo(900, 700);
        let mut m = GrayImage::new(900, 700);
        for y in 300..360 {
            for x in 400..480 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        let mut r = req("remove it");
        r.mask = Some("input:m".into());
        let inputs = PlanInputs {
            image_sizes: vec![img.size()],
            mask: Some((Size::new(900, 700), imaging::mask_bbox(&m, 127), imaging::mask_coverage(&m, 127))),
            vram_total_mib: Some(16303),
            ..Default::default()
        };
        let p = plan(&r, &inputs, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 1, std::slice::from_ref(&img), Some(&m), &BackendOptions::default()).unwrap();
        assert!(j.request.init_image.is_some() && j.request.mask_image.is_some());
        assert_eq!(j.request.strength, Some(1.0));
        let bm = imaging::decode_image(&b64_decode(j.request.mask_image.as_ref().unwrap()).unwrap()).unwrap().pixels;
        assert_eq!(bm.dimensions(), (j.request.width, j.request.height));
        assert!(bm.pixels().all(|p| p[0] == 0 || p[0] == 255), "backend mask is binary");
        let out = postprocess(fake_backend(p.working.width, p.working.height), &j.post).unwrap();
        assert_eq!(out.dimensions(), (900, 700));
        let PostKind::Inpaint { blend, region, .. } = &j.post.kind else { panic!() };
        let mut changed_inside = false;
        for (x, y, px) in out.enumerate_pixels() {
            let inside = x >= region.x && x < region.right() && y >= region.y && y < region.bottom();
            let b = if inside { blend.get_pixel(x - region.x, y - region.y)[0] } else { 0 };
            if b == 0 {
                assert_eq!(px, img.pixels.get_pixel(x, y), "pixel {x},{y} changed outside mask");
            } else if b == 255 {
                changed_inside |= px != img.pixels.get_pixel(x, y);
            }
        }
        assert!(changed_inside);
    }

    #[test]
    fn outpaint_preserves_original_region() {
        let img = photo(800, 600);
        let mut r = req("more beach");
        r.outpaint = Some(OutpaintSpec { left: 0.0, right: 50.0, top: 0.0, bottom: 0.0, units: ExpansionUnits::Percent });
        let p = plan(&r, &PlanInputs { image_sizes: vec![img.size()], vram_total_mib: Some(16303), ..Default::default() }, &RecommendedSettings::default()).unwrap();
        assert_eq!(p.output, Size::new(1200, 600));
        let j = prepare(&p, &r.prompt, 1, std::slice::from_ref(&img), None, &BackendOptions::default()).unwrap();
        assert!(j.request.prompt.contains("more beach"));
        let mask = imaging::decode_image(&b64_decode(j.request.mask_image.as_ref().unwrap()).unwrap()).unwrap().pixels;
        let op = p.outpaint.unwrap();
        let r0 = op.source_rect_working;
        assert_eq!(mask.get_pixel(r0.x + 5, r0.y + 5)[0], 0, "original area kept");
        assert_eq!(mask.get_pixel(op.working.canvas.width - 2, 10)[0], 255, "new area generated");
        let out = postprocess(fake_backend(p.working.width, p.working.height), &j.post).unwrap();
        assert_eq!(out.dimensions(), (1200, 600));
        // Everything left of the feather band equals the source exactly.
        let PostKind::Outpaint { feather, .. } = j.post.kind else { panic!() };
        for y in 0..600 {
            for x in 0..(800 - feather) {
                assert_eq!(out.get_pixel(x, y), img.pixels.get_pixel(x, y));
            }
        }
        assert_eq!(out.get_pixel(1150, 300), &Rgba([9, 99, 199, 255]));
    }

    #[test]
    fn alpha_noise_is_flattened_unless_transparency_intended() {
        let r = req("a red fox");
        let p = plan(&r, &PlanInputs { vram_total_mib: Some(16303), ..Default::default() }, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 5, &[], None, &BackendOptions::default()).unwrap();
        let mut noisy = fake_backend(1024, 1024);
        noisy.put_pixel(3, 3, Rgba([1, 2, 3, 250]));
        let out = postprocess(noisy.clone(), &j.post).unwrap();
        assert!(out.pixels().all(|p| p[3] == 255));
        let r = req("a red fox on a transparent background");
        let j = prepare(&p, &r.prompt, 5, &[], None, &BackendOptions::default()).unwrap();
        assert!(j.request.prompt.starts_with("This is an RGBA image with transparency. a red fox on a transparent background."));
        assert!(j.request.prompt.ends_with("the background is transparent."));
        let out = postprocess(noisy, &j.post).unwrap();
        assert_eq!(out.get_pixel(3, 3)[3], 250);
    }

    #[test]
    fn generate_request_has_no_images() {
        let r = req("a red fox");
        let p = plan(&r, &PlanInputs { vram_total_mib: Some(16303), ..Default::default() }, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 5, &[], None, &BackendOptions::default()).unwrap();
        let v = serde_json::to_value(&j.request).unwrap();
        assert!(v.get("ref_images").is_none() && v.get("init_image").is_none());
        assert_eq!(v["seed"], 5);
        assert_eq!(v["width"], 1024);
        let out = postprocess(fake_backend(1024, 1024), &j.post).unwrap();
        assert_eq!(out.dimensions(), (1024, 1024));
    }

    #[test]
    fn transparent_source_alpha_survives_inpaint() {
        let mut img = photo(400, 400);
        for y in 0..50 {
            for x in 0..50 {
                img.pixels.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
        img.has_transparency = true;
        let mut m = GrayImage::new(400, 400);
        for y in 200..260 {
            for x in 200..260 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        let mut r = req("add a star");
        r.mask = Some("m".into());
        let inputs = PlanInputs { image_sizes: vec![img.size()], mask: Some((Size::new(400, 400), imaging::mask_bbox(&m, 127), 0.02)), vram_total_mib: Some(16303), ..Default::default() };
        let p = plan(&r, &inputs, &RecommendedSettings::default()).unwrap();
        let j = prepare(&p, &r.prompt, 1, std::slice::from_ref(&img), Some(&m), &BackendOptions::default()).unwrap();
        let out = postprocess(fake_backend(p.working.width, p.working.height), &j.post).unwrap();
        assert_eq!(out.get_pixel(10, 10)[3], 0, "transparent corner untouched");
    }
}
