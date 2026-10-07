//! Pure dimension math: alignment, aspect ratios, working canvases, outpaint and
//! inpaint geometry. Nothing here touches pixels, so everything is unit-testable.

use serde::{Deserialize, Serialize};

/// Qwen-Image 2.1 requires every canvas and reference dimension to be a multiple of 32.
pub const MODEL_ALIGN: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
    pub fn aspect(&self) -> f64 {
        self.width as f64 / self.height.max(1) as f64
    }
    pub fn is_aligned(&self, align: u32) -> bool {
        self.width % align == 0 && self.height % align == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }
    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }
    pub fn right(&self) -> u32 {
        self.x + self.width
    }
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

pub fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}

pub fn align_down(v: u32, align: u32) -> u32 {
    ((v / align) * align).max(align)
}

/// Nearest multiple of `align`, never below `align`.
pub fn align_nearest(v: u32, align: u32) -> u32 {
    let n = ((v as f64) / align as f64).round() as u32;
    (n * align).max(align)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Human-friendly aspect label: exact reduced ratio when small ("16:9"), otherwise the
/// closest common ratio within 1% ("≈ 4:3"), otherwise a decimal ("1.85:1").
pub fn aspect_label(size: Size) -> String {
    if size.width == 0 || size.height == 0 {
        return "0:0".into();
    }
    let g = gcd(size.width, size.height);
    let (w, h) = (size.width / g, size.height / g);
    if w <= 32 && h <= 32 {
        return format!("{w}:{h}");
    }
    let a = size.aspect();
    for (cw, ch) in COMMON_RATIOS {
        let r = *cw as f64 / *ch as f64;
        if (a / r - 1.0).abs() < 0.01 {
            return format!("≈{cw}:{ch}");
        }
    }
    if a >= 1.0 { format!("{:.2}:1", a) } else { format!("1:{:.2}", 1.0 / a) }
}

const COMMON_RATIOS: &[(u32, u32)] = &[
    (1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16), (5, 4), (4, 5), (21, 9), (9, 21), (2, 1), (1, 2),
];

/// Aligned dimensions closest to `aspect` with area close to `area`, clamped to `max_side`.
pub fn dims_for_area(aspect: f64, area: f64, align: u32, min_side: u32, max_side: u32) -> Size {
    let aspect = aspect.clamp(1.0 / 8.0, 8.0);
    let mut w = (area * aspect).sqrt();
    let mut h = w / aspect;
    let max_side_f = max_side as f64;
    if w > max_side_f {
        w = max_side_f;
        h = w / aspect;
    }
    if h > max_side_f {
        h = max_side_f;
        w = h * aspect;
    }
    let w = align_nearest(w.round() as u32, align).clamp(align_up(min_side, align), align_down(max_side, align));
    let h = align_nearest(h.round() as u32, align).clamp(align_up(min_side, align), align_down(max_side, align));
    Size::new(w, h)
}

/// How a source region is mapped onto the model's working canvas.
///
/// The source is uniformly scaled (never stretched) to `content`, then placed at
/// `content_offset` inside the aligned `canvas`. The remaining border (< align on each
/// axis) is padding that is cropped away after generation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkingCanvas {
    pub source: Size,
    pub content: Size,
    pub canvas: Size,
    pub content_offset: (u32, u32),
    pub scale: f64,
}

impl WorkingCanvas {
    pub fn content_rect(&self) -> Rect {
        Rect::new(self.content_offset.0, self.content_offset.1, self.content.width, self.content.height)
    }
    pub fn is_identity(&self) -> bool {
        self.source == self.canvas && self.content == self.source
    }
    pub fn padding(&self) -> (u32, u32) {
        (self.canvas.width - self.content.width, self.canvas.height - self.content.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasLimits {
    /// Maximum canvas area in pixels.
    pub max_area: u64,
    /// When the source is smaller than this area it is scaled *up* towards it (used for
    /// inpaint crops, where more pixels on the masked detail improves quality).
    pub upscale_to_area: Option<u64>,
    pub max_upscale: f64,
    pub min_side: u32,
    pub max_side: u32,
    pub align: u32,
}

impl Default for CanvasLimits {
    fn default() -> Self {
        Self { max_area: 1024 * 1024, upscale_to_area: None, max_upscale: 1.0, min_side: 256, max_side: 4096, align: MODEL_ALIGN }
    }
}

/// Robust working-resolution calculation. Preserves aspect ratio exactly up to integer
/// rounding of the content size; alignment is achieved with minimal centered padding.
pub fn plan_working_canvas(source: Size, limits: &CanvasLimits) -> WorkingCanvas {
    assert!(source.width > 0 && source.height > 0, "empty source");
    let area = source.area() as f64;
    let mut scale = 1.0f64;
    // Sources within 8% of the budget keep their native size: shrinking by a few
    // percent costs detail and saves almost nothing.
    if area > limits.max_area as f64 * 1.08 {
        scale = (limits.max_area as f64 / area).sqrt();
    } else if let Some(target) = limits.upscale_to_area
        && area < target as f64
    {
        let target = (target.min(limits.max_area)) as f64;
        scale = (target / area).sqrt().min(limits.max_upscale.max(1.0));
    }
    // Fit max_side after alignment padding.
    let max_content = (limits.max_side - limits.align + 1) as f64;
    let longest = source.width.max(source.height) as f64;
    if longest * scale > max_content {
        scale = max_content / longest;
    }
    // Small images: the shortest side must reach min_side once aligned.
    let shortest = source.width.min(source.height) as f64;
    if shortest * scale < limits.min_side as f64 && longest * (limits.min_side as f64 / shortest) <= max_content {
        scale = limits.min_side as f64 / shortest;
    }
    let cw = ((source.width as f64 * scale).round() as u32).max(1);
    let ch = ((source.height as f64 * scale).round() as u32).max(1);
    let canvas = Size::new(
        align_up(cw, limits.align).max(align_up(limits.min_side.min(cw), limits.align)),
        align_up(ch, limits.align).max(align_up(limits.min_side.min(ch), limits.align)),
    );
    let offset = ((canvas.width - cw) / 2, (canvas.height - ch) / 2);
    WorkingCanvas { source, content: Size::new(cw, ch), canvas, content_offset: offset, scale }
}

/// Scale `src` to fit entirely within `bounds` (letterbox). Returns scaled size and the
/// centered offset.
pub fn contain(src: Size, bounds: Size) -> (Size, (u32, u32)) {
    let s = (bounds.width as f64 / src.width as f64).min(bounds.height as f64 / src.height as f64);
    let w = ((src.width as f64 * s).round() as u32).clamp(1, bounds.width);
    let h = ((src.height as f64 * s).round() as u32).clamp(1, bounds.height);
    (Size::new(w, h), ((bounds.width - w) / 2, (bounds.height - h) / 2))
}

/// Scale `src` to fully cover `bounds`. Returns the scaled size and the centered crop
/// rectangle (in scaled coordinates) that equals `bounds`.
pub fn cover(src: Size, bounds: Size) -> (Size, Rect) {
    let s = (bounds.width as f64 / src.width as f64).max(bounds.height as f64 / src.height as f64);
    let w = ((src.width as f64 * s).round() as u32).max(bounds.width);
    let h = ((src.height as f64 * s).round() as u32).max(bounds.height);
    (Size::new(w, h), Rect::new((w - bounds.width) / 2, (h - bounds.height) / 2, bounds.width, bounds.height))
}

// ---------------------------------------------------------------------------------------
// Outpainting
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Expansion {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

impl Expansion {
    pub fn is_zero(&self) -> bool {
        self.left == 0 && self.right == 0 && self.top == 0 && self.bottom == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExpansionUnits {
    #[default]
    Pixels,
    Percent,
}

/// Converts user-facing expansion values to source pixels. Percentages are relative to
/// the source width (left/right) or height (top/bottom).
pub fn expansion_to_pixels(source: Size, left: f64, right: f64, top: f64, bottom: f64, units: ExpansionUnits) -> Expansion {
    let conv = |v: f64, base: u32| -> u32 {
        let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
        match units {
            ExpansionUnits::Pixels => v.round() as u32,
            ExpansionUnits::Percent => (base as f64 * v / 100.0).round() as u32,
        }
    };
    Expansion {
        left: conv(left, source.width),
        right: conv(right, source.width),
        top: conv(top, source.height),
        bottom: conv(bottom, source.height),
    }
}

/// Expansion needed to reach a target aspect ratio by growing only the short dimension,
/// split evenly on both sides. Returns zero expansion if already at (or beyond) target.
pub fn expansion_for_aspect(source: Size, target_aspect: f64) -> Expansion {
    let a = source.aspect();
    if (a / target_aspect - 1.0).abs() < 0.005 {
        return Expansion::default();
    }
    if target_aspect > a {
        let new_w = (source.height as f64 * target_aspect).round() as u32;
        let extra = new_w.saturating_sub(source.width);
        Expansion { left: extra / 2, right: extra - extra / 2, top: 0, bottom: 0 }
    } else {
        let new_h = (source.width as f64 / target_aspect).round() as u32;
        let extra = new_h.saturating_sub(source.height);
        Expansion { left: 0, right: 0, top: extra / 2, bottom: extra - extra / 2 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutpaintPlan {
    pub source: Size,
    pub expansion: Expansion,
    /// Final output size at source pixel density.
    pub final_size: Size,
    /// Where the original image sits in the final output.
    pub source_rect_final: Rect,
    /// Working canvas for the whole final image.
    pub working: WorkingCanvas,
    /// Where the (scaled) original sits inside the working canvas.
    pub source_rect_working: Rect,
    /// Pixels (working space) on each expanded edge where the original is also regenerated
    /// so seams blend. Zero on edges that were not expanded.
    pub overlap_working: Expansion,
}

pub const MAX_FINAL_SIDE: u32 = 16384;

pub fn plan_outpaint(source: Size, expansion: Expansion, limits: &CanvasLimits, overlap_px: u32) -> Result<OutpaintPlan, String> {
    if expansion.is_zero() {
        return Err("Choose at least one side to expand.".into());
    }
    let final_size = Size::new(source.width + expansion.left + expansion.right, source.height + expansion.top + expansion.bottom);
    if final_size.width > MAX_FINAL_SIDE || final_size.height > MAX_FINAL_SIDE {
        return Err(format!("The expanded image would be {}×{}, larger than the {} px limit.", final_size.width, final_size.height, MAX_FINAL_SIDE));
    }
    let working = plan_working_canvas(final_size, limits);
    let s = working.scale;
    let ox = working.content_offset.0;
    let oy = working.content_offset.1;
    // Map source rect into working space, rounding edges (not sizes) so adjacent regions tile.
    let x0 = (expansion.left as f64 * s).round() as u32;
    let y0 = (expansion.top as f64 * s).round() as u32;
    let x1 = (((expansion.left + source.width) as f64) * s).round() as u32;
    let y1 = (((expansion.top + source.height) as f64) * s).round() as u32;
    let x1 = x1.min(working.content.width).max(x0 + 1);
    let y1 = y1.min(working.content.height).max(y0 + 1);
    let source_rect_working = Rect::new(ox + x0, oy + y0, x1 - x0, y1 - y0);
    let ov = |expanded: u32, room: u32| if expanded > 0 { overlap_px.min(room / 4) } else { 0 };
    let overlap_working = Expansion {
        left: ov(expansion.left, source_rect_working.width),
        right: ov(expansion.right, source_rect_working.width),
        top: ov(expansion.top, source_rect_working.height),
        bottom: ov(expansion.bottom, source_rect_working.height),
    };
    Ok(OutpaintPlan {
        source,
        expansion,
        final_size,
        source_rect_final: Rect::new(expansion.left, expansion.top, source.width, source.height),
        working,
        source_rect_working,
        overlap_working,
    })
}

// ---------------------------------------------------------------------------------------
// Inpainting
// ---------------------------------------------------------------------------------------

/// Region of the source to send to the model for an inpaint. Small masks on large images
/// get a context crop (higher detail, faster); large masks use the whole image.
pub fn inpaint_region(source: Size, mask_bbox: Rect, min_context: u32, whole_image_threshold: f64) -> Rect {
    let whole = Rect::new(0, 0, source.width, source.height);
    if mask_bbox.width == 0 || mask_bbox.height == 0 {
        return whole;
    }
    let pad = (mask_bbox.width.max(mask_bbox.height) as f64 * 0.5).round() as u32;
    let pad = pad.max(64);
    let mut x0 = mask_bbox.x.saturating_sub(pad);
    let mut y0 = mask_bbox.y.saturating_sub(pad);
    let mut x1 = (mask_bbox.right() + pad).min(source.width);
    let mut y1 = (mask_bbox.bottom() + pad).min(source.height);
    // Grow to at least min_context per side (when the image allows), keeping centered.
    let grow = |a0: &mut u32, a1: &mut u32, limit: u32| {
        let len = *a1 - *a0;
        let want = min_context.min(limit);
        if len < want {
            let extra = want - len;
            let before = (extra / 2).min(*a0);
            *a0 -= before;
            *a1 = (*a1 + (extra - before)).min(limit);
            let len = *a1 - *a0;
            if len < want {
                *a0 = a0.saturating_sub(want - len);
            }
        }
    };
    grow(&mut x0, &mut x1, source.width);
    grow(&mut y0, &mut y1, source.height);
    let region = Rect::new(x0, y0, x1 - x0, y1 - y0);
    if region.area() as f64 >= whole.area() as f64 * whole_image_threshold {
        whole
    } else {
        region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_area: u64) -> CanvasLimits {
        CanvasLimits { max_area, ..Default::default() }
    }

    #[test]
    fn alignment_helpers() {
        assert_eq!(align_up(1080, 32), 1088);
        assert_eq!(align_up(1024, 32), 1024);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_down(1080, 32), 1056);
        assert_eq!(align_down(10, 32), 32);
        assert_eq!(align_nearest(1080, 32), 1088);
        assert_eq!(align_nearest(1071, 32), 1056);
        assert_eq!(align_nearest(1537, 32), 1536);
        assert_eq!(align_nearest(3, 32), 32);
    }

    #[test]
    fn aspect_labels() {
        assert_eq!(aspect_label(Size::new(1920, 1080)), "16:9");
        assert_eq!(aspect_label(Size::new(1080, 1920)), "9:16");
        assert_eq!(aspect_label(Size::new(1024, 1024)), "1:1");
        assert_eq!(aspect_label(Size::new(4032, 3024)), "4:3");
        assert_eq!(aspect_label(Size::new(1537, 1025)), "≈3:2");
        assert_eq!(aspect_label(Size::new(1850, 1000)), "1.85:1");
        assert_eq!(aspect_label(Size::new(1000, 1850)), "1:1.85");
    }

    #[test]
    fn source_1536x1024_is_identity() {
        let w = plan_working_canvas(Size::new(1536, 1024), &limits(4_200_000));
        assert!(w.is_identity());
        assert_eq!(w.canvas, Size::new(1536, 1024));
    }

    #[test]
    fn source_1920x1080_pads_to_1088() {
        let w = plan_working_canvas(Size::new(1920, 1080), &limits(4_200_000));
        assert_eq!(w.content, Size::new(1920, 1080));
        assert_eq!(w.canvas, Size::new(1920, 1088));
        assert_eq!(w.content_offset, (0, 4));
        assert_eq!(w.scale, 1.0);
    }

    #[test]
    fn portrait_1080x1920() {
        let w = plan_working_canvas(Size::new(1080, 1920), &limits(4_200_000));
        assert_eq!(w.canvas, Size::new(1088, 1920));
        assert_eq!(w.content, Size::new(1080, 1920));
    }

    #[test]
    fn square_1024() {
        let w = plan_working_canvas(Size::new(1024, 1024), &limits(1024 * 1024));
        assert!(w.is_identity());
    }

    #[test]
    fn large_photo_4032x3024_scales_preserving_aspect() {
        let w = plan_working_canvas(Size::new(4032, 3024), &limits(2_359_296));
        assert!(w.canvas.area() <= 2_359_296 + 32 * 2000);
        assert!(w.canvas.is_aligned(32));
        let src_aspect = 4032.0 / 3024.0;
        let content_aspect = w.content.width as f64 / w.content.height as f64;
        assert!((content_aspect / src_aspect - 1.0).abs() < 0.002, "aspect drift {content_aspect}");
        assert!(w.padding().0 < 32 && w.padding().1 < 32);
    }

    #[test]
    fn odd_1537x1025() {
        let w = plan_working_canvas(Size::new(1537, 1025), &limits(4_200_000));
        assert_eq!(w.content, Size::new(1537, 1025));
        assert_eq!(w.canvas, Size::new(1568, 1056));
        assert_eq!(w.content_offset, (15, 15));
    }

    #[test]
    fn aspect_always_preserved_property() {
        let sizes = [(1920, 1080), (1080, 1920), (1024, 1024), (4032, 3024), (1537, 1025), (333, 777), (6000, 400), (64, 64), (8000, 8000)];
        for (w, h) in sizes {
            for area in [1_048_576u64, 2_359_296, 4_194_304] {
                let p = plan_working_canvas(Size::new(w, h), &limits(area));
                assert!(p.canvas.is_aligned(32), "{w}x{h}");
                assert!(p.canvas.width <= 4096 && p.canvas.height <= 4096, "{w}x{h} -> {:?}", p.canvas);
                assert!(p.content.width <= p.canvas.width && p.content.height <= p.canvas.height);
                let drift = (p.content.aspect() / Size::new(w, h).aspect() - 1.0).abs();
                // Rounding a side to an integer can only shift aspect by < 1px.
                let tol = 1.0 / p.content.width.min(p.content.height) as f64 + 1e-9;
                assert!(drift <= tol, "{w}x{h} area {area}: drift {drift} > {tol}");
            }
        }
    }

    #[test]
    fn tiny_image_scaled_up_to_min_side() {
        let p = plan_working_canvas(Size::new(100, 50), &limits(1_048_576));
        assert!(p.content.height >= 256);
        assert_eq!(p.content.width, 512);
    }

    #[test]
    fn upscale_for_inpaint_crops() {
        let l = CanvasLimits { max_area: 2_000_000, upscale_to_area: Some(1_048_576), max_upscale: 4.0, ..Default::default() };
        let p = plan_working_canvas(Size::new(300, 200), &l);
        assert!(p.scale > 3.0 && p.scale <= 4.0);
        let l2 = CanvasLimits { max_upscale: 2.0, ..l };
        assert_eq!(plan_working_canvas(Size::new(300, 200), &l2).scale, 2.0);
    }

    #[test]
    fn dims_for_area_matches_official_buckets_roughly() {
        let d = dims_for_area(1.0, 1024.0 * 1024.0, 32, 256, 4096);
        assert_eq!(d, Size::new(1024, 1024));
        let d = dims_for_area(16.0 / 9.0, 1024.0 * 1024.0, 32, 256, 4096);
        assert_eq!(d, Size::new(1376, 768));
        let d = dims_for_area(9.0 / 16.0, 2048.0 * 2048.0, 32, 256, 4096);
        assert_eq!(d, Size::new(1536, 2720));
        assert!(d.is_aligned(32));
    }

    #[test]
    fn contain_and_cover() {
        let (s, off) = contain(Size::new(1920, 1080), Size::new(1024, 1024));
        assert_eq!(s, Size::new(1024, 576));
        assert_eq!(off, (0, 224));
        let (s, crop) = cover(Size::new(1920, 1080), Size::new(1024, 1024));
        assert_eq!(s, Size::new(1820, 1024));
        assert_eq!(crop, Rect::new(398, 0, 1024, 1024));
    }

    #[test]
    fn expansion_units() {
        let e = expansion_to_pixels(Size::new(1000, 500), 25.0, 0.0, 10.0, 0.0, ExpansionUnits::Percent);
        assert_eq!(e, Expansion { left: 250, right: 0, top: 50, bottom: 0 });
        let e = expansion_to_pixels(Size::new(1000, 500), 256.0, 256.0, -5.0, f64::NAN, ExpansionUnits::Pixels);
        assert_eq!(e, Expansion { left: 256, right: 256, top: 0, bottom: 0 });
    }

    #[test]
    fn expansion_targets() {
        let e = expansion_for_aspect(Size::new(1024, 1024), 16.0 / 9.0);
        assert_eq!(e.left + e.right + 1024, 1820);
        assert_eq!((e.top, e.bottom), (0, 0));
        let e = expansion_for_aspect(Size::new(1920, 1080), 1.0);
        assert_eq!(e.top + e.bottom + 1080, 1920);
        let e = expansion_for_aspect(Size::new(1080, 1920), 9.0 / 16.0);
        assert!(e.is_zero());
    }

    #[test]
    fn outpaint_geometry_right_256() {
        let src = Size::new(1024, 1024);
        let p = plan_outpaint(src, Expansion { left: 0, right: 256, top: 0, bottom: 0 }, &limits(4_200_000), 32).unwrap();
        assert_eq!(p.final_size, Size::new(1280, 1024));
        assert_eq!(p.working.canvas, Size::new(1280, 1024));
        assert_eq!(p.source_rect_working, Rect::new(0, 0, 1024, 1024));
        assert_eq!(p.source_rect_final, Rect::new(0, 0, 1024, 1024));
        assert_eq!(p.overlap_working, Expansion { left: 0, right: 32, top: 0, bottom: 0 });
    }

    #[test]
    fn outpaint_geometry_scaled_and_all_sides() {
        let src = Size::new(1920, 1080);
        let e = Expansion { left: 256, right: 256, top: 128, bottom: 128 };
        let p = plan_outpaint(src, e, &limits(1_048_576), 24).unwrap();
        assert_eq!(p.final_size, Size::new(2432, 1336));
        assert!(p.working.canvas.is_aligned(32));
        assert!(p.working.canvas.area() <= 1_048_576 + 32 * 1400);
        let r = p.source_rect_working;
        // Source occupies the same fractional position in working space.
        let fx = (r.x - p.working.content_offset.0) as f64 / p.working.content.width as f64;
        assert!((fx - 256.0 / 2432.0).abs() < 0.002);
        let fw = r.width as f64 / p.working.content.width as f64;
        assert!((fw - 1920.0 / 2432.0).abs() < 0.002);
        assert!(r.right() <= p.working.canvas.width && r.bottom() <= p.working.canvas.height);
    }

    #[test]
    fn outpaint_rejects_zero_and_huge() {
        assert!(plan_outpaint(Size::new(100, 100), Expansion::default(), &limits(1 << 20), 16).is_err());
        assert!(plan_outpaint(Size::new(100, 100), Expansion { left: 20000, ..Default::default() }, &limits(1 << 20), 16).is_err());
    }

    #[test]
    fn inpaint_region_crop_and_whole() {
        let src = Size::new(4032, 3024);
        let r = inpaint_region(src, Rect::new(2000, 1500, 100, 80), 768, 0.6);
        assert!(r.x <= 2000 && r.y <= 1500 && r.right() >= 2100 && r.bottom() >= 1580);
        assert!(r.width >= 768 && r.height >= 768);
        assert!(r.right() <= 4032 && r.bottom() <= 3024);
        // A large mask uses the whole image.
        let r = inpaint_region(Size::new(1024, 1024), Rect::new(100, 100, 800, 800), 768, 0.6);
        assert_eq!(r, Rect::new(0, 0, 1024, 1024));
        // Near the corner, crop stays inside.
        let r = inpaint_region(src, Rect::new(0, 0, 50, 50), 768, 0.6);
        assert_eq!((r.x, r.y), (0, 0));
        assert!(r.width >= 768);
    }
}
