//! Writes the images an inpaint request would send (init/image 1, guide/image 2, mask) and
//! prints the prompt, without running the model.
//! `cargo run -p qls-core --example inpaint_preview -- <image> <mask> <out_dir> <action> [prompt]`

use qls_core::geometry::Size;
use qls_core::planner::{GenerationRequest, InpaintAction, Mode, Quality, plan};
use qls_core::{config::AppConfig, engine, imaging, pipeline, sdcpp};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (img, mask, out) = (imaging::load_image(a[1].as_ref()).unwrap(), imaging::load_image(a[2].as_ref()).unwrap(), std::path::PathBuf::from(&a[3]));
    let action: InpaintAction = serde_json::from_value(serde_json::json!(a[4])).unwrap();
    let prompt = a.get(5).cloned().unwrap_or_default();
    let m = imaging::mask_from_image(&mask);
    let cfg = AppConfig::default();
    let req = GenerationRequest { prompt: prompt.clone(), mode: Mode::Inpaint, mask: Some("m".into()), inpaint_action: action, quality: Quality::Quality, preserve_dimensions: true, ..Default::default() };
    let info = (Size::new(m.width(), m.height()), imaging::mask_bbox(&m, 16), imaging::mask_coverage(&m, 16));
    let p = plan(&req, &engine::plan_inputs(&cfg, vec![img.size()], Some(info), Some(16303)), &cfg.profile().recommended_settings).unwrap();
    let j = pipeline::prepare(&p, &prompt, 1, std::slice::from_ref(&img), Some(&m), &Default::default()).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let save = |name: &str, b64: &str| std::fs::write(out.join(name), sdcpp::b64_decode(b64).unwrap()).unwrap();
    save("init.png", j.request.init_image.as_ref().unwrap());
    save("mask.png", j.request.mask_image.as_ref().unwrap());
    for (i, r) in j.request.ref_images.iter().enumerate() {
        save(&format!("ref{}.png", i + 1), r);
    }
    println!("working {:?} region {:?} refs {:?}\n{}", p.working, p.inpaint_region, p.references.iter().map(|r| r.canvas.canvas).collect::<Vec<_>>(), j.request.prompt);
}
