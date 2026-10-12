fn main() {
    // Bundled model files are fetched before `tauri dev` / `tauri build` (beforeDevCommand /
    // beforeBuildCommand); a plain `cargo build` needs them fetched first.
    let upscaler = std::path::Path::new("resources").join("4xNomos2_hq_mosr.pth");
    println!("cargo:rerun-if-changed={}", upscaler.display());
    if !upscaler.is_file() {
        panic!("{} is missing: run `npm run bundled` (scripts/release/fetch-bundled.mjs) first", upscaler.display());
    }
    tauri_build::build()
}
