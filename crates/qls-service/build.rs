// Rebuild when the UI build changes: release builds embed dist/ (remote.rs, rust-embed).
fn main() {
    println!("cargo:rerun-if-changed=../../dist");
}
