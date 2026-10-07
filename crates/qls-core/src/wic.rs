//! Decoding through Windows Imaging Component, for formats the `image` crate lacks (HEIC).
//! HEIC needs Microsoft's "HEIF Image Extensions" + "HEVC Video Extensions" (Store).

use image::RgbaImage;

/// Pixels (orientation as the codec presents them) and the embedded ICC profile.
pub struct WicImage {
    pub pixels: RgbaImage,
    pub icc_profile: Option<Vec<u8>>,
}

#[cfg(windows)]
pub fn decode(bytes: &[u8]) -> Result<WicImage, String> {
    use windows::Win32::Graphics::Imaging::*;
    use windows::Win32::System::Com::*;
    use windows::core::GUID;

    // COM per call: the caller may be any tokio blocking thread. S_FALSE (already
    // initialised) is fine; RPC_E_CHANGED_MODE means an STA thread, which also works.
    let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let _guard = init.is_ok().then_some(ComGuard);

    let codec = |e: windows::core::Error| {
        format!("Windows could not decode the image ({e}). For HEIC photos install \"HEIF Image Extensions\" and \"HEVC Video Extensions\" from the Microsoft Store.")
    };
    unsafe {
        let factory: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER).map_err(|e| e.to_string())?;
        let stream = factory.CreateStream().map_err(|e| e.to_string())?;
        stream.InitializeFromMemory(bytes).map_err(|e| e.to_string())?;
        let decoder = factory.CreateDecoderFromStream(&stream, std::ptr::null(), WICDecodeMetadataCacheOnDemand).map_err(codec)?;
        let frame = decoder.GetFrame(0).map_err(codec)?;
        let (mut w, mut h) = (0u32, 0u32);
        frame.GetSize(&mut w, &mut h).map_err(codec)?;
        if w == 0 || h == 0 || (w as u64 * h as u64) > crate::imaging::MAX_INPUT_PIXELS {
            return Err(format!("Image too large ({w}×{h})."));
        }

        let converter = factory.CreateFormatConverter().map_err(|e| e.to_string())?;
        let rgba: GUID = GUID_WICPixelFormat32bppRGBA;
        converter.Initialize(&frame, &rgba, WICBitmapDitherTypeNone, None, 0.0, WICBitmapPaletteTypeCustom).map_err(codec)?;
        let mut buf = vec![0u8; w as usize * h as usize * 4];
        converter.CopyPixels(std::ptr::null(), w * 4, &mut buf).map_err(codec)?;
        let pixels = RgbaImage::from_raw(w, h, buf).ok_or("pixel buffer size mismatch")?;

        Ok(WicImage { pixels, icc_profile: icc_profile(&factory, &frame) })
    }
}

#[cfg(windows)]
unsafe fn icc_profile(factory: &windows::Win32::Graphics::Imaging::IWICImagingFactory, frame: &windows::Win32::Graphics::Imaging::IWICBitmapFrameDecode) -> Option<Vec<u8>> {
    use windows::Win32::Graphics::Imaging::*;
    unsafe {
        let mut n = 0u32;
        frame.GetColorContexts(&mut [], &mut n).ok()?;
        if n == 0 {
            return None;
        }
        let ctx: Vec<IWICColorContext> = (0..n).filter_map(|_| factory.CreateColorContext().ok()).collect();
        let mut slots: Vec<Option<IWICColorContext>> = ctx.iter().cloned().map(Some).collect();
        frame.GetColorContexts(&mut slots, &mut n).ok()?;
        for c in ctx.iter().take(n as usize) {
            if c.GetType().ok()? != WICColorContextProfile {
                continue;
            }
            let mut len = 0u32;
            c.GetProfileBytes(&mut [], &mut len).ok()?;
            let mut v = vec![0u8; len as usize];
            c.GetProfileBytes(&mut v, &mut len).ok()?;
            if len > 128 {
                v.truncate(len as usize);
                return Some(v);
            }
        }
        None
    }
}

#[cfg(windows)]
struct ComGuard;

#[cfg(windows)]
impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe { windows::Win32::System::Com::CoUninitialize() }
    }
}

#[cfg(not(windows))]
pub fn decode(_bytes: &[u8]) -> Result<WicImage, String> {
    Err("HEIC images can only be opened on Windows.".into())
}

#[cfg(test)]
mod tests {
    /// Run with QLS_TEST_HEIC=<path to a .HEIC> (needs the Store HEIF/HEVC extensions).
    #[test]
    fn decodes_real_heic() {
        let Ok(p) = std::env::var("QLS_TEST_HEIC") else { return };
        let img = super::decode(&std::fs::read(p).unwrap()).unwrap();
        println!("WIC: {}x{}, icc {:?} bytes", img.pixels.width(), img.pixels.height(), img.icc_profile.as_ref().map(|v| v.len()));
        assert!(img.pixels.width() > 0);
    }
}
