//! Minimal HEIF (ISO/IEC 23008-12) container reader: just enough to find an HDR gain map.
//!
//! Pixels are decoded elsewhere (Windows' HEIF codec for the base image, ffmpeg for the gain
//! map); this module only reads item metadata: item types, `auxC` types, references and the
//! ISO 21496-1 `tmap` item that says which item is the gain map and how to apply it.

use std::collections::HashMap;

/// True for HEIF/HEIC/AVIF-family files (`ftyp` with a HEIF brand).
pub fn is_heif(b: &[u8]) -> bool {
    if b.len() < 16 || &b[4..8] != b"ftyp" {
        return false;
    }
    let size = (u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize).clamp(16, b.len().min(256));
    let (compatible, _) = b[16..size].as_chunks::<4>();
    let mut brands = std::iter::once(&b[8..12]).chain(compatible.iter().map(|c| c.as_slice()));
    brands.any(|br| matches!(br, b"heic" | b"heix" | b"heim" | b"heis" | b"hevc" | b"hevx" | b"mif1" | b"msf1" | b"mif2"))
}

/// ISO 21496-1 gain map metadata. Values are in log2 space (as stored), single channel
/// or per RGB channel.
#[derive(Debug, Clone, PartialEq)]
pub struct GainMapMetadata {
    pub gain_map_min: [f64; 3],
    pub gain_map_max: [f64; 3],
    pub gamma: [f64; 3],
    pub base_offset: [f64; 3],
    pub alternate_offset: [f64; 3],
    pub base_hdr_headroom: f64,
    pub alternate_hdr_headroom: f64,
    pub multichannel: bool,
    pub use_base_colour_space: bool,
}

impl GainMapMetadata {
    /// Parses the payload of a `tmap` item (a version byte, then the ISO 21496-1 structure).
    pub fn parse_tmap(d: &[u8]) -> Option<Self> {
        let mut r = Reader { b: d, p: 0 };
        if r.u8()? != 0 {
            return None;
        }
        Self::parse_iso(&d[1..])
    }

    /// Parses the bare ISO 21496-1 metadata (as in a JPEG APP2 segment after its URN).
    pub fn parse_iso(d: &[u8]) -> Option<Self> {
        let mut r = Reader { b: d, p: 0 };
        let min_version = r.u16()?;
        let _writer = r.u16()?;
        if min_version != 0 {
            return None;
        }
        let flags = r.u8()?;
        let multichannel = flags & 0x80 != 0;
        let use_base_colour_space = flags & 0x40 != 0;
        let ch = if multichannel { 3 } else { 1 };
        let mut m = Self {
            gain_map_min: [0.0; 3],
            gain_map_max: [0.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.0; 3],
            alternate_offset: [0.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 0.0,
            multichannel,
            use_base_colour_space,
        };
        let frac = |n: f64, d: u32| if d == 0 { None } else { Some(n / d as f64) };
        if flags & 0x08 != 0 {
            let den = r.u32()?;
            m.base_hdr_headroom = frac(r.u32()? as f64, den)?;
            m.alternate_hdr_headroom = frac(r.u32()? as f64, den)?;
            for c in 0..ch {
                m.gain_map_min[c] = frac(r.i32()? as f64, den)?;
                m.gain_map_max[c] = frac(r.i32()? as f64, den)?;
                m.gamma[c] = frac(r.u32()? as f64, den)?;
                m.base_offset[c] = frac(r.i32()? as f64, den)?;
                m.alternate_offset[c] = frac(r.i32()? as f64, den)?;
            }
        } else {
            m.base_hdr_headroom = frac(r.u32()? as f64, r.u32()?)?;
            m.alternate_hdr_headroom = frac(r.u32()? as f64, r.u32()?)?;
            for c in 0..ch {
                m.gain_map_min[c] = frac(r.i32()? as f64, r.u32()?)?;
                m.gain_map_max[c] = frac(r.i32()? as f64, r.u32()?)?;
                m.gamma[c] = frac(r.u32()? as f64, r.u32()?)?;
                m.base_offset[c] = frac(r.i32()? as f64, r.u32()?)?;
                m.alternate_offset[c] = frac(r.i32()? as f64, r.u32()?)?;
            }
        }
        if !multichannel {
            for a in [&mut m.gain_map_min, &mut m.gain_map_max, &mut m.gamma, &mut m.base_offset, &mut m.alternate_offset] {
                a[1] = a[0];
                a[2] = a[0];
            }
        }
        let sane = (0..ch).all(|c| m.gain_map_max[c] >= m.gain_map_min[c] && m.gamma[c] > 0.0)
            && m.alternate_hdr_headroom.is_finite()
            && m.alternate_hdr_headroom != m.base_hdr_headroom;
        sane.then_some(m)
    }

    /// Serializes as ISO 21496-1 (no leading version byte), fractions over 1,000,000.
    pub fn to_iso(&self) -> Vec<u8> {
        const DEN: f64 = 1_000_000.0;
        let mut o = Vec::with_capacity(64);
        o.extend_from_slice(&0u16.to_be_bytes()); // minimum_version
        o.extend_from_slice(&0u16.to_be_bytes()); // writer_version
        let mut flags = 0x08u8; // common denominator
        if self.multichannel {
            flags |= 0x80;
        }
        if self.use_base_colour_space {
            flags |= 0x40;
        }
        o.push(flags);
        let u = |v: f64| ((v * DEN).round().clamp(0.0, u32::MAX as f64) as u32).to_be_bytes();
        let s = |v: f64| ((v * DEN).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32).to_be_bytes();
        o.extend_from_slice(&(DEN as u32).to_be_bytes());
        o.extend_from_slice(&u(self.base_hdr_headroom));
        o.extend_from_slice(&u(self.alternate_hdr_headroom));
        for c in 0..if self.multichannel { 3 } else { 1 } {
            o.extend_from_slice(&s(self.gain_map_min[c]));
            o.extend_from_slice(&s(self.gain_map_max[c]));
            o.extend_from_slice(&u(self.gamma[c]));
            o.extend_from_slice(&s(self.base_offset[c]));
            o.extend_from_slice(&s(self.alternate_offset[c]));
        }
        o
    }
}

/// What a HEIF file says about its HDR gain map.
#[derive(Debug, Clone)]
pub struct HeifGainMap {
    /// Item id of the gain map (ffmpeg uses it as the stream / stream-group id).
    pub item_id: u32,
    /// True when the gain map is a `grid` of tiles (ffmpeg exposes it as a stream group).
    pub is_grid: bool,
    pub metadata: GainMapMetadata,
}

#[derive(Debug, Default, Clone)]
struct Item {
    typ: [u8; 4],
    aux: Option<String>,
    icc: Option<(usize, usize)>,
    refs: Vec<([u8; 4], Vec<u32>)>,
    /// (construction_method, extents as (offset, length)).
    loc: Option<(u8, Vec<(u64, u64)>)>,
}

/// Finds the gain map of a HEIF file through its ISO 21496-1 `tmap` item. Files with only
/// Apple's older proprietary gain map (no `tmap`) return None: their headroom lives in Apple
/// maker notes, which we don't interpret.
pub fn gain_map(b: &[u8]) -> Option<HeifGainMap> {
    let (items, primary, idat) = parse(b)?;
    for it in items.values().filter(|i| &i.typ == b"tmap") {
        let dimg = it.refs.iter().find(|(t, _)| t == b"dimg").map(|(_, to)| to)?;
        if dimg.len() != 2 || dimg[0] != primary {
            continue;
        }
        let (cm, ext) = it.loc.as_ref()?;
        let data = read_extents(b, idat, *cm, ext)?;
        let metadata = GainMapMetadata::parse_tmap(&data)?;
        let gm = items.get(&dimg[1])?;
        return Some(HeifGainMap { item_id: dimg[1], is_grid: &gm.typ == b"grid", metadata });
    }
    None
}

/// The ICC profile of the primary image (`colr` box of type `prof`), if any. Windows' HEIF
/// codec doesn't expose it, and iPhone photos are Display P3.
pub fn primary_icc(b: &[u8]) -> Option<Vec<u8>> {
    let (items, primary, _) = parse(b)?;
    let it = items.get(&primary)?;
    // A grid usually carries the profile itself; otherwise take its first tile's.
    let range = it.icc.or_else(|| {
        let tiles = it.refs.iter().find(|(t, _)| t == b"dimg").map(|(_, to)| to)?;
        items.get(tiles.first()?)?.icc
    })?;
    Some(b[range.0..range.1].to_vec())
}

/// Item type and `auxC` type of every item, for diagnostics and tests.
pub fn describe(b: &[u8]) -> Vec<(u32, String, Option<String>)> {
    let Some((items, _, _)) = parse(b) else { return vec![] };
    let mut v: Vec<_> = items.into_iter().map(|(id, i)| (id, String::from_utf8_lossy(&i.typ).into_owned(), i.aux)).collect();
    v.sort_by_key(|x| x.0);
    v
}

fn read_extents(b: &[u8], idat: Option<(usize, usize)>, cm: u8, ext: &[(u64, u64)]) -> Option<Vec<u8>> {
    let base = match cm {
        0 => (0usize, b.len()),
        1 => idat?,
        _ => return None,
    };
    let mut out = Vec::new();
    for &(off, len) in ext {
        let s = base.0.checked_add(off as usize)?;
        let e = s.checked_add(len as usize)?;
        if e > base.1 || len > 1 << 20 {
            return None;
        }
        out.extend_from_slice(&b[s..e]);
    }
    Some(out)
}

type Parsed = (HashMap<u32, Item>, u32, Option<(usize, usize)>);

fn parse(b: &[u8]) -> Option<Parsed> {
    if !is_heif(b) {
        return None;
    }
    let (ms, me) = boxes(b, 0, b.len()).find(|x| &x.0 == b"meta").map(|x| (x.1 + 4, x.2))?;
    let meta: HashMap<[u8; 4], (usize, usize)> = boxes(b, ms, me).map(|(t, s, e)| (t, (s, e))).collect();
    let mut items: HashMap<u32, Item> = HashMap::new();

    // iinf: item ids and types (infe version 2/3).
    let (s, e) = *meta.get(b"iinf")?;
    let mut r = Reader { b: &b[..e], p: s };
    let ver = r.u8()?;
    r.skip(3)?;
    if ver == 0 { r.u16()? as u32 } else { r.u32()? };
    for (t, cs, ce) in boxes(b, r.p, e) {
        if &t != b"infe" {
            continue;
        }
        let mut r = Reader { b: &b[..ce], p: cs };
        let v = r.u8()?;
        r.skip(3)?;
        if v < 2 {
            continue;
        }
        let id = if v == 2 { r.u16()? as u32 } else { r.u32()? };
        r.skip(2)?;
        let typ: [u8; 4] = r.take(4)?.try_into().ok()?;
        items.entry(id).or_default().typ = typ;
    }

    // iloc: where each item's bytes are.
    let (s, e) = *meta.get(b"iloc")?;
    let mut r = Reader { b: &b[..e], p: s };
    let ver = r.u8()?;
    r.skip(3)?;
    let (a, c) = (r.u8()?, r.u8()?);
    let (osz, lsz, bsz) = (a >> 4, a & 15, c >> 4);
    let isz = if ver == 1 || ver == 2 { c & 15 } else { 0 };
    let count = if ver < 2 { r.u16()? as u32 } else { r.u32()? };
    for _ in 0..count.min(1 << 16) {
        let id = if ver < 2 { r.u16()? as u32 } else { r.u32()? };
        let cm = if ver == 1 || ver == 2 { (r.u16()? & 15) as u8 } else { 0 };
        r.skip(2)?;
        let base = r.uint(bsz)?;
        let n = r.u16()?;
        let mut ext = Vec::with_capacity(n as usize);
        for _ in 0..n {
            if isz > 0 {
                r.uint(isz)?;
            }
            let off = r.uint(osz)?;
            let len = r.uint(lsz)?;
            ext.push((base + off, len));
        }
        items.entry(id).or_default().loc = Some((cm, ext));
    }

    // iref: dimg / auxl / cdsc ... references.
    if let Some(&(s, e)) = meta.get(b"iref") {
        let ver = *b.get(s)?;
        for (t, cs, ce) in boxes(b, s + 4, e) {
            let mut r = Reader { b: &b[..ce], p: cs };
            let from = if ver == 0 { r.u16()? as u32 } else { r.u32()? };
            let n = r.u16()?;
            let to = (0..n).map(|_| if ver == 0 { r.u16().map(u32::from) } else { r.u32() }).collect::<Option<Vec<_>>>()?;
            items.entry(from).or_default().refs.push((t, to));
        }
    }

    // iprp/ipco + ipma: the auxC type of each item.
    if let Some(&(s, e)) = meta.get(b"iprp") {
        let iprp: HashMap<[u8; 4], (usize, usize)> = boxes(b, s, e).map(|(t, s, e)| (t, (s, e))).collect();
        if let (Some(&(cs, ce)), Some(&(ms, me))) = (iprp.get(b"ipco"), iprp.get(b"ipma")) {
            let props: Vec<_> = boxes(b, cs, ce).collect();
            let mut r = Reader { b: &b[..me], p: ms };
            let ver = r.u8()?;
            let flags = (r.u8()? as u32) << 16 | r.u16()? as u32;
            let n = r.u32()?;
            for _ in 0..n.min(1 << 16) {
                let id = if ver < 1 { r.u16()? as u32 } else { r.u32()? };
                let k = r.u8()?;
                for _ in 0..k {
                    let idx = if flags & 1 != 0 { (r.u16()? & 0x7fff) as usize } else { (r.u8()? & 0x7f) as usize };
                    if idx == 0 {
                        continue;
                    }
                    if let Some(&(t, ps, pe)) = props.get(idx - 1) {
                        if &t == b"colr" && pe > ps + 132 && &b[ps..ps + 4] == b"prof" {
                            items.entry(id).or_default().icc = Some((ps + 4, pe));
                        }
                        if &t == b"auxC" && pe > ps + 4 {
                            let urn = &b[ps + 4..pe];
                            let urn = urn.split(|&x| x == 0).next().unwrap_or_default();
                            items.entry(id).or_default().aux = Some(String::from_utf8_lossy(urn).into_owned());
                        }
                    }
                }
            }
        }
    }

    let (ps, _) = *meta.get(b"pitm")?;
    let mut r = Reader { b, p: ps };
    let primary = if r.u8()? == 0 {
        r.skip(3)?;
        r.u16()? as u32
    } else {
        r.skip(3)?;
        r.u32()?
    };
    let idat = meta.get(b"idat").copied();
    Some((items, primary, idat))
}

/// Child boxes in `b[start..end]` as (type, body start, box end).
fn boxes(b: &[u8], start: usize, end: usize) -> impl Iterator<Item = ([u8; 4], usize, usize)> + '_ {
    let end = end.min(b.len());
    let mut off = start;
    std::iter::from_fn(move || {
        if off + 8 > end {
            return None;
        }
        let size = u32::from_be_bytes(b[off..off + 4].try_into().ok()?) as u64;
        let typ: [u8; 4] = b[off + 4..off + 8].try_into().ok()?;
        let (size, hdr) = match size {
            0 => ((end - off) as u64, 8),
            1 if off + 16 <= end => (u64::from_be_bytes(b[off + 8..off + 16].try_into().ok()?), 16),
            1 => return None,
            s => (s, 8),
        };
        if size < hdr as u64 || off as u64 + size > end as u64 {
            return None;
        }
        let item = (typ, off + hdr, off + size as usize);
        off += size as usize;
        Some(item)
    })
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_be_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }
    fn uint(&mut self, size: u8) -> Option<u64> {
        match size {
            0 => Some(0),
            4 => self.u32().map(u64::from),
            8 => self.take(8).map(|s| u64::from_be_bytes(s.try_into().unwrap())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `tmap` payload of an iPhone 15 photo (iOS 18), read with an independent parser.
    const IPHONE_TMAP: &str = "00000000004000000000000f4240001c5683000f4240ffff23b4000f4240001c5683000f4240000a23e7000f42400000000a000f42400000000a000f4240";

    #[test]
    fn parses_iphone_tmap() {
        let m = GainMapMetadata::parse_tmap(&hex::decode(IPHONE_TMAP).unwrap()).unwrap();
        assert!(!m.multichannel && m.use_base_colour_space);
        assert_eq!(m.base_hdr_headroom, 0.0);
        assert!((m.alternate_hdr_headroom - 1.857155).abs() < 1e-6);
        assert!((m.gain_map_min[0] + 0.056396).abs() < 1e-6 && (m.gain_map_max[2] - 1.857155).abs() < 1e-6);
        assert!((m.gamma[0] - 0.664551).abs() < 1e-6);
    }

    #[test]
    fn iso_round_trip() {
        let m = GainMapMetadata::parse_tmap(&hex::decode(IPHONE_TMAP).unwrap()).unwrap();
        let back = GainMapMetadata::parse_iso(&m.to_iso()).unwrap();
        assert!((back.alternate_hdr_headroom - m.alternate_hdr_headroom).abs() < 1e-6);
        assert!((back.gain_map_min[0] - m.gain_map_min[0]).abs() < 1e-6 && (back.gamma[1] - m.gamma[1]).abs() < 1e-6);
    }

    #[test]
    fn rejects_garbage() {
        assert!(!is_heif(b"\x89PNG\r\n\x1a\n0000"));
        assert!(gain_map(b"not a heif file at all").is_none());
        assert!(GainMapMetadata::parse_tmap(&[0, 0, 0]).is_none());
        let mut fake = vec![0, 0, 0, 24];
        fake.extend_from_slice(b"ftypheic\0\0\0\0mif1heic");
        assert!(is_heif(&fake));
        assert!(gain_map(&fake).is_none(), "no meta box");
    }

    /// Run with QLS_TEST_HEIC=<iPhone .HEIC with HDR> to check a real file.
    #[test]
    fn real_file_gain_map() {
        let Ok(p) = std::env::var("QLS_TEST_HEIC") else { return };
        let b = std::fs::read(p).unwrap();
        let g = gain_map(&b).expect("gain map");
        assert!(g.metadata.alternate_hdr_headroom > 0.5);
        let icc = primary_icc(&b).expect("iPhone photos carry an ICC profile");
        assert_eq!(&icc[36..40], b"acsp", "valid ICC header");
        let d = describe(&b);
        assert!(d.iter().any(|(id, _, aux)| *id == g.item_id && aux.as_deref().is_some_and(|a| a.contains("hdrgainmap"))), "{d:?}");
    }
}
