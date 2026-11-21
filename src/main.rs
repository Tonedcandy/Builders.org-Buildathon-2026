use std::cmp::Ordering;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};
use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_array::{Date32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use memchr::memchr;
use memmap2::Mmap;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rustc_hash::FxHashMap;
use serde::Deserialize;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const SESSION_GAP_MS: i64 = 1_800_000;
const ASSIST_WINDOW_MS: i64 = 7 * MS_PER_DAY;
const SHARDS: usize = 256;
const CHUNK_SIZE: usize = 2 * 1024 * 1024;
const BLOOM_BITS: usize = 1 << 21; // 256 KB (fits in L2 cache)
const BLOOM_MASK: usize = BLOOM_BITS - 1;

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

enum FileBuf {
    Empty,
    Mapped(Mmap),
}

impl FileBuf {
    fn open(path: &Path) -> Self {
        let f = File::open(path).unwrap();
        let len = f.metadata().unwrap().len();
        if len == 0 {
            FileBuf::Empty
        } else {
            FileBuf::Mapped(unsafe { Mmap::map(&f).unwrap() })
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[u8] {
        match self {
            FileBuf::Empty => &[],
            FileBuf::Mapped(m) => m,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct SliceRef(u64);

impl SliceRef {
    const EMPTY: Self = Self(0);

    #[inline(always)]
    fn new(buf_idx: u16, off: u32, len: u16) -> Self {
        Self(((buf_idx as u64) << 48) | ((len as u64) << 32) | (off as u64))
    }

    #[inline(always)]
    fn as_bytes<'a>(self, bufs: &[&'a [u8]]) -> &'a [u8] {
        let buf_idx = (self.0 >> 48) as usize;
        let len = ((self.0 >> 32) & 0xFFFF) as usize;
        let off = (self.0 & 0xFFFF_FFFF) as usize;
        &bufs[buf_idx][off..off + len]
    }
}

/// Compact 16-byte string key:
/// - len <= 15: stored 100% inline (bytes 0..len, zero-padded, MSB byte = len)
/// - len > 15:  hi = (hash_56 | (255 << 56)), lo = SliceRef(u64)
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct InlineStr {
    hi: u64,
    lo: u64,
}

impl InlineStr {
    const EMPTY: Self = Self { hi: 0, lo: 0 };

    #[inline(always)]
    fn from_bytes(b: &[u8], sref: SliceRef, hash: u64) -> Self {
        let len = b.len();
        if len <= 15 {
            let mut raw = [0u8; 16];
            raw[..len].copy_from_slice(b);
            raw[15] = len as u8;
            let lo = u64::from_le_bytes(raw[0..8].try_into().unwrap());
            let hi = u64::from_le_bytes(raw[8..16].try_into().unwrap());
            Self { hi, lo }
        } else {
            Self {
                hi: (hash & 0x00FF_FFFF_FFFF_FFFF) | (255u64 << 56),
                lo: sref.0,
            }
        }
    }

    #[inline(always)]
    fn is_inline(self) -> bool {
        (self.hi >> 56) != 255
    }

    #[inline(always)]
    fn inline_len(self) -> usize {
        (self.hi >> 56) as usize
    }

    #[inline(always)]
    fn as_inline_bytes(self, scratch: &mut [u8; 16]) -> &[u8] {
        scratch[0..8].copy_from_slice(&self.lo.to_le_bytes());
        scratch[8..16].copy_from_slice(&self.hi.to_le_bytes());
        &scratch[..self.inline_len()]
    }

    #[inline(always)]
    fn as_bytes<'a>(self, scratch: &'a mut [u8; 16], bufs: &[&'a [u8]]) -> &'a [u8] {
        if self.is_inline() {
            self.as_inline_bytes(scratch)
        } else {
            SliceRef(self.lo).as_bytes(bufs)
        }
    }

    #[inline(always)]
    fn cmp_key(self, other: Self, bufs: &[&[u8]]) -> Ordering {
        self.hi.cmp(&other.hi).then_with(|| {
            if self.is_inline() {
                self.lo.cmp(&other.lo)
            } else {
                SliceRef(self.lo)
                    .as_bytes(bufs)
                    .cmp(SliceRef(other.lo).as_bytes(bufs))
            }
        })
    }

    #[inline(always)]
    fn eq_key(self, other: Self, bufs: &[&[u8]]) -> bool {
        if self.hi != other.hi {
            return false;
        }
        if self.is_inline() {
            self.lo == other.lo
        } else {
            SliceRef(self.lo).as_bytes(bufs) == SliceRef(other.lo).as_bytes(bufs)
        }
    }
}

#[inline(always)]
fn fast_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0x517cc1b727220a95 ^ (bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h.rotate_left(5) ^ w).wrapping_mul(0xf1357aea2e62a9c5);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = 0u64;
        for (i, &b) in rem.iter().enumerate() {
            tail |= (b as u64) << (i * 8);
        }
        h = (h.rotate_left(5) ^ tail).wrapping_mul(0xf1357aea2e62a9c5);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

#[inline(always)]
fn shard_of(h: u64) -> usize {
    (h >> 56) as usize
}

#[inline(always)]
fn has_byte_mask(word: u64, byte: u8) -> u64 {
    let x = word ^ (0x0101_0101_0101_0101u64 * (byte as u64));
    let non_zero =
        (x | ((x & 0x7F7F_7F7F_7F7F_7F7F) + 0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
    non_zero ^ 0x8080_8080_8080_8080
}

#[inline(always)]
fn split_2_swar(line: &[u8]) -> Option<((usize, usize), (usize, usize), bool)> {
    let mut comma = usize::MAX;
    let mut has_q = 0u64;
    let mut i = 0;
    let len = line.len();
    while i + 8 <= len {
        let w = u64::from_le_bytes(line[i..i + 8].try_into().unwrap());
        has_q |= has_byte_mask(w, b'"');
        let mut cm = has_byte_mask(w, b',');
        while cm != 0 {
            if comma != usize::MAX {
                return None;
            }
            comma = i + (cm.trailing_zeros() as usize >> 3);
            cm &= cm - 1;
        }
        i += 8;
    }
    while i < len {
        let b = line[i];
        if b == b'"' {
            has_q = 1;
        } else if b == b',' {
            if comma != usize::MAX {
                return None;
            }
            comma = i;
        }
        i += 1;
    }
    if comma == usize::MAX {
        return None;
    }
    Some(((0, comma), (comma + 1, len), has_q != 0))
}

#[inline(always)]
fn split_4_swar(line: &[u8]) -> Option<([usize; 3], bool)> {
    let mut pos = [0usize; 3];
    let mut found = 0usize;
    let mut has_q = 0u64;
    let mut i = 0;
    let len = line.len();
    while i + 8 <= len {
        let w = u64::from_le_bytes(line[i..i + 8].try_into().unwrap());
        has_q |= has_byte_mask(w, b'"');
        let mut cm = has_byte_mask(w, b',');
        while cm != 0 {
            if found == 3 {
                return None;
            }
            pos[found] = i + (cm.trailing_zeros() as usize >> 3);
            found += 1;
            cm &= cm - 1;
        }
        i += 8;
    }
    while i < len {
        let b = line[i];
        if b == b'"' {
            has_q = 1;
        } else if b == b',' {
            if found == 3 {
                return None;
            }
            pos[found] = i;
            found += 1;
        }
        i += 1;
    }
    if found != 3 {
        return None;
    }
    Some((pos, has_q != 0))
}

#[inline(always)]
fn split_7_swar(line: &[u8]) -> Option<([usize; 6], bool)> {
    let mut pos = [0usize; 6];
    let mut found = 0usize;
    let mut has_q = 0u64;
    let mut i = 0;
    let len = line.len();
    while i + 8 <= len {
        let w = u64::from_le_bytes(line[i..i + 8].try_into().unwrap());
        has_q |= has_byte_mask(w, b'"');
        let mut cm = has_byte_mask(w, b',');
        while cm != 0 {
            if found == 6 {
                return None;
            }
            pos[found] = i + (cm.trailing_zeros() as usize >> 3);
            found += 1;
            cm &= cm - 1;
        }
        i += 8;
    }
    while i < len {
        let b = line[i];
        if b == b'"' {
            has_q = 1;
        } else if b == b',' {
            if found == 6 {
                return None;
            }
            pos[found] = i;
            found += 1;
        }
        i += 1;
    }
    if found != 6 {
        return None;
    }
    Some((pos, has_q != 0))
}

#[inline(always)]
fn unquote_range(buf: &[u8], start: usize, end: usize) -> Option<(usize, usize)> {
    let f = &buf[start..end];
    if memchr(b'"', f).is_none() {
        return Some((start, end));
    }
    if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' {
        let inner = &f[1..f.len() - 1];
        if memchr(b'"', inner).is_none() {
            return Some((start + 1, end - 1));
        }
    }
    None
}

#[inline(always)]
fn parse_2_digits(a: u8, b: u8) -> Option<i32> {
    let d0 = a.wrapping_sub(b'0');
    let d1 = b.wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some((d0 as i32) * 10 + (d1 as i32))
}

#[inline(always)]
fn parse_4_digits(b: &[u8]) -> Option<i32> {
    let d0 = b[0].wrapping_sub(b'0');
    let d1 = b[1].wrapping_sub(b'0');
    let d2 = b[2].wrapping_sub(b'0');
    let d3 = b[3].wrapping_sub(b'0');
    if (d0 | d1 | d2 | d3) > 9 {
        // Check each individually since e.g. 8|2 = 10 > 9
        if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
            return None;
        }
    }
    Some((d0 as i32) * 1000 + (d1 as i32) * 100 + (d2 as i32) * 10 + (d3 as i32))
}

#[inline(always)]
fn days_from_civil(y: i32, m: i32, d: i32) -> Option<i64> {
    if !(1..=12).contains(&m) || d < 1 {
        return None;
    }
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let max_d = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap {
                29
            } else {
                28
            }
        }
        _ => return None,
    };
    if d > max_d {
        return None;
    }
    let y = (y as i64) - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + (d as u64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + (doe as i64) - 719_468)
}

struct DateCache {
    keys: [[u8; 10]; 64],
    vals: [i64; 64],
}

impl DateCache {
    fn new() -> Self {
        Self {
            keys: [[b'?'; 10]; 64],
            vals: [0; 64],
        }
    }

    #[inline(always)]
    fn parse_datetime_prefix(&mut self, b: &[u8]) -> Option<i64> {
        if b.len() < 19 || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let slot = (((b[6] as usize) << 5) ^ ((b[8] as usize) << 4) ^ (b[9] as usize)) & 63;
        let day_ms = if &b[0..10] == &self.keys[slot] {
            self.vals[slot]
        } else {
            if b[4] != b'-' || b[7] != b'-' {
                return None;
            }
            let y = parse_4_digits(&b[0..4])?;
            let m = parse_2_digits(b[5], b[6])?;
            let d = parse_2_digits(b[8], b[9])?;
            let days = days_from_civil(y, m, d)?;
            let ms = days * MS_PER_DAY;
            self.keys[slot].copy_from_slice(&b[0..10]);
            self.vals[slot] = ms;
            ms
        };
        let hh = parse_2_digits(b[11], b[12])?;
        let mm = parse_2_digits(b[14], b[15])?;
        let ss = parse_2_digits(b[17], b[18])?;
        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        Some(day_ms + ((hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
    }

    #[inline(always)]
    fn parse_ts(&mut self, b: &[u8]) -> Option<i64> {
        let len = b.len();
        if len == 20 && b[19] == b'Z' {
            return self.parse_datetime_prefix(b);
        }
        if len == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
            let base = self.parse_datetime_prefix(b)?;
            let oh = parse_2_digits(b[20], b[21])?;
            let om = parse_2_digits(b[23], b[24])?;
            if oh > 23 || om > 59 {
                return None;
            }
            let off_ms = ((oh as i64) * 3600 + (om as i64) * 60) * 1000;
            return Some(if b[19] == b'+' {
                base - off_ms
            } else {
                base + off_ms
            });
        }
        if len == 0 {
            return None;
        }
        let mut val: i64 = 0;
        for &c in b {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            val = val.checked_mul(10)?.checked_add(d as i64)?;
        }
        Some(val)
    }
}

#[inline(always)]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut val: u64 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.saturating_mul(10).saturating_add(d as u64);
        if val > 10_000_000 {
            return None;
        }
    }
    if val >= 1 {
        Some(val as u32)
    } else {
        None
    }
}

fn list_csvs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
        .collect();
    files.sort();
    files
}

fn make_chunks(buf_indices: std::ops::Range<usize>, bufs: &[&[u8]]) -> Vec<(u16, usize, usize)> {
    let mut chunks = Vec::new();
    for bi in buf_indices {
        let buf = bufs[bi];
        let Some(first_nl) = memchr(b'\n', buf) else {
            continue;
        };
        let mut pos = first_nl + 1;
        while pos < buf.len() {
            let target = (pos + CHUNK_SIZE).min(buf.len());
            let end = if target == buf.len() {
                buf.len()
            } else {
                match memchr(b'\n', &buf[target..]) {
                    Some(rel) => target + rel + 1,
                    None => buf.len(),
                }
            };
            chunks.push((bi as u16, pos, end));
            pos = end;
        }
    }
    chunks
}

#[inline(always)]
fn for_each_chunk_line(buf: &[u8], start: usize, end: usize, mut f: impl FnMut(usize, usize)) {
    let mut pos = start;
    while pos < end {
        let line_end = match memchr(b'\n', &buf[pos..end]) {
            Some(rel) => pos + rel,
            None => end,
        };
        if line_end > pos {
            f(pos, line_end);
        }
        pos = line_end + 1;
    }
}

struct Dsu<'a> {
    parent: Vec<u32>,
    refs: Vec<SliceRef>,
    id_to_idx: FxHashMap<&'a [u8], u32>,
}

impl<'a> Dsu<'a> {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            refs: Vec::new(),
            id_to_idx: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn get_or_insert(&mut self, s: &'a [u8], sref: SliceRef) -> u32 {
        if let Some(&idx) = self.id_to_idx.get(s) {
            return idx;
        }
        let idx = self.refs.len() as u32;
        self.refs.push(sref);
        self.parent.push(idx);
        self.id_to_idx.insert(s, idx);
        idx
    }

    #[inline(always)]
    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[x as usize] != root {
            let nxt = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = nxt;
        }
        root
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32, bufs: &[&'a [u8]]) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            if self.refs[ra as usize].as_bytes(bufs) < self.refs[rb as usize].as_bytes(bufs) {
                self.parent[rb as usize] = ra;
            } else {
                self.parent[ra as usize] = rb;
            }
        }
    }
}

struct IdLookup<'a> {
    has_links: bool,
    bloom: Vec<u64>,
    map: FxHashMap<&'a [u8], (InlineStr, u64)>,
}

impl<'a> IdLookup<'a> {
    fn build(link_chunks: &[(u16, usize, usize)], bufs: &[&'a [u8]], ncpu: usize) -> Self {
        if link_chunks.is_empty() {
            return Self {
                has_links: false,
                bloom: Vec::new(),
                map: FxHashMap::default(),
            };
        }
        let next = AtomicUsize::new(0);
        let pairs_per_thread: Vec<Vec<(SliceRef, SliceRef)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu.min(link_chunks.len()))
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let ci = next.fetch_add(1, AtomicOrd::Relaxed);
                            if ci >= link_chunks.len() {
                                break;
                            }
                            let (bi, start, end) = link_chunks[ci];
                            let buf = bufs[bi as usize];
                            for_each_chunk_line(buf, start, end, |ls, le| {
                                let line = &buf[ls..le];
                                let Some(((a0, a1), (b0, b1), has_q)) = split_2_swar(line) else {
                                    return;
                                };
                                let (as0, ae0, bs0, be0) = if !has_q {
                                    (ls + a0, ls + a1, ls + b0, ls + b1)
                                } else {
                                    let (Some((as0, ae0)), Some((bs0, be0))) = (
                                        unquote_range(buf, ls + a0, ls + a1),
                                        unquote_range(buf, ls + b0, ls + b1),
                                    ) else {
                                        return;
                                    };
                                    (as0, ae0, bs0, be0)
                                };
                                if ae0 == as0 || be0 == bs0 {
                                    return;
                                }
                                out.push((
                                    SliceRef::new(bi, as0 as u32, (ae0 - as0) as u16),
                                    SliceRef::new(bi, bs0 as u32, (be0 - bs0) as u16),
                                ));
                            });
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut dsu = Dsu::new();
        for vec in &pairs_per_thread {
            for &(ra, rb) in vec {
                let ia = dsu.get_or_insert(ra.as_bytes(bufs), ra);
                let ib = dsu.get_or_insert(rb.as_bytes(bufs), rb);
                dsu.union(ia, ib, bufs);
            }
        }

        let n = dsu.refs.len();
        if n == 0 {
            return Self {
                has_links: false,
                bloom: Vec::new(),
                map: FxHashMap::default(),
            };
        }

        let mut root_cache: Vec<Option<(InlineStr, u64)>> = vec![None; n];
        let mut bloom = vec![0u64; BLOOM_BITS / 64];
        let mut map: FxHashMap<&'a [u8], (InlineStr, u64)> =
            FxHashMap::with_capacity_and_hasher(n, Default::default());

        for i in 0..n {
            let r = dsu.find(i as u32) as usize;
            if r != i {
                let canonical = *root_cache[r].get_or_insert_with(|| {
                    let rr = dsu.refs[r];
                    let rb = rr.as_bytes(bufs);
                    let rh = fast_hash(rb);
                    (InlineStr::from_bytes(rb, rr, rh), rh)
                });
                let kb = dsu.refs[i].as_bytes(bufs);
                let kh = fast_hash(kb);
                let b1 = (kh as usize) & BLOOM_MASK;
                let b2 = ((kh >> 21) as usize) & BLOOM_MASK;
                bloom[b1 >> 6] |= 1u64 << (b1 & 63);
                bloom[b2 >> 6] |= 1u64 << (b2 & 63);
                map.insert(kb, canonical);
            }
        }

        Self {
            has_links: !map.is_empty(),
            bloom,
            map,
        }
    }

    #[inline(always)]
    fn resolve(&self, uid: &[u8], uid_ref: SliceRef, uid_hash: u64) -> (InlineStr, u64) {
        if self.has_links {
            let b1 = (uid_hash as usize) & BLOOM_MASK;
            if (self.bloom[b1 >> 6] & (1u64 << (b1 & 63))) != 0 {
                let b2 = ((uid_hash >> 21) as usize) & BLOOM_MASK;
                if (self.bloom[b2 >> 6] & (1u64 << (b2 & 63))) != 0 {
                    if let Some(&canon) = self.map.get(uid) {
                        return canon;
                    }
                }
            }
        }
        (InlineStr::from_bytes(uid, uid_ref, uid_hash), uid_hash)
    }
}

#[derive(Clone, Copy)]
struct RawEvent {
    eid_hash: u64,
    ingest_ts: i64,
    ts: i64, // i64::MIN means invalid (kept for dedup only)
    eid_ref: SliceRef,
    person: InlineStr,
    person_hash: u64,
    price_cat: u64, // low 32: price_cents (0 = not purchase), high 16: cat_idx
}

#[derive(Clone, Copy)]
struct ValidEvent {
    person: InlineStr,
    ts: i64,
    price_cat: u64,
}

#[derive(Clone, Copy)]
struct RawClick {
    cid_hash: u64,
    cid_ref: SliceRef,
    person: InlineStr,
    person_hash: u64,
    ts: i64,
    campaign_idx: u16,
}

#[derive(Clone, Copy)]
struct ValidClick {
    person: InlineStr,
    ts: i64,
    cid_ref: SliceRef,
    campaign_idx: u16,
}

#[derive(Default, Clone, Copy)]
struct CampaignDayAgg {
    sessions: i64,
    purchasing_sessions: i64,
    revenue_cents: i64,
    assisted_sessions: i64,
}

fn main() {
    let data_dir = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "/output".into()));
    let ncpu: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        })
        .clamp(1, 64);

    let manifest: Manifest =
        serde_json::from_slice(&fs::read(data_dir.join("manifest.json")).unwrap()).unwrap();
    let mut dc0 = DateCache::new();
    let win_start = dc0.parse_ts(manifest.window_start.as_bytes()).unwrap();
    let win_end = dc0.parse_ts(manifest.window_end.as_bytes()).unwrap();
    let click_win_start = win_start - ASSIST_WINDOW_MS;

    let min_day = win_start.div_euclid(MS_PER_DAY) as i32;
    let max_day = (win_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days_i64 = (max_day as i64) - (min_day as i64) + 1;

    // 1. Mmap all files (zero-copy)
    let prod_buf = FileBuf::open(&data_dir.join("products.csv"));
    let camp_buf = FileBuf::open(&data_dir.join("campaigns.csv"));

    let link_paths = list_csvs(&data_dir.join("links"));
    let event_paths = list_csvs(&data_dir.join("events"));
    let click_paths = list_csvs(&data_dir.join("clicks"));

    let mut file_bufs: Vec<FileBuf> =
        Vec::with_capacity(link_paths.len() + event_paths.len() + click_paths.len());
    let link_start_idx = 0;
    for p in &link_paths {
        file_bufs.push(FileBuf::open(p));
    }
    let event_start_idx = file_bufs.len();
    for p in &event_paths {
        file_bufs.push(FileBuf::open(p));
    }
    let click_start_idx = file_bufs.len();
    for p in &click_paths {
        file_bufs.push(FileBuf::open(p));
    }
    let all_bufs: Vec<&[u8]> = file_bufs.iter().map(|b| b.as_slice()).collect();

    // 2. Load products.csv -> category index (0 = "unknown")
    let mut categories: Vec<&str> = vec!["unknown"];
    let mut cat_to_idx: FxHashMap<&[u8], u16> = FxHashMap::default();
    cat_to_idx.insert(b"unknown", 0);
    let mut prod_to_cat: FxHashMap<&[u8], u16> = FxHashMap::default();
    {
        let b = prod_buf.as_slice();
        if let Some(nl) = memchr(b'\n', b) {
            for_each_chunk_line(b, nl + 1, b.len(), |ls, le| {
                let line = &b[ls..le];
                if let Some(((p0, p1), (c0, c1), has_q)) = split_2_swar(line) {
                    let (ps, pe, cs, ce) = if !has_q {
                        (ls + p0, ls + p1, ls + c0, ls + c1)
                    } else {
                        let (Some((ps, pe)), Some((cs, ce))) = (
                            unquote_range(b, ls + p0, ls + p1),
                            unquote_range(b, ls + c0, ls + c1),
                        ) else {
                            return;
                        };
                        (ps, pe, cs, ce)
                    };
                    let p = &b[ps..pe];
                    let c = &b[cs..ce];
                    let idx = *cat_to_idx.entry(c).or_insert_with(|| {
                        let i = categories.len() as u16;
                        categories.push(std::str::from_utf8(c).unwrap());
                        i
                    });
                    prod_to_cat.insert(p, idx);
                }
            });
        }
    }

    // 3. Load campaigns.csv -> campaign index (0 = "organic")
    let mut camp_names: Vec<&str> = vec!["organic"];
    let mut camp_channels: Vec<&str> = vec!["organic"];
    let mut camp_to_idx: FxHashMap<&[u8], u16> = FxHashMap::default();
    {
        let b = camp_buf.as_slice();
        if let Some(nl) = memchr(b'\n', b) {
            for_each_chunk_line(b, nl + 1, b.len(), |ls, le| {
                let line = &b[ls..le];
                if let Some(((c0, c1), (ch0, ch1), has_q)) = split_2_swar(line) {
                    let (cs, ce, chs, che) = if !has_q {
                        (ls + c0, ls + c1, ls + ch0, ls + ch1)
                    } else {
                        let (Some((cs, ce)), Some((chs, che))) = (
                            unquote_range(b, ls + c0, ls + c1),
                            unquote_range(b, ls + ch0, ls + ch1),
                        ) else {
                            return;
                        };
                        (cs, ce, chs, che)
                    };
                    let c = &b[cs..ce];
                    let ch = &b[chs..che];
                    let idx = camp_names.len() as u16;
                    camp_names.push(std::str::from_utf8(c).unwrap());
                    camp_channels.push(std::str::from_utf8(ch).unwrap());
                    camp_to_idx.insert(c, idx);
                }
            });
        }
    }

    // 4. Build chunks and ID lookup
    let link_chunks = make_chunks(link_start_idx..event_start_idx, &all_bufs);
    let event_chunks = make_chunks(event_start_idx..click_start_idx, &all_bufs);
    let click_chunks = make_chunks(click_start_idx..all_bufs.len(), &all_bufs);

    let id_lookup = IdLookup::build(&link_chunks, &all_bufs, ncpu);

    // 5a. Parse events in parallel across 2 MB byte chunks
    let raw_events_per_thread: Vec<Vec<Vec<RawEvent>>> = {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu)
                .map(|_| {
                    s.spawn(|| {
                        let mut shards: Vec<Vec<RawEvent>> =
                            (0..SHARDS).map(|_| Vec::new()).collect();
                        let mut dc = DateCache::new();
                        loop {
                            let ci = next.fetch_add(1, AtomicOrd::Relaxed);
                            if ci >= event_chunks.len() {
                                break;
                            }
                            let (bi, start, end) = event_chunks[ci];
                            let buf = all_bufs[bi as usize];
                            for_each_chunk_line(buf, start, end, |ls, le| {
                                let line = &buf[ls..le];
                                let Some((pos, has_q)) = split_7_swar(line) else {
                                    return;
                                };
                                let (
                                    (s1, e1),
                                    (s2, e2),
                                    (s3, e3),
                                    (s4, e4),
                                    (s5, e5),
                                    (s6, e6),
                                    (s7, e7),
                                ) = if !has_q {
                                    (
                                        (ls, ls + pos[0]),
                                        (ls + pos[0] + 1, ls + pos[1]),
                                        (ls + pos[1] + 1, ls + pos[2]),
                                        (ls + pos[2] + 1, ls + pos[3]),
                                        (ls + pos[3] + 1, ls + pos[4]),
                                        (ls + pos[4] + 1, ls + pos[5]),
                                        (ls + pos[5] + 1, le),
                                    )
                                } else {
                                    let (
                                        Some(r1),
                                        Some(r2),
                                        Some(r3),
                                        Some(r4),
                                        Some(r5),
                                        Some(r6),
                                        Some(r7),
                                    ) = (
                                        unquote_range(buf, ls, ls + pos[0]),
                                        unquote_range(buf, ls + pos[0] + 1, ls + pos[1]),
                                        unquote_range(buf, ls + pos[1] + 1, ls + pos[2]),
                                        unquote_range(buf, ls + pos[2] + 1, ls + pos[3]),
                                        unquote_range(buf, ls + pos[3] + 1, ls + pos[4]),
                                        unquote_range(buf, ls + pos[4] + 1, ls + pos[5]),
                                        unquote_range(buf, ls + pos[5] + 1, le),
                                    )
                                    else {
                                        return;
                                    };
                                    (r1, r2, r3, r4, r5, r6, r7)
                                };

                                if e1 == s1 {
                                    return;
                                }
                                let Some(ingest_ts) = dc.parse_ts(&buf[s4..e4]) else {
                                    return;
                                };
                                let eid = &buf[s1..e1];
                                let eid_hash = fast_hash(eid);
                                let eid_ref = SliceRef::new(bi, s1 as u32, (e1 - s1) as u16);

                                let mut out_ts = i64::MIN;
                                let mut out_person = InlineStr::EMPTY;
                                let mut out_phash = 0u64;
                                let mut out_price_cat = 0u64;

                                if e2 > s2 && e6 > s6 {
                                    let et = &buf[s5..e5];
                                    let is_purchase = if et == b"view" || et == b"cart" {
                                        Some(false)
                                    } else if et == b"purchase" {
                                        Some(true)
                                    } else {
                                        None
                                    };
                                    if let Some(is_p) = is_purchase {
                                        if let Some(ts) = dc.parse_ts(&buf[s3..e3]) {
                                            if ts >= win_start
                                                && ts < win_end
                                                && (ingest_ts - ts) <= MS_PER_DAY
                                            {
                                                if is_p {
                                                    if let Some(price) = parse_price(&buf[s7..e7]) {
                                                        let uid = &buf[s2..e2];
                                                        let uref = SliceRef::new(
                                                            bi,
                                                            s2 as u32,
                                                            (e2 - s2) as u16,
                                                        );
                                                        let uh = fast_hash(uid);
                                                        let (p, ph) =
                                                            id_lookup.resolve(uid, uref, uh);
                                                        let cat_idx = *prod_to_cat
                                                            .get(&buf[s6..e6])
                                                            .unwrap_or(&0);
                                                        out_ts = ts;
                                                        out_person = p;
                                                        out_phash = ph;
                                                        out_price_cat = (price as u64)
                                                            | ((cat_idx as u64) << 32);
                                                    }
                                                } else {
                                                    let uid = &buf[s2..e2];
                                                    let uref = SliceRef::new(
                                                        bi,
                                                        s2 as u32,
                                                        (e2 - s2) as u16,
                                                    );
                                                    let uh = fast_hash(uid);
                                                    let (p, ph) = id_lookup.resolve(uid, uref, uh);
                                                    out_ts = ts;
                                                    out_person = p;
                                                    out_phash = ph;
                                                }
                                            }
                                        }
                                    }
                                }

                                shards[shard_of(eid_hash)].push(RawEvent {
                                    eid_hash,
                                    ingest_ts,
                                    ts: out_ts,
                                    eid_ref,
                                    person: out_person,
                                    person_hash: out_phash,
                                    price_cat: out_price_cat,
                                });
                            });
                        }
                        shards
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    // 5b. Dedup events using flat 8-byte-slot open-addressing table and re-shard by person
    let event_person_shards: Vec<Vec<Vec<ValidEvent>>> = {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu)
                .map(|_| {
                    s.spawn(|| {
                        let mut out: Vec<Vec<ValidEvent>> =
                            (0..SHARDS).map(|_| Vec::new()).collect();
                        let mut table: Vec<u64> = Vec::new();
                        let mut unique: Vec<RawEvent> = Vec::new();
                        loop {
                            let sh = next.fetch_add(1, AtomicOrd::Relaxed);
                            if sh >= SHARDS {
                                break;
                            }
                            let n: usize =
                                raw_events_per_thread.iter().map(|t| t[sh].len()).sum();
                            if n == 0 {
                                continue;
                            }
                            let cap = ((n * 3) / 2 + 16).next_power_of_two().max(64);
                            let mask = cap - 1;
                            table.clear();
                            table.resize(cap, 0);
                            unique.clear();
                            if unique.capacity() < n {
                                unique.reserve(n - unique.capacity());
                            }

                            for t in &raw_events_per_thread {
                                for &ev in &t[sh] {
                                    let tag = (ev.eid_hash >> 24) as u32;
                                    let mut pos = (ev.eid_hash as usize) & mask;
                                    loop {
                                        let entry = table[pos];
                                        if entry == 0 {
                                            let idx = unique.len() as u32;
                                            unique.push(ev);
                                            table[pos] =
                                                ((tag as u64) << 32) | ((idx as u64) + 1);
                                            break;
                                        }
                                        if (entry >> 32) as u32 == tag {
                                            let idx = ((entry & 0xFFFF_FFFF) - 1) as usize;
                                            let existing = &mut unique[idx];
                                            if existing.eid_hash == ev.eid_hash
                                                && existing.eid_ref.as_bytes(&all_bufs)
                                                    == ev.eid_ref.as_bytes(&all_bufs)
                                            {
                                                if ev.ingest_ts < existing.ingest_ts {
                                                    *existing = ev;
                                                }
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }

                            for ev in &unique {
                                if ev.ts != i64::MIN {
                                    out[shard_of(ev.person_hash)].push(ValidEvent {
                                        person: ev.person,
                                        ts: ev.ts,
                                        price_cat: ev.price_cat,
                                    });
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    drop(raw_events_per_thread);

    // 6a. Parse clicks in parallel across 2 MB byte chunks
    let raw_clicks_per_thread: Vec<Vec<Vec<RawClick>>> = {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu)
                .map(|_| {
                    s.spawn(|| {
                        let mut shards: Vec<Vec<RawClick>> =
                            (0..SHARDS).map(|_| Vec::new()).collect();
                        let mut dc = DateCache::new();
                        loop {
                            let ci = next.fetch_add(1, AtomicOrd::Relaxed);
                            if ci >= click_chunks.len() {
                                break;
                            }
                            let (bi, start, end) = click_chunks[ci];
                            let buf = all_bufs[bi as usize];
                            for_each_chunk_line(buf, start, end, |ls, le| {
                                let line = &buf[ls..le];
                                let Some((pos, has_q)) = split_4_swar(line) else {
                                    return;
                                };
                                let ((s1, e1), (s2, e2), (s3, e3), (s4, e4)) = if !has_q {
                                    (
                                        (ls, ls + pos[0]),
                                        (ls + pos[0] + 1, ls + pos[1]),
                                        (ls + pos[1] + 1, ls + pos[2]),
                                        (ls + pos[2] + 1, le),
                                    )
                                } else {
                                    let (Some(r1), Some(r2), Some(r3), Some(r4)) = (
                                        unquote_range(buf, ls, ls + pos[0]),
                                        unquote_range(buf, ls + pos[0] + 1, ls + pos[1]),
                                        unquote_range(buf, ls + pos[1] + 1, ls + pos[2]),
                                        unquote_range(buf, ls + pos[2] + 1, le),
                                    ) else {
                                        return;
                                    };
                                    (r1, r2, r3, r4)
                                };
                                if e1 == s1 || e2 == s2 {
                                    return;
                                }
                                let Some(ts) = dc.parse_ts(&buf[s4..e4]) else {
                                    return;
                                };
                                if ts < click_win_start || ts >= win_end {
                                    return;
                                }
                                let Some(&campaign_idx) = camp_to_idx.get(&buf[s3..e3]) else {
                                    return;
                                };
                                let cid = &buf[s1..e1];
                                let cid_hash = fast_hash(cid);
                                let cid_ref = SliceRef::new(bi, s1 as u32, (e1 - s1) as u16);
                                let uid = &buf[s2..e2];
                                let uref = SliceRef::new(bi, s2 as u32, (e2 - s2) as u16);
                                let uh = fast_hash(uid);
                                let (person, person_hash) = id_lookup.resolve(uid, uref, uh);

                                shards[shard_of(cid_hash)].push(RawClick {
                                    cid_hash,
                                    cid_ref,
                                    person,
                                    person_hash,
                                    ts,
                                    campaign_idx,
                                });
                            });
                        }
                        shards
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    // 6b. Dedup clicks using flat 8-byte-slot open-addressing table and re-shard by person
    let click_person_shards: Vec<Vec<Vec<ValidClick>>> = {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu)
                .map(|_| {
                    s.spawn(|| {
                        let mut out: Vec<Vec<ValidClick>> =
                            (0..SHARDS).map(|_| Vec::new()).collect();
                        let mut table: Vec<u64> = Vec::new();
                        let mut unique: Vec<RawClick> = Vec::new();
                        loop {
                            let sh = next.fetch_add(1, AtomicOrd::Relaxed);
                            if sh >= SHARDS {
                                break;
                            }
                            let n: usize =
                                raw_clicks_per_thread.iter().map(|t| t[sh].len()).sum();
                            if n == 0 {
                                continue;
                            }
                            let cap = ((n * 3) / 2 + 16).next_power_of_two().max(64);
                            let mask = cap - 1;
                            table.clear();
                            table.resize(cap, 0);
                            unique.clear();
                            if unique.capacity() < n {
                                unique.reserve(n - unique.capacity());
                            }

                            for t in &raw_clicks_per_thread {
                                for &cl in &t[sh] {
                                    let tag = (cl.cid_hash >> 24) as u32;
                                    let mut pos = (cl.cid_hash as usize) & mask;
                                    loop {
                                        let entry = table[pos];
                                        if entry == 0 {
                                            let idx = unique.len() as u32;
                                            unique.push(cl);
                                            table[pos] =
                                                ((tag as u64) << 32) | ((idx as u64) + 1);
                                            break;
                                        }
                                        if (entry >> 32) as u32 == tag {
                                            let idx = ((entry & 0xFFFF_FFFF) - 1) as usize;
                                            let existing = &unique[idx];
                                            if existing.cid_hash == cl.cid_hash
                                                && existing.cid_ref.as_bytes(&all_bufs)
                                                    == cl.cid_ref.as_bytes(&all_bufs)
                                            {
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }

                            for cl in &unique {
                                out[shard_of(cl.person_hash)].push(ValidClick {
                                    person: cl.person,
                                    ts: cl.ts,
                                    cid_ref: cl.cid_ref,
                                    campaign_idx: cl.campaign_idx,
                                });
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    drop(raw_clicks_per_thread);

    // 7. Sort person-shards largest first (eliminates whale tail latency), sessionize,
    //    and write user_daily/part-{tid}.parquet directly from each worker thread!
    fs::create_dir_all(out_dir.join("user_daily")).unwrap();
    fs::create_dir_all(out_dir.join("campaign_daily")).unwrap();

    let mut shard_order: Vec<(usize, usize)> = (0..SHARDS)
        .map(|ps| {
            let cnt: usize = event_person_shards.iter().map(|t| t[ps].len()).sum();
            (ps, cnt)
        })
        .collect();
    shard_order.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let num_campaigns = camp_names.len();
    let num_categories = categories.len();
    let use_flat_camp = num_days_i64 > 0
        && num_days_i64 <= 4096
        && (num_days_i64 as usize).saturating_mul(num_campaigns) <= 2_000_000;
    let flat_camp_len = if use_flat_camp {
        (num_days_i64 as usize) * num_campaigns
    } else {
        0
    };

    let user_schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let next_shard = AtomicUsize::new(0);
    let camp_results: Vec<(Vec<CampaignDayAgg>, FxHashMap<(u16, i32), CampaignDayAgg>)> =
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..ncpu)
                .map(|tid| {
                    let user_schema = user_schema.clone();
                    let out_dir = &out_dir;
                    let all_bufs = &all_bufs;
                    let event_person_shards = &event_person_shards;
                    let click_person_shards = &click_person_shards;
                    let shard_order = &shard_order;
                    let categories = &categories;
                    let next_shard = &next_shard;
                    s.spawn(move || {
                        let mut shard_events: Vec<ValidEvent> = Vec::new();
                        let mut shard_clicks: Vec<ValidClick> = Vec::new();
                        let mut camp_counts = vec![0u32; num_campaigns];
                        let mut active_camps: Vec<u16> = Vec::new();
                        let mut cat_rev_arr = vec![0i64; num_categories];
                        let mut touched_cats: Vec<u16> = Vec::new();

                        let mut camp_flat = if use_flat_camp {
                            vec![CampaignDayAgg::default(); flat_camp_len]
                        } else {
                            Vec::new()
                        };
                        let mut camp_map: FxHashMap<(u16, i32), CampaignDayAgg> =
                            FxHashMap::default();

                        // Direct Arrow column builders for this thread's user_daily part
                        let mut uid_builder = StringBuilder::with_capacity(65536, 65536 * 10);
                        let mut days_vec: Vec<i32> = Vec::with_capacity(65536);
                        let mut sess_vec: Vec<i64> = Vec::with_capacity(65536);
                        let mut ev_vec: Vec<i64> = Vec::with_capacity(65536);
                        let mut rev_vec: Vec<i64> = Vec::with_capacity(65536);
                        let mut cat_builder = StringBuilder::with_capacity(65536, 65536 * 4);
                        let mut uid_scratch = [0u8; 16];

                        loop {
                            let ord_idx = next_shard.fetch_add(1, AtomicOrd::Relaxed);
                            if ord_idx >= SHARDS {
                                break;
                            }
                            let (ps, ev_total) = shard_order[ord_idx];
                            if ev_total == 0 {
                                continue;
                            }

                            shard_events.clear();
                            if shard_events.capacity() < ev_total {
                                shard_events.reserve(ev_total - shard_events.capacity());
                            }
                            for t in event_person_shards {
                                shard_events.extend_from_slice(&t[ps]);
                            }
                            shard_events.sort_unstable_by(|a, b| {
                                a.person
                                    .cmp_key(b.person, all_bufs)
                                    .then_with(|| a.ts.cmp(&b.ts))
                            });

                            let cl_total: usize =
                                click_person_shards.iter().map(|t| t[ps].len()).sum();
                            shard_clicks.clear();
                            if cl_total > 0 {
                                if shard_clicks.capacity() < cl_total {
                                    shard_clicks.reserve(cl_total - shard_clicks.capacity());
                                }
                                for t in click_person_shards {
                                    shard_clicks.extend_from_slice(&t[ps]);
                                }
                                shard_clicks.sort_unstable_by(|a, b| {
                                    a.person
                                        .cmp_key(b.person, all_bufs)
                                        .then_with(|| a.ts.cmp(&b.ts))
                                        .then_with(|| {
                                            b.cid_ref
                                                .as_bytes(all_bufs)
                                                .cmp(a.cid_ref.as_bytes(all_bufs))
                                        })
                                });
                            }

                            let mut ei = 0usize;
                            let mut ci = 0usize;
                            let ev_len = shard_events.len();
                            let cl_len = shard_clicks.len();

                            while ei < ev_len {
                                let person = shard_events[ei].person;
                                let e_start = ei;
                                ei += 1;
                                while ei < ev_len && shard_events[ei].person.eq_key(person, all_bufs)
                                {
                                    ei += 1;
                                }
                                let p_events = &shard_events[e_start..ei];

                                while ci < cl_len
                                    && shard_clicks[ci].person.cmp_key(person, all_bufs)
                                        == Ordering::Less
                                {
                                    ci += 1;
                                }
                                let c_start = ci;
                                while ci < cl_len && shard_clicks[ci].person.eq_key(person, all_bufs)
                                {
                                    ci += 1;
                                }
                                let p_clicks = &shard_clicks[c_start..ci];

                                let uid_str = std::str::from_utf8(
                                    person.as_bytes(&mut uid_scratch, all_bufs),
                                )
                                .unwrap();

                                for &c in &active_camps {
                                    camp_counts[c as usize] = 0;
                                }
                                active_camps.clear();

                                let mut left = 0usize;
                                let mut right = 0usize;

                                let mut cur_day = i32::MIN;
                                let mut day_sessions = 0i64;
                                let mut day_events = 0i64;
                                let mut day_rev = 0i64;

                                let mut k = 0usize;
                                while k < p_events.len() {
                                    let sess_start = p_events[k].ts;
                                    let day = sess_start.div_euclid(MS_PER_DAY) as i32;

                                    if day != cur_day {
                                        if cur_day != i32::MIN {
                                            // Flush previous day
                                            uid_builder.append_value(uid_str);
                                            days_vec.push(cur_day);
                                            sess_vec.push(day_sessions);
                                            ev_vec.push(day_events);
                                            rev_vec.push(day_rev);
                                            if touched_cats.is_empty() {
                                                cat_builder.append_null();
                                            } else {
                                                let mut best_c = touched_cats[0];
                                                let mut best_r = cat_rev_arr[best_c as usize];
                                                cat_rev_arr[best_c as usize] = 0;
                                                for &c in &touched_cats[1..] {
                                                    let r = cat_rev_arr[c as usize];
                                                    cat_rev_arr[c as usize] = 0;
                                                    if r > best_r
                                                        || (r == best_r
                                                            && categories[c as usize]
                                                                < categories[best_c as usize])
                                                    {
                                                        best_r = r;
                                                        best_c = c;
                                                    }
                                                }
                                                touched_cats.clear();
                                                cat_builder.append_value(categories[best_c as usize]);
                                            }
                                        }
                                        cur_day = day;
                                        day_sessions = 0;
                                        day_events = 0;
                                        day_rev = 0;
                                    }

                                    let mut sess_events = 0i64;
                                    let mut sess_rev = 0i64;
                                    let mut has_purchase = false;
                                    let mut prev_ts = sess_start;

                                    while k < p_events.len() {
                                        let ev = &p_events[k];
                                        if ev.ts - prev_ts > SESSION_GAP_MS {
                                            break;
                                        }
                                        prev_ts = ev.ts;
                                        sess_events += 1;
                                        let price = (ev.price_cat & 0xFFFF_FFFF) as i64;
                                        if price > 0 {
                                            has_purchase = true;
                                            sess_rev += price;
                                            let cat = (ev.price_cat >> 32) as usize;
                                            if cat_rev_arr[cat] == 0 {
                                                touched_cats.push(cat as u16);
                                            }
                                            cat_rev_arr[cat] += price;
                                        }
                                        k += 1;
                                    }
                                    day_sessions += 1;
                                    day_events += sess_events;
                                    day_rev += sess_rev;

                                    while right < p_clicks.len() && p_clicks[right].ts <= sess_start
                                    {
                                        let c = p_clicks[right].campaign_idx;
                                        if camp_counts[c as usize] == 0 {
                                            active_camps.push(c);
                                        }
                                        camp_counts[c as usize] += 1;
                                        right += 1;
                                    }
                                    let min_click_ts = sess_start - ASSIST_WINDOW_MS;
                                    while left < right && p_clicks[left].ts < min_click_ts {
                                        let c = p_clicks[left].campaign_idx;
                                        camp_counts[c as usize] -= 1;
                                        left += 1;
                                    }

                                    let credited_camp = if left < right {
                                        p_clicks[right - 1].campaign_idx
                                    } else {
                                        0
                                    };

                                    if use_flat_camp {
                                        let day_off = ((day - min_day) as usize) * num_campaigns;
                                        let cr = &mut camp_flat[day_off + credited_camp as usize];
                                        cr.sessions += 1;
                                        if has_purchase {
                                            cr.purchasing_sessions += 1;
                                            cr.revenue_cents += sess_rev;
                                        }
                                        if left < right {
                                            active_camps.retain(|&c| camp_counts[c as usize] > 0);
                                            for &c in &active_camps {
                                                camp_flat[day_off + c as usize]
                                                    .assisted_sessions += 1;
                                            }
                                        } else {
                                            active_camps.clear();
                                        }
                                    } else {
                                        let cr =
                                            camp_map.entry((credited_camp, day)).or_default();
                                        cr.sessions += 1;
                                        if has_purchase {
                                            cr.purchasing_sessions += 1;
                                            cr.revenue_cents += sess_rev;
                                        }
                                        if left < right {
                                            active_camps.retain(|&c| camp_counts[c as usize] > 0);
                                            for &c in &active_camps {
                                                camp_map
                                                    .entry((c, day))
                                                    .or_default()
                                                    .assisted_sessions += 1;
                                            }
                                        } else {
                                            active_camps.clear();
                                        }
                                    }
                                }

                                // Flush final day for this person
                                if cur_day != i32::MIN {
                                    uid_builder.append_value(uid_str);
                                    days_vec.push(cur_day);
                                    sess_vec.push(day_sessions);
                                    ev_vec.push(day_events);
                                    rev_vec.push(day_rev);
                                    if touched_cats.is_empty() {
                                        cat_builder.append_null();
                                    } else {
                                        let mut best_c = touched_cats[0];
                                        let mut best_r = cat_rev_arr[best_c as usize];
                                        cat_rev_arr[best_c as usize] = 0;
                                        for &c in &touched_cats[1..] {
                                            let r = cat_rev_arr[c as usize];
                                            cat_rev_arr[c as usize] = 0;
                                            if r > best_r
                                                || (r == best_r
                                                    && categories[c as usize]
                                                        < categories[best_c as usize])
                                            {
                                                best_r = r;
                                                best_c = c;
                                            }
                                        }
                                        touched_cats.clear();
                                        cat_builder.append_value(categories[best_c as usize]);
                                    }
                                }
                            }
                        }

                        // Write this thread's user_daily part immediately
                        if !days_vec.is_empty() || tid == 0 {
                            let batch = RecordBatch::try_new(
                                user_schema.clone(),
                                vec![
                                    Arc::new(uid_builder.finish()),
                                    Arc::new(Date32Array::from(days_vec)),
                                    Arc::new(Int64Array::from(sess_vec)),
                                    Arc::new(Int64Array::from(ev_vec)),
                                    Arc::new(Int64Array::from(rev_vec)),
                                    Arc::new(cat_builder.finish()),
                                ],
                            )
                            .unwrap();

                            let file = File::create(
                                out_dir.join(format!("user_daily/part-{tid}.parquet")),
                            )
                            .unwrap();
                            let props = WriterProperties::builder()
                                .set_compression(Compression::UNCOMPRESSED)
                                .set_dictionary_enabled(false)
                                .set_statistics_enabled(EnabledStatistics::None)
                                .build();
                            let mut writer =
                                ArrowWriter::try_new(file, user_schema, Some(props)).unwrap();
                            writer.write(&batch).unwrap();
                            writer.close().unwrap();
                        }

                        (camp_flat, camp_map)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

    // 8. Merge campaign_daily and write campaign_daily/part-0.parquet
    let mut cid_builder = StringBuilder::new();
    let mut ch_builder = StringBuilder::new();
    let mut days_vec = Vec::new();
    let mut sess_vec = Vec::new();
    let mut psess_vec = Vec::new();
    let mut rev_vec = Vec::new();
    let mut ass_vec = Vec::new();

    if use_flat_camp {
        let mut merged = vec![CampaignDayAgg::default(); flat_camp_len];
        for (flat, _) in &camp_results {
            for i in 0..flat_camp_len {
                let v = flat[i];
                if v.sessions != 0 || v.assisted_sessions != 0 {
                    merged[i].sessions += v.sessions;
                    merged[i].purchasing_sessions += v.purchasing_sessions;
                    merged[i].revenue_cents += v.revenue_cents;
                    merged[i].assisted_sessions += v.assisted_sessions;
                }
            }
        }
        for d in 0..(num_days_i64 as usize) {
            let day = min_day + (d as i32);
            let base = d * num_campaigns;
            for c in 0..num_campaigns {
                let v = merged[base + c];
                if v.sessions != 0 || v.assisted_sessions != 0 {
                    cid_builder.append_value(camp_names[c]);
                    ch_builder.append_value(camp_channels[c]);
                    days_vec.push(day);
                    sess_vec.push(v.sessions);
                    psess_vec.push(v.purchasing_sessions);
                    rev_vec.push(v.revenue_cents);
                    ass_vec.push(v.assisted_sessions);
                }
            }
        }
    } else {
        let mut merged: FxHashMap<(u16, i32), CampaignDayAgg> = FxHashMap::default();
        for (_, cmap) in &camp_results {
            for (&k, &v) in cmap {
                let e = merged.entry(k).or_default();
                e.sessions += v.sessions;
                e.purchasing_sessions += v.purchasing_sessions;
                e.revenue_cents += v.revenue_cents;
                e.assisted_sessions += v.assisted_sessions;
            }
        }
        for ((c, day), v) in merged {
            cid_builder.append_value(camp_names[c as usize]);
            ch_builder.append_value(camp_channels[c as usize]);
            days_vec.push(day);
            sess_vec.push(v.sessions);
            psess_vec.push(v.purchasing_sessions);
            rev_vec.push(v.revenue_cents);
            ass_vec.push(v.assisted_sessions);
        }
    }

    let camp_schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        camp_schema.clone(),
        vec![
            Arc::new(cid_builder.finish()),
            Arc::new(ch_builder.finish()),
            Arc::new(Date32Array::from(days_vec)),
            Arc::new(Int64Array::from(sess_vec)),
            Arc::new(Int64Array::from(psess_vec)),
            Arc::new(Int64Array::from(rev_vec)),
            Arc::new(Int64Array::from(ass_vec)),
        ],
    )
    .unwrap();

    let file = File::create(out_dir.join("campaign_daily/part-0.parquet")).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let mut writer = ArrowWriter::try_new(file, camp_schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    std::process::exit(0);
}