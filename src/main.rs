use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrd};
use std::sync::{Barrier, OnceLock};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const SESSION_GAP_MS: u32 = 1_800_000;
const ASSIST_WINDOW_MS: i64 = 7 * MS_PER_DAY;
const ASSIST_WINDOW_U32: u32 = (7 * MS_PER_DAY) as u32;
const DEDUP_SHARDS: usize = 128;
const PERSON_SHARDS: usize = 64;
const CHUNK_SIZE: usize = 2 * 1024 * 1024;
const BLOOM_BITS: usize = 1 << 21; // 256 KB (L2-resident)
const BLOOM_MASK: usize = BLOOM_BITS - 1;

// ============================================================================
// Zero-copy mmap (never unmapped; reclaimed instantaneously on exit(0))
// ============================================================================

struct MappedFile {
    ptr: *const u8,
    len: usize,
}

unsafe impl Sync for MappedFile {}
unsafe impl Send for MappedFile {}

impl MappedFile {
    fn open(path: &Path) -> Self {
        let f = File::open(path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        unsafe {
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        }
        Self {
            ptr: ptr as *const u8,
            len,
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

// ============================================================================
// Order-preserving 64-bit packed string (Pack64)
// Inline (bit 63 == 0): 1..=10 chars from [0-9A-Za-z_], 6 bits/char big-endian
// Overflow (bit 63 == 1): [1:1][arena_id:8][offset:32][len:23]
// ============================================================================

const ENCODE_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut c = b'0';
    while c <= b'9' {
        t[c as usize] = 1 + (c - b'0');
        c += 1;
    }
    let mut c = b'A';
    while c <= b'Z' {
        t[c as usize] = 11 + (c - b'A');
        c += 1;
    }
    t[b'_' as usize] = 37;
    let mut c = b'a';
    while c <= b'z' {
        t[c as usize] = 38 + (c - b'a');
        c += 1;
    }
    t
};

const DECODE_LUT: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 10 {
        t[1 + i] = b'0' + (i as u8);
        i += 1;
    }
    let mut i = 0;
    while i < 26 {
        t[11 + i] = b'A' + (i as u8);
        i += 1;
    }
    t[37] = b'_';
    let mut i = 0;
    while i < 26 {
        t[38 + i] = b'a' + (i as u8);
        i += 1;
    }
    t
};

#[inline(always)]
fn pack64_try_inline(b: &[u8]) -> Option<u64> {
    let len = b.len();
    if len > 10 {
        return None;
    }
    let mut acc = 0u64;
    let mut i = 0;
    while i < len {
        let code = unsafe { *ENCODE_LUT.get_unchecked(*b.get_unchecked(i) as usize) };
        if code == 0 {
            return None;
        }
        acc |= (code as u64) << (57 - i * 6);
        i += 1;
    }
    Some(acc)
}

#[inline(always)]
fn pack64(b: &[u8], arena_id: u8, arena: &mut Vec<u8>) -> u64 {
    if let Some(v) = pack64_try_inline(b) {
        return v;
    }
    let off = arena.len() as u64;
    arena.extend_from_slice(b);
    (1u64 << 63) | ((arena_id as u64) << 55) | (off << 23) | (b.len() as u64)
}

#[inline(always)]
fn overflow_bytes<'a>(v: u64, arenas: &'a [Vec<u8>]) -> &'a [u8] {
    let aid = ((v >> 55) & 0xFF) as usize;
    let off = ((v >> 23) & 0xFFFF_FFFF) as usize;
    let len = (v & 0x7F_FFFF) as usize;
    &arenas[aid][off..off + len]
}

#[inline(always)]
fn decode_inline(v: u64, out: &mut [u8; 10]) -> usize {
    let mut len = 0;
    while len < 10 {
        let code = ((v >> (57 - len * 6)) & 63) as usize;
        if code == 0 {
            break;
        }
        out[len] = DECODE_LUT[code];
        len += 1;
    }
    len
}

#[inline(always)]
fn pack64_bytes<'a>(v: u64, scratch: &'a mut [u8; 10], arenas: &'a [Vec<u8>]) -> &'a [u8] {
    if (v >> 63) == 0 {
        let n = decode_inline(v, scratch);
        &scratch[..n]
    } else {
        overflow_bytes(v, arenas)
    }
}

#[inline(always)]
fn eq_pack64(a: u64, b: u64, arenas: &[Vec<u8>]) -> bool {
    if ((a | b) >> 63) == 0 {
        a == b
    } else if (a >> 63) != (b >> 63) {
        false
    } else {
        overflow_bytes(a, arenas) == overflow_bytes(b, arenas)
    }
}

#[inline(always)]
fn cmp_pack64(a: u64, b: u64, arenas: &[Vec<u8>]) -> Ordering {
    if ((a | b) >> 63) == 0 {
        a.cmp(&b)
    } else {
        let mut sa = [0u8; 10];
        let mut sb = [0u8; 10];
        pack64_bytes(a, &mut sa, arenas).cmp(pack64_bytes(b, &mut sb, arenas))
    }
}

#[inline(always)]
fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^ (k >> 33)
}

#[inline(always)]
fn hash_bytes(bytes: &[u8]) -> u64 {
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
    mix64(h)
}

#[inline(always)]
fn hash_pack64(v: u64, arenas: &[Vec<u8>]) -> u64 {
    if (v >> 63) == 0 {
        mix64(v)
    } else {
        hash_bytes(overflow_bytes(v, arenas))
    }
}

// ============================================================================
// L1-hot write-combining shard buffer (eliminates L1 DTLB thrashing)
// ============================================================================

struct ShardWriter<T: Copy, const S: usize, const B: usize> {
    bufs: Box<[[T; B]; S]>,
    lens: [u8; S],
    out: Vec<Vec<T>>,
}

impl<T: Copy, const S: usize, const B: usize> ShardWriter<T, S, B> {
    fn new(zero: T, est_per_shard: usize) -> Self {
        let out = (0..S).map(|_| Vec::with_capacity(est_per_shard)).collect();
        Self {
            bufs: vec![[zero; B]; S].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!()),
            lens: [0u8; S],
            out,
        }
    }

    #[inline(always)]
    fn push(&mut self, shard: usize, val: T) {
        let len = unsafe { *self.lens.get_unchecked(shard) } as usize;
        unsafe {
            *self.bufs.get_unchecked_mut(shard).get_unchecked_mut(len) = val;
        }
        let next = len + 1;
        if next == B {
            unsafe {
                self.out
                    .get_unchecked_mut(shard)
                    .extend_from_slice(self.bufs.get_unchecked(shard));
                *self.lens.get_unchecked_mut(shard) = 0;
            }
        } else {
            unsafe {
                *self.lens.get_unchecked_mut(shard) = next as u8;
            }
        }
    }

    fn finish(mut self) -> Vec<Vec<T>> {
        for s in 0..S {
            let len = self.lens[s] as usize;
            if len > 0 {
                self.out[s].extend_from_slice(&self.bufs[s][..len]);
            }
        }
        self.out
    }
}

// ============================================================================
// Minimal FlatHashMap for fallback dimension tables
// ============================================================================

struct BytesMap<V: Copy> {
    keys: Vec<Vec<u8>>,
    vals: Vec<V>,
    table: Vec<u32>,
    mask: usize,
}

impl<V: Copy> BytesMap<V> {
    fn with_capacity(cap_hint: usize) -> Self {
        let cap = (cap_hint * 2).next_power_of_two().max(64);
        Self {
            keys: Vec::with_capacity(cap_hint),
            vals: Vec::with_capacity(cap_hint),
            table: vec![0; cap],
            mask: cap - 1,
        }
    }

    fn grow(&mut self) {
        let new_cap = self.table.len() * 2;
        self.table = vec![0; new_cap];
        self.mask = new_cap - 1;
        for (i, k) in self.keys.iter().enumerate() {
            let mut pos = (hash_bytes(k) as usize) & self.mask;
            while self.table[pos] != 0 {
                pos = (pos + 1) & self.mask;
            }
            self.table[pos] = (i as u32) + 1;
        }
    }

    fn insert(&mut self, key: &[u8], val: V) -> V {
        if (self.keys.len() + 1) * 10 >= self.table.len() * 7 {
            self.grow();
        }
        let mut pos = (hash_bytes(key) as usize) & self.mask;
        loop {
            let e = self.table[pos];
            if e == 0 {
                let idx = self.keys.len();
                self.keys.push(key.to_vec());
                self.vals.push(val);
                self.table[pos] = (idx as u32) + 1;
                return val;
            }
            let idx = (e - 1) as usize;
            if self.keys[idx].as_slice() == key {
                self.vals[idx] = val;
                return val;
            }
            pos = (pos + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn get(&self, key: &[u8]) -> Option<V> {
        let mut pos = (hash_bytes(key) as usize) & self.mask;
        loop {
            let e = self.table[pos];
            if e == 0 {
                return None;
            }
            let idx = (e - 1) as usize;
            if self.keys[idx].as_slice() == key {
                return Some(self.vals[idx]);
            }
            pos = (pos + 1) & self.mask;
        }
    }
}

// ============================================================================
// Fast CSV Scanner: Runtime AVX2 (32B/iter) + Borrow-free SWAR fallback
// ============================================================================

#[inline(always)]
fn has_byte_mask_swar(word: u64, byte: u8) -> u64 {
    let x = word ^ (0x0101_0101_0101_0101u64 * (byte as u64));
    let non_zero =
        (x | ((x & 0x7F7F_7F7F_7F7F_7F7F) + 0x7F7F_7F7F_7F7F_7F7F)) & 0x8080_8080_8080_8080;
    non_zero ^ 0x8080_8080_8080_8080
}

#[inline(always)]
fn has_quote_fast(b: &[u8]) -> bool {
    let len = b.len();
    let mut i = 0;
    while i + 8 <= len {
        let w = u64::from_le_bytes(unsafe { *(b.as_ptr().add(i) as *const [u8; 8]) });
        if has_byte_mask_swar(w, b'"') != 0 {
            return true;
        }
        i += 8;
    }
    while i < len {
        if unsafe { *b.get_unchecked(i) } == b'"' {
            return true;
        }
        i += 1;
    }
    false
}

#[inline(always)]
fn unquote(f: &[u8]) -> Option<&[u8]> {
    let len = f.len();
    if len >= 2
        && unsafe { *f.get_unchecked(0) == b'"' && *f.get_unchecked(len - 1) == b'"' }
    {
        let inner = &f[1..len - 1];
        if !has_quote_fast(inner) {
            return Some(inner);
        }
        return None;
    }
    if has_quote_fast(f) {
        return None;
    }
    Some(f)
}

#[inline(always)]
fn scan_slice_scalar<const N: usize, F: FnMut(&[u8], &[usize; N], usize, bool)>(
    buf: &[u8],
    mut on_line: F,
) {
    let len = buf.len();
    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut c_cnt = 0usize;
    let mut has_q = false;
    let mut i = 0usize;

    while i + 8 <= len {
        let w = u64::from_le_bytes(unsafe { *(buf.as_ptr().add(i) as *const [u8; 8]) });
        let mut nl = has_byte_mask_swar(w, b'\n');
        let mut cm = has_byte_mask_swar(w, b',');
        let mut qt = has_byte_mask_swar(w, b'"');

        while nl != 0 {
            let nl_bit = nl.trailing_zeros();
            let before = (1u64 << nl_bit) - 1;
            let mut cm_b = cm & before;
            while cm_b != 0 {
                let pos = i + (cm_b.trailing_zeros() as usize >> 3);
                if c_cnt < N {
                    commas[c_cnt] = pos - line_start;
                }
                c_cnt += 1;
                cm_b &= cm_b - 1;
            }
            if (qt & before) != 0 {
                has_q = true;
            }
            let nl_pos = i + (nl_bit as usize >> 3);
            if nl_pos > line_start {
                on_line(&buf[line_start..nl_pos], &commas, c_cnt, has_q);
            }
            let clear = (!0u64).checked_shl(nl_bit + 1).unwrap_or(0);
            nl &= clear;
            cm &= clear;
            qt &= clear;
            line_start = nl_pos + 1;
            c_cnt = 0;
            has_q = false;
        }
        while cm != 0 {
            let pos = i + (cm.trailing_zeros() as usize >> 3);
            if c_cnt < N {
                commas[c_cnt] = pos - line_start;
            }
            c_cnt += 1;
            cm &= cm - 1;
        }
        if qt != 0 {
            has_q = true;
        }
        i += 8;
    }

    while i < len {
        let b = unsafe { *buf.get_unchecked(i) };
        if b == b'\n' {
            if i > line_start {
                on_line(&buf[line_start..i], &commas, c_cnt, has_q);
            }
            line_start = i + 1;
            c_cnt = 0;
            has_q = false;
        } else if b == b',' {
            if c_cnt < N {
                commas[c_cnt] = i - line_start;
            }
            c_cnt += 1;
        } else if b == b'"' {
            has_q = true;
        }
        i += 1;
    }
    if line_start < len {
        on_line(&buf[line_start..len], &commas, c_cnt, has_q);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_slice_avx2<const N: usize, F: FnMut(&[u8], &[usize; N], usize, bool)>(
    buf: &[u8],
    mut on_line: F,
) {
    use std::arch::x86_64::*;
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);

    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut c_cnt = 0usize;
    let mut has_q = false;
    let mut i = 0usize;

    while i + 32 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let mut nl = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_nl)) as u32;
        let mut cm = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_cm)) as u32;
        let mut qt = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_qt)) as u32;

        while nl != 0 {
            let nl_bit = nl.trailing_zeros();
            let before = (1u32 << nl_bit) - 1;
            let mut cm_b = cm & before;
            while cm_b != 0 {
                let pos = i + (cm_b.trailing_zeros() as usize);
                if c_cnt < N {
                    commas[c_cnt] = pos - line_start;
                }
                c_cnt += 1;
                cm_b &= cm_b - 1;
            }
            if (qt & before) != 0 {
                has_q = true;
            }
            let nl_pos = i + (nl_bit as usize);
            if nl_pos > line_start {
                on_line(&buf[line_start..nl_pos], &commas, c_cnt, has_q);
            }
            let clear = (!0u32).checked_shl(nl_bit + 1).unwrap_or(0);
            nl &= clear;
            cm &= clear;
            qt &= clear;
            line_start = nl_pos + 1;
            c_cnt = 0;
            has_q = false;
        }
        while cm != 0 {
            let pos = i + (cm.trailing_zeros() as usize);
            if c_cnt < N {
                commas[c_cnt] = pos - line_start;
            }
            c_cnt += 1;
            cm &= cm - 1;
        }
        if qt != 0 {
            has_q = true;
        }
        i += 32;
    }

    while i < len {
        let b = *buf.get_unchecked(i);
        if b == b'\n' {
            if i > line_start {
                on_line(&buf[line_start..i], &commas, c_cnt, has_q);
            }
            line_start = i + 1;
            c_cnt = 0;
            has_q = false;
        } else if b == b',' {
            if c_cnt < N {
                commas[c_cnt] = i - line_start;
            }
            c_cnt += 1;
        } else if b == b'"' {
            has_q = true;
        }
        i += 1;
    }
    if line_start < len {
        on_line(&buf[line_start..len], &commas, c_cnt, has_q);
    }
}

#[inline(always)]
fn scan_slice<const N: usize, F: FnMut(&[u8], &[usize; N], usize, bool)>(
    buf: &[u8],
    use_avx2: bool,
    on_line: F,
) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        unsafe { scan_slice_avx2::<N, F>(buf, on_line) };
        return;
    }
    let _ = use_avx2;
    scan_slice_scalar::<N, F>(buf, on_line);
}

#[inline(always)]
fn find_nl(buf: &[u8], start: usize) -> usize {
    let mut i = start;
    let len = buf.len();
    while i + 8 <= len {
        let w = u64::from_le_bytes(unsafe { *(buf.as_ptr().add(i) as *const [u8; 8]) });
        let m = has_byte_mask_swar(w, b'\n');
        if m != 0 {
            return i + (m.trailing_zeros() as usize >> 3) + 1;
        }
        i += 8;
    }
    while i < len {
        if unsafe { *buf.get_unchecked(i) } == b'\n' {
            return i + 1;
        }
        i += 1;
    }
    len
}

// ============================================================================
// Fast DateCache (1-entry register hit + 64-entry L1 table) & Timestamp Parsing
// ============================================================================

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
    if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
        return None;
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
    last_hi: u64,
    last_lo: u16,
    last_ms: i64,
    keys_hi: [u64; 64],
    keys_lo: [u16; 64],
    vals: [i64; 64],
}

impl DateCache {
    fn new() -> Self {
        Self {
            last_hi: 0,
            last_lo: 0,
            last_ms: 0,
            keys_hi: [0; 64],
            keys_lo: [0; 64],
            vals: [0; 64],
        }
    }

    #[inline(always)]
    fn parse_datetime_prefix(&mut self, b: &[u8]) -> Option<i64> {
        if b.len() < 19 || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let p = b.as_ptr();
        let hi = u64::from_le_bytes(unsafe { *(p as *const [u8; 8]) });
        let lo = u16::from_le_bytes(unsafe { *(p.add(8) as *const [u8; 2]) });

        let day_ms = if hi == self.last_hi && lo == self.last_lo {
            self.last_ms
        } else {
            let slot = (((b[6] as usize) << 5) ^ ((b[8] as usize) << 4) ^ (b[9] as usize)) & 63;
            if self.keys_hi[slot] == hi && self.keys_lo[slot] == lo {
                let ms = self.vals[slot];
                self.last_hi = hi;
                self.last_lo = lo;
                self.last_ms = ms;
                ms
            } else {
                if b[4] != b'-' || b[7] != b'-' {
                    return None;
                }
                let y = parse_4_digits(&b[0..4])?;
                let m = parse_2_digits(b[5], b[6])?;
                let d = parse_2_digits(b[8], b[9])?;
                let days = days_from_civil(y, m, d)?;
                let ms = days * MS_PER_DAY;
                self.keys_hi[slot] = hi;
                self.keys_lo[slot] = lo;
                self.vals[slot] = ms;
                self.last_hi = hi;
                self.last_lo = lo;
                self.last_ms = ms;
                ms
            }
        };
        let hh = parse_2_digits(unsafe { *p.add(11) }, unsafe { *p.add(12) })?;
        let mm = parse_2_digits(unsafe { *p.add(14) }, unsafe { *p.add(15) })?;
        let ss = parse_2_digits(unsafe { *p.add(17) }, unsafe { *p.add(18) })?;
        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        Some(day_ms + ((hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
    }

    #[inline(always)]
    fn parse_ts(&mut self, b: &[u8]) -> Option<i64> {
        let len = b.len();
        if len == 20 && unsafe { *b.get_unchecked(19) } == b'Z' {
            return self.parse_datetime_prefix(b);
        }
        if len == 13 {
            let mut val: i64 = 0;
            let mut i = 0;
            while i < 13 {
                let d = unsafe { b.get_unchecked(i).wrapping_sub(b'0') };
                if d > 9 {
                    return None;
                }
                val = val * 10 + (d as i64);
                i += 1;
            }
            return Some(val);
        }
        if len == 25
            && (unsafe { *b.get_unchecked(19) } == b'+'
                || unsafe { *b.get_unchecked(19) } == b'-')
            && unsafe { *b.get_unchecked(22) } == b':'
        {
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

#[inline(always)]
fn parse_prefixed_uint(b: &[u8], prefix: &[u8]) -> Option<usize> {
    let plen = prefix.len();
    if b.len() <= plen || !b.starts_with(prefix) {
        return None;
    }
    // Reject leading zeros (e.g. "p01") so fallback handles them exact-match
    if b[plen] == b'0' && b.len() > plen + 1 {
        return None;
    }
    let mut val = 0usize;
    for &c in &b[plen..] {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.checked_mul(10)?.checked_add(d as usize)?;
        if val >= 1_000_000 {
            return None;
        }
    }
    Some(val)
}

fn parse_manifest_field(json: &[u8], key: &[u8]) -> String {
    let mut i = 0;
    while i + key.len() < json.len() {
        if &json[i..i + key.len()] == key {
            let mut j = i + key.len();
            while j < json.len() && json[j] != b':' {
                j += 1;
            }
            j += 1;
            while j < json.len() && json[j] != b'"' {
                j += 1;
            }
            j += 1;
            let start = j;
            while j < json.len() && json[j] != b'"' {
                j += 1;
            }
            return std::str::from_utf8(&json[start..j]).unwrap().to_owned();
        }
        i += 1;
    }
    panic!("missing manifest key");
}

// ============================================================================
// Linked IDs DSU + Flat u64->u64 Lookup + L2 Bloom Filter
// ============================================================================

struct IdLookup {
    has_links: bool,
    bloom: Vec<u64>,
    keys: Vec<u64>,
    vals: Vec<u64>,
    mask: usize,
}

impl IdLookup {
    fn from_pairs(all_pairs: &[Vec<(u64, u64)>], arenas: &[Vec<u8>]) -> Self {
        let total_pairs: usize = all_pairs.iter().map(|v| v.len()).sum();
        if total_pairs == 0 {
            return Self {
                has_links: false,
                bloom: Vec::new(),
                keys: Vec::new(),
                vals: Vec::new(),
                mask: 0,
            };
        }

        let est_nodes = (total_pairs * 2).max(64);
        let tab_cap = (est_nodes * 2).next_power_of_two().max(64);
        let mut tab = vec![0u32; tab_cap];
        let mut tab_mask = tab_cap - 1;
        let mut node_keys: Vec<u64> = Vec::with_capacity(est_nodes);
        let mut parent: Vec<u32> = Vec::with_capacity(est_nodes);

        #[inline(always)]
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            let mut root = x;
            while unsafe { *parent.get_unchecked(root as usize) } != root {
                root = unsafe { *parent.get_unchecked(root as usize) };
            }
            while unsafe { *parent.get_unchecked(x as usize) } != root {
                let nxt = unsafe { *parent.get_unchecked(x as usize) };
                unsafe { *parent.get_unchecked_mut(x as usize) = root };
                x = nxt;
            }
            root
        }

        let mut get_node = |k: u64,
                            tab: &mut Vec<u32>,
                            tab_mask: &mut usize,
                            node_keys: &mut Vec<u64>,
                            parent: &mut Vec<u32>|
         -> u32 {
            if (node_keys.len() + 1) * 10 >= tab.len() * 7 {
                let nc = tab.len() * 2;
                *tab = vec![0u32; nc];
                *tab_mask = nc - 1;
                for (i, &nk) in node_keys.iter().enumerate() {
                    let mut p = (hash_pack64(nk, arenas) as usize) & *tab_mask;
                    while tab[p] != 0 {
                        p = (p + 1) & *tab_mask;
                    }
                    tab[p] = (i as u32) + 1;
                }
            }
            let mut p = (hash_pack64(k, arenas) as usize) & *tab_mask;
            loop {
                let e = unsafe { *tab.get_unchecked(p) };
                if e == 0 {
                    let idx = node_keys.len() as u32;
                    node_keys.push(k);
                    parent.push(idx);
                    unsafe { *tab.get_unchecked_mut(p) = idx + 1 };
                    return idx;
                }
                let idx = (e - 1) as usize;
                if eq_pack64(unsafe { *node_keys.get_unchecked(idx) }, k, arenas) {
                    return idx as u32;
                }
                p = (p + 1) & *tab_mask;
            }
        };

        for pv in all_pairs {
            for &(a, b) in pv {
                let ia = get_node(a, &mut tab, &mut tab_mask, &mut node_keys, &mut parent);
                let ib = get_node(b, &mut tab, &mut tab_mask, &mut node_keys, &mut parent);
                let ra = find(&mut parent, ia);
                let rb = find(&mut parent, ib);
                if ra != rb {
                    if cmp_pack64(node_keys[ra as usize], node_keys[rb as usize], arenas)
                        == Ordering::Less
                    {
                        parent[rb as usize] = ra;
                    } else {
                        parent[ra as usize] = rb;
                    }
                }
            }
        }

        let n = node_keys.len();
        let mut mapped_cnt = 0usize;
        for i in 0..n {
            if find(&mut parent, i as u32) != i as u32 {
                mapped_cnt += 1;
            }
        }
        if mapped_cnt == 0 {
            return Self {
                has_links: false,
                bloom: Vec::new(),
                keys: Vec::new(),
                vals: Vec::new(),
                mask: 0,
            };
        }

        let map_cap = (mapped_cnt * 2).next_power_of_two().max(64);
        let mask = map_cap - 1;
        let mut keys = vec![0u64; map_cap];
        let mut vals = vec![0u64; map_cap];
        let mut bloom = vec![0u64; BLOOM_BITS / 64];

        for i in 0..n {
            let r = find(&mut parent, i as u32) as usize;
            if r != i {
                let k = node_keys[i];
                let v = node_keys[r];
                let h = hash_pack64(k, arenas);
                let b1 = (h as usize) & BLOOM_MASK;
                let b2 = ((h >> 21) as usize) & BLOOM_MASK;
                bloom[b1 >> 6] |= 1u64 << (b1 & 63);
                bloom[b2 >> 6] |= 1u64 << (b2 & 63);
                let mut pos = (h as usize) & mask;
                while keys[pos] != 0 {
                    pos = (pos + 1) & mask;
                }
                keys[pos] = k;
                vals[pos] = v;
            }
        }

        Self {
            has_links: true,
            bloom,
            keys,
            vals,
            mask,
        }
    }

    #[inline(always)]
    fn resolve_pack64(&self, p: u64, person_arenas: &[Vec<u8>]) -> (u64, u64) {
        let h = hash_pack64(p, person_arenas);
        if self.has_links {
            let b1 = (h as usize) & BLOOM_MASK;
            if (unsafe { *self.bloom.get_unchecked(b1 >> 6) } & (1u64 << (b1 & 63))) != 0 {
                let b2 = ((h >> 21) as usize) & BLOOM_MASK;
                if (unsafe { *self.bloom.get_unchecked(b2 >> 6) } & (1u64 << (b2 & 63))) != 0 {
                    let mut pos = (h as usize) & self.mask;
                    loop {
                        let k = unsafe { *self.keys.get_unchecked(pos) };
                        if k == 0 {
                            break;
                        }
                        if eq_pack64(k, p, person_arenas) {
                            let v = unsafe { *self.vals.get_unchecked(pos) };
                            return (v, hash_pack64(v, person_arenas));
                        }
                        pos = (pos + 1) & self.mask;
                    }
                }
            }
        }
        (p, h)
    }
}

// ============================================================================
// Zero-Copy Uncompressed PLAIN Parquet v1 Writer (Direct column writes)
// ============================================================================

struct ThriftBuf(Vec<u8>);

impl ThriftBuf {
    fn new() -> Self {
        Self(Vec::with_capacity(512))
    }

    #[inline]
    fn write_uvarint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(((v as u8) & 0x7F) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    #[inline]
    fn write_zigzag_i32(&mut self, v: i32) {
        let z = ((v << 1) ^ (v >> 31)) as u32;
        self.write_uvarint(z as u64);
    }

    #[inline]
    fn write_zigzag_i64(&mut self, v: i64) {
        let z = ((v << 1) ^ (v >> 63)) as u64;
        self.write_uvarint(z);
    }

    #[inline]
    fn write_field_hdr(&mut self, last_id: &mut i16, id: i16, wire_type: u8) {
        let delta = id - *last_id;
        if delta > 0 && delta <= 15 {
            self.0.push(((delta as u8) << 4) | wire_type);
        } else {
            self.0.push(wire_type);
            self.write_zigzag_i32(id as i32);
        }
        *last_id = id;
    }

    fn write_i32_field(&mut self, last_id: &mut i16, id: i16, v: i32) {
        self.write_field_hdr(last_id, id, 5);
        self.write_zigzag_i32(v);
    }

    fn write_i64_field(&mut self, last_id: &mut i16, id: i16, v: i64) {
        self.write_field_hdr(last_id, id, 6);
        self.write_zigzag_i64(v);
    }

    fn write_str(&mut self, s: &str) {
        self.write_uvarint(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }

    fn write_str_field(&mut self, last_id: &mut i16, id: i16, s: &str) {
        self.write_field_hdr(last_id, id, 8);
        self.write_str(s);
    }

    fn write_list_hdr(&mut self, last_id: &mut i16, id: i16, elem_type: u8, len: usize) {
        self.write_field_hdr(last_id, id, 9);
        if len < 15 {
            self.0.push(((len as u8) << 4) | elem_type);
        } else {
            self.0.push(0xF0 | elem_type);
            self.write_uvarint(len as u64);
        }
    }

    fn write_struct_start(&mut self, last_id: &mut i16, id: i16) {
        self.write_field_hdr(last_id, id, 12);
    }

    fn write_stop(&mut self) {
        self.0.push(0);
    }
}

#[derive(Clone, Copy)]
enum ColKind {
    Utf8Req,
    Utf8Opt,
    Date32,
    Int32,
    Int64,
}

struct ColSpec<'a> {
    name: &'a str,
    kind: ColKind,
    plain: &'a [u8],
    def_bits: Option<&'a [u8]>,
}

fn encode_rle_def_bits(def_bits: &[u8], out: &mut Vec<u8>) {
    let len_pos = out.len();
    out.extend_from_slice(&[0u8; 4]);
    let start = out.len();
    for chunk in def_bits.chunks(63) {
        let groups = chunk.len() as u8;
        out.push((groups << 1) | 1);
        out.extend_from_slice(chunk);
    }
    let rle_len = (out.len() - start) as u32;
    out[len_pos..len_pos + 4].copy_from_slice(&rle_len.to_le_bytes());
}

fn write_parquet_file(path: &Path, num_rows: usize, cols: &[ColSpec]) {
    let mut f = File::create(path).unwrap();
    f.write_all(b"PAR1").unwrap();
    let mut file_pos: i64 = 4;

    let mut chunk_offsets = Vec::with_capacity(cols.len());
    let mut chunk_sizes = Vec::with_capacity(cols.len());

    if num_rows > 0 {
        let mut rle_scratch = Vec::new();
        for c in cols {
            let page_off = file_pos;
            let payload_len = if let Some(bits) = c.def_bits {
                rle_scratch.clear();
                encode_rle_def_bits(bits, &mut rle_scratch);
                rle_scratch.len() + c.plain.len()
            } else {
                c.plain.len()
            };

            let mut hdr = ThriftBuf::new();
            let mut lid = 0i16;
            hdr.write_i32_field(&mut lid, 1, 0); // DATA_PAGE
            hdr.write_i32_field(&mut lid, 2, payload_len as i32);
            hdr.write_i32_field(&mut lid, 3, payload_len as i32);
            hdr.write_struct_start(&mut lid, 5);
            {
                let mut dlid = 0i16;
                hdr.write_i32_field(&mut dlid, 1, num_rows as i32);
                hdr.write_i32_field(&mut dlid, 2, 0); // PLAIN
                hdr.write_i32_field(&mut dlid, 3, 3); // RLE
                hdr.write_i32_field(&mut dlid, 4, 3); // RLE
                hdr.write_stop();
            }
            hdr.write_stop();

            if c.def_bits.is_some() {
                hdr.0.extend_from_slice(&rle_scratch);
            }
            f.write_all(&hdr.0).unwrap();
            f.write_all(c.plain).unwrap();
            let total_size = (hdr.0.len() + c.plain.len()) as i64;
            file_pos += total_size;
            chunk_offsets.push(page_off);
            chunk_sizes.push(total_size);
        }
    }

    let mut meta = ThriftBuf::new();
    let mut lid = 0i16;
    meta.write_i32_field(&mut lid, 1, 1);
    meta.write_list_hdr(&mut lid, 2, 12, cols.len() + 1);
    {
        let mut slid = 0i16;
        meta.write_str_field(&mut slid, 4, "schema");
        meta.write_i32_field(&mut slid, 5, cols.len() as i32);
        meta.write_stop();
    }
    for c in cols {
        let mut slid = 0i16;
        let (ptype, rep, ctype, ltype) = match c.kind {
            ColKind::Utf8Req => (6, 0, Some(0), Some(1)),
            ColKind::Utf8Opt => (6, 1, Some(0), Some(1)),
            ColKind::Date32 => (1, 0, Some(6), Some(6)),
            ColKind::Int32 => (1, 0, None, None),
            ColKind::Int64 => (2, 0, None, None),
        };
        meta.write_i32_field(&mut slid, 1, ptype);
        meta.write_i32_field(&mut slid, 3, rep);
        meta.write_str_field(&mut slid, 4, c.name);
        if let Some(ct) = ctype {
            meta.write_i32_field(&mut slid, 6, ct);
        }
        if let Some(lt) = ltype {
            meta.write_struct_start(&mut slid, 10);
            let mut ulid = 0i16;
            meta.write_struct_start(&mut ulid, lt);
            meta.write_stop();
            meta.write_stop();
        }
        meta.write_stop();
    }

    meta.write_i64_field(&mut lid, 3, num_rows as i64);

    if num_rows > 0 {
        meta.write_list_hdr(&mut lid, 4, 12, 1);
        let mut rglid = 0i16;
        meta.write_list_hdr(&mut rglid, 1, 12, cols.len());
        let mut rg_total_bytes = 0i64;
        for (idx, c) in cols.iter().enumerate() {
            let ptype = match c.kind {
                ColKind::Utf8Req | ColKind::Utf8Opt => 6,
                ColKind::Date32 | ColKind::Int32 => 1,
                ColKind::Int64 => 2,
            };
            let is_opt = matches!(c.kind, ColKind::Utf8Opt);
            let off = chunk_offsets[idx];
            let sz = chunk_sizes[idx];
            rg_total_bytes += sz;

            let mut clid = 0i16;
            meta.write_i64_field(&mut clid, 2, off);
            meta.write_struct_start(&mut clid, 3);
            {
                let mut mlid = 0i16;
                meta.write_i32_field(&mut mlid, 1, ptype);
                if is_opt {
                    meta.write_list_hdr(&mut mlid, 2, 5, 2);
                    meta.write_zigzag_i32(0);
                    meta.write_zigzag_i32(3);
                } else {
                    meta.write_list_hdr(&mut mlid, 2, 5, 1);
                    meta.write_zigzag_i32(0);
                }
                meta.write_list_hdr(&mut mlid, 3, 8, 1);
                meta.write_str(c.name);
                meta.write_i32_field(&mut mlid, 4, 0);
                meta.write_i64_field(&mut mlid, 5, num_rows as i64);
                meta.write_i64_field(&mut mlid, 6, sz);
                meta.write_i64_field(&mut mlid, 7, sz);
                meta.write_i64_field(&mut mlid, 9, off);
                meta.write_stop();
            }
            meta.write_stop();
        }
        meta.write_i64_field(&mut rglid, 2, rg_total_bytes);
        meta.write_i64_field(&mut rglid, 3, num_rows as i64);
        meta.write_stop();
    } else {
        meta.write_list_hdr(&mut lid, 4, 12, 0);
    }
    meta.write_stop();

    let meta_len = meta.0.len() as u32;
    meta.0.extend_from_slice(&meta_len.to_le_bytes());
    meta.0.extend_from_slice(b"PAR1");
    f.write_all(&meta.0).unwrap();
}

#[inline(always)]
fn push_plain_str(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s);
}

// ============================================================================
// Core Pipeline Structs
// ============================================================================

#[derive(Clone, Copy)]
struct RawEvent {
    eid: u64,
    ingest_ts: i64,
    raw_uid: u64,
    ts_off: u32,    // u32::MAX = invalid (kept for dedup shadowing only)
    price_cat: u32, // low 24 bits: price_cents (0 = not purchase), high 8 bits: cat_idx
}

const ZERO_RAW_EVENT: RawEvent = RawEvent {
    eid: 0,
    ingest_ts: 0,
    raw_uid: 0,
    ts_off: 0,
    price_cat: 0,
};

#[derive(Clone, Copy)]
struct ValidEvent {
    person: u64,
    ts_off: u32,
    price_cat: u32,
}

const ZERO_VALID_EVENT: ValidEvent = ValidEvent {
    person: 0,
    ts_off: 0,
    price_cat: 0,
};

#[derive(Clone, Copy)]
struct RawClick {
    cid: u64,
    raw_uid: u64,
    ts_off: u32,
    campaign_idx: u16,
}

const ZERO_RAW_CLICK: RawClick = RawClick {
    cid: 0,
    raw_uid: 0,
    ts_off: 0,
    campaign_idx: 0,
};

#[derive(Clone, Copy)]
struct ValidClick {
    person: u64,
    cid: u64,
    ts_off: u32,
    campaign_idx: u16,
}

const ZERO_VALID_CLICK: ValidClick = ValidClick {
    person: 0,
    cid: 0,
    ts_off: 0,
    campaign_idx: 0,
};

#[derive(Clone, Copy)]
struct ClickPacked {
    ts_off: u32,
    campaign_idx: u16,
    cid: u64,
}

#[derive(Default, Clone, Copy)]
struct CampaignDayAgg {
    sessions: i32,
    purchasing_sessions: i32,
    revenue_cents: i64,
    assisted_sessions: i32,
}

fn mmap_csv_dir(dir: &Path) -> Vec<MappedFile> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
        .collect();
    paths.sort();
    paths.iter().map(|p| MappedFile::open(p)).collect()
}

fn build_chunks(files: &[MappedFile]) -> Vec<(usize, usize, usize)> {
    let mut chunks = Vec::new();
    for (fi, mf) in files.iter().enumerate() {
        let b = mf.as_slice();
        if b.is_empty() {
            continue;
        }
        let header_end = find_nl(b, 0);
        let mut pos = header_end;
        while pos < b.len() {
            let target = (pos + CHUNK_SIZE).min(b.len());
            let end = if target == b.len() {
                b.len()
            } else {
                find_nl(b, target)
            };
            chunks.push((fi, pos, end));
            pos = end;
        }
    }
    chunks
}

struct SyncCell<T>(std::cell::UnsafeCell<Option<T>>);
unsafe impl<T: Send> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    fn new() -> Self {
        Self(std::cell::UnsafeCell::new(None))
    }
    unsafe fn set(&self, v: T) {
        *self.0.get() = Some(v);
    }
    unsafe fn get(&self) -> &T {
        (*self.0.get()).as_ref().unwrap()
    }
    unsafe fn take(&self) -> T {
        (*self.0.get()).take().unwrap()
    }
}

fn main() {
    let data_dir = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(
        std::env::var("OUTPUT_DIR")
            .or_else(|_| std::env::var("OUT_DIR"))
            .unwrap_or_else(|_| "/output".into()),
    );
    let ncpu: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        })
        .clamp(1, 64);

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let manifest_bytes = fs::read(data_dir.join("manifest.json")).unwrap();
    let ws_str = parse_manifest_field(&manifest_bytes, b"\"window_start\"");
    let we_str = parse_manifest_field(&manifest_bytes, b"\"window_end\"");
    let mut dc0 = DateCache::new();
    let win_start = dc0.parse_ts(ws_str.as_bytes()).unwrap();
    let win_end = dc0.parse_ts(we_str.as_bytes()).unwrap();
    let click_win_start = win_start - ASSIST_WINDOW_MS;
    assert!(win_end - click_win_start < (u32::MAX as i64));

    let min_day = win_start.div_euclid(MS_PER_DAY) as i32;
    let max_day = (win_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days = ((max_day as i64) - (min_day as i64) + 1) as usize;
    let win_start_rem = win_start.rem_euclid(MS_PER_DAY) as u64;

    // 1. Load products.csv & campaigns.csv (mmap)
    let mut categories: Vec<String> = vec!["unknown".to_owned()];
    let mut cat_to_idx: BytesMap<u8> = BytesMap::with_capacity(64);
    cat_to_idx.insert(b"unknown", 0);
    let mut prod_fast: Vec<u8> = Vec::new();
    let mut prod_fallback: BytesMap<u8> = BytesMap::with_capacity(64);
    {
        let mf = MappedFile::open(&data_dir.join("products.csv"));
        let b = mf.as_slice();
        let start = find_nl(b, 0);
        scan_slice::<1, _>(&b[start..], use_avx2, |line, c, cnt, hq| {
            if cnt != 1 {
                return;
            }
            let (p, cat) = if !hq {
                (&line[..c[0]], &line[c[0] + 1..])
            } else {
                let (Some(p), Some(cat)) = (unquote(&line[..c[0]]), unquote(&line[c[0] + 1..]))
                else {
                    return;
                };
                (p, cat)
            };
            let idx = match cat_to_idx.get(cat) {
                Some(i) => i,
                None => {
                    let i = categories.len() as u8;
                    categories.push(std::str::from_utf8(cat).unwrap().to_owned());
                    cat_to_idx.insert(cat, i);
                    i
                }
            };
            if let Some(pid) = parse_prefixed_uint(p, b"p") {
                if pid >= prod_fast.len() {
                    prod_fast.resize(pid + 1, 0);
                }
                prod_fast[pid] = idx;
            } else {
                prod_fallback.insert(p, idx);
            }
        });
    }

    let mut camp_names: Vec<String> = vec!["organic".to_owned()];
    let mut camp_channels: Vec<String> = vec!["organic".to_owned()];
    let mut camp_fast: Vec<u16> = Vec::new();
    let mut camp_fallback: BytesMap<u16> = BytesMap::with_capacity(64);
    {
        let mf = MappedFile::open(&data_dir.join("campaigns.csv"));
        let b = mf.as_slice();
        let start = find_nl(b, 0);
        scan_slice::<1, _>(&b[start..], use_avx2, |line, c, cnt, hq| {
            if cnt != 1 {
                return;
            }
            let (cp, ch) = if !hq {
                (&line[..c[0]], &line[c[0] + 1..])
            } else {
                let (Some(cp), Some(ch)) = (unquote(&line[..c[0]]), unquote(&line[c[0] + 1..]))
                else {
                    return;
                };
                (cp, ch)
            };
            let idx = camp_names.len() as u16;
            camp_names.push(std::str::from_utf8(cp).unwrap().to_owned());
            camp_channels.push(std::str::from_utf8(ch).unwrap().to_owned());
            if let Some(cid) = parse_prefixed_uint(cp, b"c_") {
                if cid >= camp_fast.len() {
                    camp_fast.resize(cid + 1, 0);
                }
                camp_fast[cid] = idx;
            } else {
                camp_fallback.insert(cp, idx);
            }
        });
    }

    let link_files = mmap_csv_dir(&data_dir.join("links"));
    let event_files = mmap_csv_dir(&data_dir.join("events"));
    let click_files = mmap_csv_dir(&data_dir.join("clicks"));

    let link_chunks = build_chunks(&link_files);
    let event_chunks = build_chunks(&event_files);
    let click_chunks = build_chunks(&click_files);

    let total_event_bytes: usize = event_files.iter().map(|f| f.len).sum();
    let total_click_bytes: usize = click_files.iter().map(|f| f.len).sum();
    let est_ev_dedup = ((total_event_bytes / 68) / (ncpu * DEDUP_SHARDS)).max(64);
    let est_ev_person = ((total_event_bytes / 72) / (ncpu * PERSON_SHARDS)).max(64);
    let est_cl_dedup = ((total_click_bytes / 45) / (ncpu * DEDUP_SHARDS)).max(32);
    let est_cl_person = ((total_click_bytes / 48) / (ncpu * PERSON_SHARDS)).max(32);

    fs::create_dir_all(out_dir.join("user_daily")).unwrap();
    fs::create_dir_all(out_dir.join("campaign_daily")).unwrap();

    // Shared synchronization state for single persistent thread pool
    let barrier = Barrier::new(ncpu);
    let next_link = AtomicUsize::new(0);
    let next_ev_chunk = AtomicUsize::new(0);
    let next_cl_chunk = AtomicUsize::new(0);
    let next_dedup_ev = AtomicUsize::new(0);
    let next_dedup_cl = AtomicUsize::new(0);
    let next_person_shard = AtomicUsize::new(0);

    let link_pairs_slots: Vec<SyncCell<Vec<(u64, u64)>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let link_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let id_lookup_cell: OnceLock<IdLookup> = OnceLock::new();

    let raw_ev_slots: Vec<SyncCell<Vec<Vec<RawEvent>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let eid_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let ev_parena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();

    let raw_cl_slots: Vec<SyncCell<Vec<Vec<RawClick>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let cid_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let cl_parena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();

    let person_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let eid_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let cid_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();

    let ev_person_slots: Vec<SyncCell<Vec<Vec<ValidEvent>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let cl_person_slots: Vec<SyncCell<Vec<Vec<ValidClick>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let shard_order_cell: OnceLock<Vec<(usize, usize)>> = OnceLock::new();

    let num_campaigns = camp_names.len();
    let num_categories = categories.len();
    let flat_camp_len = num_days * num_campaigns;

    // Single persistent worker pool — zero repeated thread spawns!
    let camp_results: Vec<Vec<CampaignDayAgg>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..ncpu)
            .map(|tid| {
                let barrier = &barrier;
                let next_link = &next_link;
                let next_ev_chunk = &next_ev_chunk;
                let next_cl_chunk = &next_cl_chunk;
                let next_dedup_ev = &next_dedup_ev;
                let next_dedup_cl = &next_dedup_cl;
                let next_person_shard = &next_person_shard;
                let link_files = &link_files;
                let event_files = &event_files;
                let click_files = &click_files;
                let link_chunks = &link_chunks;
                let event_chunks = &event_chunks;
                let click_chunks = &click_chunks;
                let link_pairs_slots = &link_pairs_slots;
                let link_arena_slots = &link_arena_slots;
                let id_lookup_cell = &id_lookup_cell;
                let raw_ev_slots = &raw_ev_slots;
                let eid_arena_slots = &eid_arena_slots;
                let ev_parena_slots = &ev_parena_slots;
                let raw_cl_slots = &raw_cl_slots;
                let cid_arena_slots = &cid_arena_slots;
                let cl_parena_slots = &cl_parena_slots;
                let person_arenas_cell = &person_arenas_cell;
                let eid_arenas_cell = &eid_arenas_cell;
                let cid_arenas_cell = &cid_arenas_cell;
                let ev_person_slots = &ev_person_slots;
                let cl_person_slots = &cl_person_slots;
                let shard_order_cell = &shard_order_cell;
                let prod_fast = &prod_fast;
                let prod_fallback = &prod_fallback;
                let camp_fast = &camp_fast;
                let camp_fallback = &camp_fallback;
                let categories = &categories;
                let out_dir = &out_dir;

                s.spawn(move || {
                    // ---- WAVE 1A: Parse links in parallel ----
                    let mut l_pairs = Vec::new();
                    let mut l_arena = Vec::new();
                    loop {
                        let ci = next_link.fetch_add(1, AtomicOrd::Relaxed);
                        if ci >= link_chunks.len() {
                            break;
                        }
                        let (fi, s0, e0) = link_chunks[ci];
                        let slice = &link_files[fi].as_slice()[s0..e0];
                        scan_slice::<1, _>(slice, use_avx2, |line, c, cnt, hq| {
                            if cnt != 1 {
                                return;
                            }
                            let (a, b) = if !hq {
                                (&line[..c[0]], &line[c[0] + 1..])
                            } else {
                                let (Some(a), Some(b)) =
                                    (unquote(&line[..c[0]]), unquote(&line[c[0] + 1..]))
                                else {
                                    return;
                                };
                                (a, b)
                            };
                            if a.is_empty() || b.is_empty() {
                                return;
                            }
                            let pa = pack64(a, tid as u8, &mut l_arena);
                            let pb = pack64(b, tid as u8, &mut l_arena);
                            l_pairs.push((pa, pb));
                        });
                    }
                    unsafe {
                        link_pairs_slots[tid].set(l_pairs);
                        link_arena_slots[tid].set(l_arena);
                    }
                    barrier.wait();

                    // ---- WAVE 1B: Thread 0 builds DSU while ALL threads parse events & clicks ----
                    if tid == 0 {
                        let mut all_pairs = Vec::with_capacity(ncpu);
                        let mut l_arenas = Vec::with_capacity(ncpu);
                        for t in 0..ncpu {
                            all_pairs.push(unsafe { link_pairs_slots[t].take() });
                            l_arenas.push(unsafe { link_arena_slots[t].get() }.clone());
                        }
                        let lookup = IdLookup::from_pairs(&all_pairs, &l_arenas);
                        std::mem::forget(all_pairs);
                        std::mem::forget(l_arenas);
                        let _ = id_lookup_cell.set(lookup);
                    }

                    // Parse events into L1-buffered 128 shards
                    let mut ev_writer =
                        ShardWriter::<RawEvent, DEDUP_SHARDS, 8>::new(ZERO_RAW_EVENT, est_ev_dedup);
                    let mut eid_arena = Vec::new();
                    let mut ev_parena = Vec::new();
                    let mut dc = DateCache::new();
                    let e_aid = tid as u8;
                    let ep_aid = (ncpu + tid) as u8;

                    loop {
                        let ci = next_ev_chunk.fetch_add(1, AtomicOrd::Relaxed);
                        if ci >= event_chunks.len() {
                            break;
                        }
                        let (fi, s0, e0) = event_chunks[ci];
                        let slice = &event_files[fi].as_slice()[s0..e0];
                        scan_slice::<6, _>(slice, use_avx2, |line, c, cnt, hq| {
                            if cnt != 6 {
                                return;
                            }
                            let (f1, f2, f3, f4, f5, f6, f7) = if !hq {
                                (
                                    &line[..c[0]],
                                    &line[c[0] + 1..c[1]],
                                    &line[c[1] + 1..c[2]],
                                    &line[c[2] + 1..c[3]],
                                    &line[c[3] + 1..c[4]],
                                    &line[c[4] + 1..c[5]],
                                    &line[c[5] + 1..],
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
                                    unquote(&line[..c[0]]),
                                    unquote(&line[c[0] + 1..c[1]]),
                                    unquote(&line[c[1] + 1..c[2]]),
                                    unquote(&line[c[2] + 1..c[3]]),
                                    unquote(&line[c[3] + 1..c[4]]),
                                    unquote(&line[c[4] + 1..c[5]]),
                                    unquote(&line[c[5] + 1..]),
                                )
                                else {
                                    return;
                                };
                                (r1, r2, r3, r4, r5, r6, r7)
                            };

                            if f1.is_empty() {
                                return;
                            }
                            let Some(ingest_ts) = dc.parse_ts(f4) else {
                                return;
                            };
                            let (eid, eid_h) = match pack64_try_inline(f1) {
                                Some(p) => (p, mix64(p)),
                                None => (pack64(f1, e_aid, &mut eid_arena), hash_bytes(f1)),
                            };

                            let mut ts_off = u32::MAX;
                            let mut raw_uid = 0u64;
                            let mut price_cat = 0u32;

                            if !f2.is_empty() && !f6.is_empty() {
                                let is_p = match f5.len() {
                                    4 => {
                                        let w = u32::from_le_bytes(unsafe {
                                            *(f5.as_ptr() as *const [u8; 4])
                                        });
                                        if w == u32::from_le_bytes(*b"view")
                                            || w == u32::from_le_bytes(*b"cart")
                                        {
                                            Some(false)
                                        } else {
                                            None
                                        }
                                    }
                                    8 => {
                                        let w = u64::from_le_bytes(unsafe {
                                            *(f5.as_ptr() as *const [u8; 8])
                                        });
                                        if w == u64::from_le_bytes(*b"purchase") {
                                            Some(true)
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                };
                                if let Some(is_p) = is_p {
                                    if let Some(ts) = dc.parse_ts(f3) {
                                        if ts >= win_start
                                            && ts < win_end
                                            && (ingest_ts - ts) <= MS_PER_DAY
                                        {
                                            if is_p {
                                                if let Some(pr) = parse_price(f7) {
                                                    raw_uid = pack64(f2, ep_aid, &mut ev_parena);
                                                    let cat = if let Some(pid) =
                                                        parse_prefixed_uint(f6, b"p")
                                                    {
                                                        if pid < prod_fast.len() {
                                                            unsafe { *prod_fast.get_unchecked(pid) }
                                                        } else {
                                                            0
                                                        }
                                                    } else {
                                                        prod_fallback.get(f6).unwrap_or(0)
                                                    };
                                                    ts_off = (ts - win_start) as u32;
                                                    price_cat = pr | ((cat as u32) << 24);
                                                }
                                            } else {
                                                raw_uid = pack64(f2, ep_aid, &mut ev_parena);
                                                ts_off = (ts - win_start) as u32;
                                            }
                                        }
                                    }
                                }
                            }

                            ev_writer.push(
                                (eid_h >> 57) as usize,
                                RawEvent {
                                    eid,
                                    ingest_ts,
                                    raw_uid,
                                    ts_off,
                                    price_cat,
                                },
                            );
                        });
                    }

                    // Parse clicks into L1-buffered 128 shards
                    let mut cl_writer =
                        ShardWriter::<RawClick, DEDUP_SHARDS, 8>::new(ZERO_RAW_CLICK, est_cl_dedup);
                    let mut cid_arena = Vec::new();
                    let mut cl_parena = Vec::new();
                    let c_aid = tid as u8;
                    let cp_aid = (2 * ncpu + tid) as u8;

                    loop {
                        let ci = next_cl_chunk.fetch_add(1, AtomicOrd::Relaxed);
                        if ci >= click_chunks.len() {
                            break;
                        }
                        let (fi, s0, e0) = click_chunks[ci];
                        let slice = &click_files[fi].as_slice()[s0..e0];
                        scan_slice::<3, _>(slice, use_avx2, |line, c, cnt, hq| {
                            if cnt != 3 {
                                return;
                            }
                            let (f1, f2, f3, f4) = if !hq {
                                (
                                    &line[..c[0]],
                                    &line[c[0] + 1..c[1]],
                                    &line[c[1] + 1..c[2]],
                                    &line[c[2] + 1..],
                                )
                            } else {
                                let (Some(r1), Some(r2), Some(r3), Some(r4)) = (
                                    unquote(&line[..c[0]]),
                                    unquote(&line[c[0] + 1..c[1]]),
                                    unquote(&line[c[1] + 1..c[2]]),
                                    unquote(&line[c[2] + 1..]),
                                )
                                else {
                                    return;
                                };
                                (r1, r2, r3, r4)
                            };
                            if f1.is_empty() || f2.is_empty() {
                                return;
                            }
                            let Some(ts) = dc.parse_ts(f4) else {
                                return;
                            };
                            if ts < click_win_start || ts >= win_end {
                                return;
                            }
                            let camp_idx = if let Some(cid) = parse_prefixed_uint(f3, b"c_") {
                                if cid < camp_fast.len() {
                                    unsafe { *camp_fast.get_unchecked(cid) }
                                } else {
                                    0
                                }
                            } else {
                                camp_fallback.get(f3).unwrap_or(0)
                            };
                            if camp_idx == 0 {
                                return;
                            }
                            let (cid, cid_h) = match pack64_try_inline(f1) {
                                Some(p) => (p, mix64(p)),
                                None => (pack64(f1, c_aid, &mut cid_arena), hash_bytes(f1)),
                            };
                            let raw_uid = pack64(f2, cp_aid, &mut cl_parena);
                            cl_writer.push(
                                (cid_h >> 57) as usize,
                                RawClick {
                                    cid,
                                    raw_uid,
                                    ts_off: (ts - click_win_start) as u32,
                                    campaign_idx: camp_idx,
                                },
                            );
                        });
                    }

                    unsafe {
                        raw_ev_slots[tid].set(ev_writer.finish());
                        eid_arena_slots[tid].set(eid_arena);
                        ev_parena_slots[tid].set(ev_parena);
                        raw_cl_slots[tid].set(cl_writer.finish());
                        cid_arena_slots[tid].set(cid_arena);
                        cl_parena_slots[tid].set(cl_parena);
                    }
                    barrier.wait();

                    if tid == 0 {
                        let mut p_arenas = Vec::with_capacity(ncpu * 3);
                        for t in 0..ncpu {
                            p_arenas.push(unsafe { link_arena_slots[t].take() });
                        }
                        for t in 0..ncpu {
                            p_arenas.push(unsafe { ev_parena_slots[t].take() });
                        }
                        for t in 0..ncpu {
                            p_arenas.push(unsafe { cl_parena_slots[t].take() });
                        }
                        let mut e_arenas = Vec::with_capacity(ncpu);
                        let mut c_arenas = Vec::with_capacity(ncpu);
                        for t in 0..ncpu {
                            e_arenas.push(unsafe { eid_arena_slots[t].take() });
                            c_arenas.push(unsafe { cid_arena_slots[t].take() });
                        }
                        let _ = person_arenas_cell.set(p_arenas);
                        let _ = eid_arenas_cell.set(e_arenas);
                        let _ = cid_arenas_cell.set(c_arenas);
                    }
                    barrier.wait();

                    // ---- WAVE 2: Dedup events & clicks + resolve IDs + L1-buffered person scatter ----
                    let id_lookup = id_lookup_cell.get().unwrap();
                    let person_arenas = person_arenas_cell.get().unwrap();
                    let eid_arenas = eid_arenas_cell.get().unwrap();
                    let cid_arenas = cid_arenas_cell.get().unwrap();

                    let mut ev_p_writer = ShardWriter::<ValidEvent, PERSON_SHARDS, 16>::new(
                        ZERO_VALID_EVENT,
                        est_ev_person,
                    );
                    let mut tab32: Vec<u32> = Vec::new();
                    let mut tab64: Vec<u64> = Vec::new();
                    let mut unique_ev: Vec<RawEvent> = Vec::new();

                    loop {
                        let sh = next_dedup_ev.fetch_add(1, AtomicOrd::Relaxed);
                        if sh >= DEDUP_SHARDS {
                            break;
                        }
                        let mut n = 0usize;
                        for t in 0..ncpu {
                            n += unsafe { raw_ev_slots[t].get() }[sh].len();
                        }
                        if n == 0 {
                            continue;
                        }
                        let cap = ((n * 3) / 2 + 16).next_power_of_two().max(64);
                        let mask = cap - 1;
                        unique_ev.clear();
                        if unique_ev.capacity() < n {
                            unique_ev.reserve(n - unique_ev.capacity());
                        }

                        if n < 1_000_000 {
                            tab32.clear();
                            tab32.resize(cap, 0);
                            for t in 0..ncpu {
                                let slice = &unsafe { raw_ev_slots[t].get() }[sh];
                                for &ev in slice {
                                    let h = hash_pack64(ev.eid, eid_arenas);
                                    let tag = ((h >> 24) as u32) & 0xFFF;
                                    let mut pos = (h as usize) & mask;
                                    loop {
                                        let e = unsafe { *tab32.get_unchecked(pos) };
                                        if e == 0 {
                                            let idx = unique_ev.len() as u32;
                                            unique_ev.push(ev);
                                            unsafe {
                                                *tab32.get_unchecked_mut(pos) =
                                                    (tag << 20) | (idx + 1);
                                            }
                                            break;
                                        }
                                        if (e >> 20) == tag {
                                            let idx = ((e & 0xF_FFFF) - 1) as usize;
                                            let ex = unsafe { unique_ev.get_unchecked_mut(idx) };
                                            if eq_pack64(ex.eid, ev.eid, eid_arenas) {
                                                if ev.ingest_ts < ex.ingest_ts {
                                                    *ex = ev;
                                                }
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }
                        } else {
                            tab64.clear();
                            tab64.resize(cap, 0);
                            for t in 0..ncpu {
                                let slice = &unsafe { raw_ev_slots[t].get() }[sh];
                                for &ev in slice {
                                    let h = hash_pack64(ev.eid, eid_arenas);
                                    let tag = (h >> 24) as u32;
                                    let mut pos = (h as usize) & mask;
                                    loop {
                                        let e = tab64[pos];
                                        if e == 0 {
                                            let idx = unique_ev.len() as u32;
                                            unique_ev.push(ev);
                                            tab64[pos] =
                                                ((tag as u64) << 32) | ((idx as u64) + 1);
                                            break;
                                        }
                                        if (e >> 32) as u32 == tag {
                                            let idx = ((e & 0xFFFF_FFFF) - 1) as usize;
                                            let ex = &mut unique_ev[idx];
                                            if eq_pack64(ex.eid, ev.eid, eid_arenas) {
                                                if ev.ingest_ts < ex.ingest_ts {
                                                    *ex = ev;
                                                }
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }
                        }

                        for ev in &unique_ev {
                            if ev.ts_off != u32::MAX {
                                let (person, ph) =
                                    id_lookup.resolve_pack64(ev.raw_uid, person_arenas);
                                ev_p_writer.push(
                                    (ph >> 58) as usize,
                                    ValidEvent {
                                        person,
                                        ts_off: ev.ts_off,
                                        price_cat: ev.price_cat,
                                    },
                                );
                            }
                        }
                    }

                    let mut cl_p_writer = ShardWriter::<ValidClick, PERSON_SHARDS, 8>::new(
                        ZERO_VALID_CLICK,
                        est_cl_person,
                    );
                    let mut unique_cl: Vec<RawClick> = Vec::new();

                    loop {
                        let sh = next_dedup_cl.fetch_add(1, AtomicOrd::Relaxed);
                        if sh >= DEDUP_SHARDS {
                            break;
                        }
                        let mut n = 0usize;
                        for t in 0..ncpu {
                            n += unsafe { raw_cl_slots[t].get() }[sh].len();
                        }
                        if n == 0 {
                            continue;
                        }
                        let cap = ((n * 3) / 2 + 16).next_power_of_two().max(64);
                        let mask = cap - 1;
                        unique_cl.clear();
                        if unique_cl.capacity() < n {
                            unique_cl.reserve(n - unique_cl.capacity());
                        }

                        if n < 1_000_000 {
                            tab32.clear();
                            tab32.resize(cap, 0);
                            for t in 0..ncpu {
                                let slice = &unsafe { raw_cl_slots[t].get() }[sh];
                                for &cl in slice {
                                    let h = hash_pack64(cl.cid, cid_arenas);
                                    let tag = ((h >> 24) as u32) & 0xFFF;
                                    let mut pos = (h as usize) & mask;
                                    loop {
                                        let e = unsafe { *tab32.get_unchecked(pos) };
                                        if e == 0 {
                                            let idx = unique_cl.len() as u32;
                                            unique_cl.push(cl);
                                            unsafe {
                                                *tab32.get_unchecked_mut(pos) =
                                                    (tag << 20) | (idx + 1);
                                            }
                                            break;
                                        }
                                        if (e >> 20) == tag {
                                            let idx = ((e & 0xF_FFFF) - 1) as usize;
                                            let ex = unsafe { unique_cl.get_unchecked(idx) };
                                            if eq_pack64(ex.cid, cl.cid, cid_arenas) {
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }
                        } else {
                            tab64.clear();
                            tab64.resize(cap, 0);
                            for t in 0..ncpu {
                                let slice = &unsafe { raw_cl_slots[t].get() }[sh];
                                for &cl in slice {
                                    let h = hash_pack64(cl.cid, cid_arenas);
                                    let tag = (h >> 24) as u32;
                                    let mut pos = (h as usize) & mask;
                                    loop {
                                        let e = tab64[pos];
                                        if e == 0 {
                                            let idx = unique_cl.len() as u32;
                                            unique_cl.push(cl);
                                            tab64[pos] =
                                                ((tag as u64) << 32) | ((idx as u64) + 1);
                                            break;
                                        }
                                        if (e >> 32) as u32 == tag {
                                            let idx = ((e & 0xFFFF_FFFF) - 1) as usize;
                                            if eq_pack64(unique_cl[idx].cid, cl.cid, cid_arenas) {
                                                break;
                                            }
                                        }
                                        pos = (pos + 1) & mask;
                                    }
                                }
                            }
                        }

                        for cl in &unique_cl {
                            let (person, ph) =
                                id_lookup.resolve_pack64(cl.raw_uid, person_arenas);
                            cl_p_writer.push(
                                (ph >> 58) as usize,
                                ValidClick {
                                    person,
                                    cid: cl.cid,
                                    ts_off: cl.ts_off,
                                    campaign_idx: cl.campaign_idx,
                                },
                            );
                        }
                    }

                    std::mem::forget(unique_ev);
                    std::mem::forget(unique_cl);
                    std::mem::forget(tab32);
                    std::mem::forget(tab64);

                    unsafe {
                        ev_person_slots[tid].set(ev_p_writer.finish());
                        cl_person_slots[tid].set(cl_p_writer.finish());
                    }
                    barrier.wait();

                    if tid == 0 {
                        let mut order: Vec<(usize, usize)> = (0..PERSON_SHARDS)
                            .map(|ps| {
                                let mut cnt = 0usize;
                                for t in 0..ncpu {
                                    cnt += unsafe { ev_person_slots[t].get() }[ps].len();
                                }
                                (ps, cnt)
                            })
                            .collect();
                        order.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                        let _ = shard_order_cell.set(order);
                    }
                    barrier.wait();

                    // ---- WAVE 3: Group-by-person, sessionize, stream directly to Parquet ----
                    let shard_order = shard_order_cell.get().unwrap();
                    let mut pid_tab: Vec<u32> = Vec::new();
                    let mut person_keys: Vec<u64> = Vec::new();
                    let mut ev_counts: Vec<u32> = Vec::new();
                    let mut ev_pids: Vec<u32> = Vec::new();
                    let mut ev_offsets: Vec<u32> = Vec::new();
                    let mut ev_cursor: Vec<u32> = Vec::new();
                    let mut grouped_ev: Vec<u64> = Vec::new();

                    let mut cl_counts: Vec<u32> = Vec::new();
                    let mut matched_cl: Vec<(u32, ClickPacked)> = Vec::new();
                    let mut cl_offsets: Vec<u32> = Vec::new();
                    let mut cl_cursor: Vec<u32> = Vec::new();
                    let mut grouped_cl: Vec<ClickPacked> = Vec::new();

                    let mut camp_counts = vec![0u32; num_campaigns];
                    let mut active_camps: Vec<u16> = Vec::new();
                    let mut cat_rev_arr = vec![0i64; num_categories];
                    let mut touched_cats: Vec<u8> = Vec::new();
                    let mut camp_flat = vec![CampaignDayAgg::default(); flat_camp_len];

                    let mut out_rows = 0usize;
                    let mut uid_plain: Vec<u8> = Vec::with_capacity(4 << 20);
                    let mut day_plain: Vec<u8> = Vec::with_capacity(1 << 20);
                    let mut sess_plain: Vec<u8> = Vec::with_capacity(1 << 20);
                    let mut ev_plain: Vec<u8> = Vec::with_capacity(1 << 20);
                    let mut rev_plain: Vec<u8> = Vec::with_capacity(2 << 20);
                    let mut cat_def_bits: Vec<u8> = Vec::with_capacity(1 << 17);
                    let mut cat_plain: Vec<u8> = Vec::with_capacity(1 << 20);
                    let mut uid_scratch = [0u8; 10];

                    loop {
                        let ord_idx = next_person_shard.fetch_add(1, AtomicOrd::Relaxed);
                        if ord_idx >= PERSON_SHARDS {
                            break;
                        }
                        let (ps, ev_total) = shard_order[ord_idx];
                        if ev_total == 0 {
                            continue;
                        }

                        let est_persons = (ev_total / 4).max(64);
                        let cap = (est_persons * 2).next_power_of_two().max(64);
                        pid_tab.clear();
                        pid_tab.resize(cap, 0);
                        let mut mask = cap - 1;
                        person_keys.clear();
                        ev_counts.clear();
                        ev_pids.clear();
                        if ev_pids.capacity() < ev_total {
                            ev_pids.reserve(ev_total - ev_pids.capacity());
                        }

                        for t in 0..ncpu {
                            let slice = &unsafe { ev_person_slots[t].get() }[ps];
                            for &ev in slice {
                                if (person_keys.len() + 1) * 10 >= pid_tab.len() * 7 {
                                    let nc = pid_tab.len() * 2;
                                    pid_tab.clear();
                                    pid_tab.resize(nc, 0);
                                    mask = nc - 1;
                                    for (i, &pk) in person_keys.iter().enumerate() {
                                        let mut p =
                                            (hash_pack64(pk, person_arenas) as usize) & mask;
                                        while pid_tab[p] != 0 {
                                            p = (p + 1) & mask;
                                        }
                                        pid_tab[p] = (i as u32) + 1;
                                    }
                                }
                                let h = hash_pack64(ev.person, person_arenas);
                                let mut pos = (h as usize) & mask;
                                let pid = loop {
                                    let e = unsafe { *pid_tab.get_unchecked(pos) };
                                    if e == 0 {
                                        let id = person_keys.len() as u32;
                                        person_keys.push(ev.person);
                                        ev_counts.push(1);
                                        unsafe {
                                            *pid_tab.get_unchecked_mut(pos) = id + 1;
                                        }
                                        break id;
                                    }
                                    let id = (e - 1) as usize;
                                    if eq_pack64(
                                        unsafe { *person_keys.get_unchecked(id) },
                                        ev.person,
                                        person_arenas,
                                    ) {
                                        unsafe {
                                            *ev_counts.get_unchecked_mut(id) += 1;
                                        }
                                        break id as u32;
                                    }
                                    pos = (pos + 1) & mask;
                                };
                                ev_pids.push(pid);
                            }
                        }

                        let num_persons = person_keys.len();
                        ev_offsets.clear();
                        ev_offsets.reserve(num_persons + 1);
                        let mut sum = 0u32;
                        for &c in &ev_counts {
                            ev_offsets.push(sum);
                            sum += c;
                        }
                        ev_offsets.push(sum);

                        ev_cursor.clear();
                        ev_cursor.extend_from_slice(&ev_offsets[..num_persons]);
                        grouped_ev.resize(ev_total, 0);

                        let mut ei = 0usize;
                        for t in 0..ncpu {
                            let slice = &unsafe { ev_person_slots[t].get() }[ps];
                            for &ev in slice {
                                let pid = unsafe { *ev_pids.get_unchecked(ei) } as usize;
                                ei += 1;
                                let dst = unsafe {
                                    let cur = ev_cursor.get_unchecked_mut(pid);
                                    let d = *cur as usize;
                                    *cur += 1;
                                    d
                                };
                                unsafe {
                                    *grouped_ev.get_unchecked_mut(dst) =
                                        ((ev.ts_off as u64) << 32) | (ev.price_cat as u64);
                                }
                            }
                        }

                        cl_counts.clear();
                        cl_counts.resize(num_persons, 0);
                        matched_cl.clear();
                        for t in 0..ncpu {
                            let slice = &unsafe { cl_person_slots[t].get() }[ps];
                            for &cl in slice {
                                let h = hash_pack64(cl.person, person_arenas);
                                let mut pos = (h as usize) & mask;
                                loop {
                                    let e = unsafe { *pid_tab.get_unchecked(pos) };
                                    if e == 0 {
                                        break;
                                    }
                                    let id = (e - 1) as usize;
                                    if eq_pack64(
                                        unsafe { *person_keys.get_unchecked(id) },
                                        cl.person,
                                        person_arenas,
                                    ) {
                                        unsafe {
                                            *cl_counts.get_unchecked_mut(id) += 1;
                                        }
                                        matched_cl.push((
                                            id as u32,
                                            ClickPacked {
                                                ts_off: cl.ts_off,
                                                campaign_idx: cl.campaign_idx,
                                                cid: cl.cid,
                                            },
                                        ));
                                        break;
                                    }
                                    pos = (pos + 1) & mask;
                                }
                            }
                        }

                        cl_offsets.clear();
                        cl_offsets.reserve(num_persons + 1);
                        let mut csum = 0u32;
                        for &c in &cl_counts {
                            cl_offsets.push(csum);
                            csum += c;
                        }
                        cl_offsets.push(csum);
                        cl_cursor.clear();
                        cl_cursor.extend_from_slice(&cl_offsets[..num_persons]);
                        grouped_cl.resize(
                            matched_cl.len(),
                            ClickPacked {
                                ts_off: 0,
                                campaign_idx: 0,
                                cid: 0,
                            },
                        );
                        for &(pid, cp) in &matched_cl {
                            let dst = unsafe {
                                let cur = cl_cursor.get_unchecked_mut(pid as usize);
                                let d = *cur as usize;
                                *cur += 1;
                                d
                            };
                            unsafe { *grouped_cl.get_unchecked_mut(dst) = cp };
                        }

                        for pid in 0..num_persons {
                            let es = unsafe { *ev_offsets.get_unchecked(pid) } as usize;
                            let ee = unsafe { *ev_offsets.get_unchecked(pid + 1) } as usize;
                            let p_events = &mut grouped_ev[es..ee];
                            if p_events.len() > 1 {
                                p_events.sort_unstable();
                            }

                            let cs = unsafe { *cl_offsets.get_unchecked(pid) } as usize;
                            let ce = unsafe { *cl_offsets.get_unchecked(pid + 1) } as usize;
                            let p_clicks = &mut grouped_cl[cs..ce];
                            if p_clicks.len() > 1 {
                                p_clicks.sort_unstable_by(|a, b| {
                                    a.ts_off
                                        .cmp(&b.ts_off)
                                        .then_with(|| cmp_pack64(b.cid, a.cid, cid_arenas))
                                });
                            }

                            let uid_bytes =
                                pack64_bytes(person_keys[pid], &mut uid_scratch, person_arenas);

                            for &c in &active_camps {
                                camp_counts[c as usize] = 0;
                            }
                            active_camps.clear();

                            let mut left = 0usize;
                            let mut right = 0usize;
                            let mut cur_day = i32::MIN;
                            let mut day_sessions = 0i32;
                            let mut day_events = 0i32;
                            let mut day_rev = 0i64;

                            let mut flush_day = |day: i32,
                                                 d_sess: i32,
                                                 d_ev: i32,
                                                 d_rev: i64,
                                                 touched: &mut Vec<u8>,
                                                 crev: &mut [i64]| {
                                push_plain_str(&mut uid_plain, uid_bytes);
                                day_plain.extend_from_slice(&day.to_le_bytes());
                                sess_plain.extend_from_slice(&d_sess.to_le_bytes());
                                ev_plain.extend_from_slice(&d_ev.to_le_bytes());
                                rev_plain.extend_from_slice(&d_rev.to_le_bytes());

                                let bit_idx = out_rows & 7;
                                if bit_idx == 0 {
                                    cat_def_bits.push(0);
                                }
                                if !touched.is_empty() {
                                    *cat_def_bits.last_mut().unwrap() |= 1u8 << bit_idx;
                                    let mut best_c = touched[0];
                                    let mut best_r = crev[best_c as usize];
                                    crev[best_c as usize] = 0;
                                    for &c in &touched[1..] {
                                        let r = crev[c as usize];
                                        crev[c as usize] = 0;
                                        if r > best_r
                                            || (r == best_r
                                                && categories[c as usize]
                                                    < categories[best_c as usize])
                                        {
                                            best_r = r;
                                            best_c = c;
                                        }
                                    }
                                    touched.clear();
                                    push_plain_str(
                                        &mut cat_plain,
                                        categories[best_c as usize].as_bytes(),
                                    );
                                }
                                out_rows += 1;
                            };

                            let mut k = 0usize;
                            let ev_n = p_events.len();
                            while k < ev_n {
                                let sess_start_off =
                                    (unsafe { *p_events.get_unchecked(k) } >> 32) as u32;
                                let day_idx =
                                    ((win_start_rem + (sess_start_off as u64)) / 86_400_000) as i32;
                                let day = min_day + day_idx;

                                if day != cur_day {
                                    if cur_day != i32::MIN {
                                        flush_day(
                                            cur_day,
                                            day_sessions,
                                            day_events,
                                            day_rev,
                                            &mut touched_cats,
                                            &mut cat_rev_arr,
                                        );
                                    }
                                    cur_day = day;
                                    day_sessions = 0;
                                    day_events = 0;
                                    day_rev = 0;
                                }

                                let mut sess_events = 0i32;
                                let mut sess_rev = 0i64;
                                let mut has_purchase = false;
                                let mut prev_off = sess_start_off;

                                while k < ev_n {
                                    let packed = unsafe { *p_events.get_unchecked(k) };
                                    let ev_off = (packed >> 32) as u32;
                                    if ev_off - prev_off > SESSION_GAP_MS {
                                        break;
                                    }
                                    prev_off = ev_off;
                                    sess_events += 1;
                                    let pc = packed as u32;
                                    let price = (pc & 0x00FF_FFFF) as i64;
                                    if price > 0 {
                                        has_purchase = true;
                                        sess_rev += price;
                                        let cat = (pc >> 24) as usize;
                                        if cat_rev_arr[cat] == 0 {
                                            touched_cats.push(cat as u8);
                                        }
                                        cat_rev_arr[cat] += price;
                                    }
                                    k += 1;
                                }
                                day_sessions += 1;
                                day_events += sess_events;
                                day_rev += sess_rev;

                                let sess_cl_off = sess_start_off + ASSIST_WINDOW_U32;
                                while right < p_clicks.len()
                                    && unsafe { p_clicks.get_unchecked(right).ts_off }
                                        <= sess_cl_off
                                {
                                    let c = unsafe { p_clicks.get_unchecked(right).campaign_idx };
                                    if camp_counts[c as usize] == 0 {
                                        active_camps.push(c);
                                    }
                                    camp_counts[c as usize] += 1;
                                    right += 1;
                                }
                                while left < right
                                    && unsafe { p_clicks.get_unchecked(left).ts_off }
                                        < sess_start_off
                                {
                                    let c = unsafe { p_clicks.get_unchecked(left).campaign_idx };
                                    camp_counts[c as usize] -= 1;
                                    left += 1;
                                }

                                let credited_camp = if left < right {
                                    unsafe { p_clicks.get_unchecked(right - 1).campaign_idx }
                                } else {
                                    0
                                };

                                let day_off = (day_idx as usize) * num_campaigns;
                                let cr = unsafe {
                                    camp_flat.get_unchecked_mut(day_off + credited_camp as usize)
                                };
                                cr.sessions += 1;
                                if has_purchase {
                                    cr.purchasing_sessions += 1;
                                    cr.revenue_cents += sess_rev;
                                }
                                if left < right {
                                    active_camps.retain(|&c| camp_counts[c as usize] > 0);
                                    for &c in &active_camps {
                                        unsafe {
                                            camp_flat
                                                .get_unchecked_mut(day_off + c as usize)
                                                .assisted_sessions += 1;
                                        }
                                    }
                                } else {
                                    active_camps.clear();
                                }
                            }

                            if cur_day != i32::MIN {
                                flush_day(
                                    cur_day,
                                    day_sessions,
                                    day_events,
                                    day_rev,
                                    &mut touched_cats,
                                    &mut cat_rev_arr,
                                );
                            }
                        }
                    }

                    if out_rows > 0 || tid == 0 {
                        let cols = [
                            ColSpec {
                                name: "user_id",
                                kind: ColKind::Utf8Req,
                                plain: &uid_plain,
                                def_bits: None,
                            },
                            ColSpec {
                                name: "day",
                                kind: ColKind::Date32,
                                plain: &day_plain,
                                def_bits: None,
                            },
                            ColSpec {
                                name: "sessions",
                                kind: ColKind::Int32,
                                plain: &sess_plain,
                                def_bits: None,
                            },
                            ColSpec {
                                name: "events",
                                kind: ColKind::Int32,
                                plain: &ev_plain,
                                def_bits: None,
                            },
                            ColSpec {
                                name: "revenue_cents",
                                kind: ColKind::Int64,
                                plain: &rev_plain,
                                def_bits: None,
                            },
                            ColSpec {
                                name: "top_category",
                                kind: ColKind::Utf8Opt,
                                plain: &cat_plain,
                                def_bits: Some(&cat_def_bits),
                            },
                        ];
                        write_parquet_file(
                            &out_dir.join(format!("user_daily/part-{tid}.parquet")),
                            out_rows,
                            &cols,
                        );
                    }

                    std::mem::forget(uid_plain);
                    std::mem::forget(day_plain);
                    std::mem::forget(sess_plain);
                    std::mem::forget(ev_plain);
                    std::mem::forget(rev_plain);
                    std::mem::forget(cat_def_bits);
                    std::mem::forget(cat_plain);
                    std::mem::forget(pid_tab);
                    std::mem::forget(person_keys);
                    std::mem::forget(grouped_ev);
                    std::mem::forget(grouped_cl);

                    camp_flat
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge campaign_daily and write campaign_daily/part-0.parquet
    let mut merged = vec![CampaignDayAgg::default(); flat_camp_len];
    for flat in &camp_results {
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

    let mut camp_rows = 0usize;
    let mut cid_plain = Vec::with_capacity(1 << 18);
    let mut ch_plain = Vec::with_capacity(1 << 18);
    let mut cday_plain = Vec::with_capacity(1 << 17);
    let mut csess_plain = Vec::with_capacity(1 << 17);
    let mut cpsess_plain = Vec::with_capacity(1 << 17);
    let mut crev_plain = Vec::with_capacity(1 << 18);
    let mut cass_plain = Vec::with_capacity(1 << 17);

    for d in 0..num_days {
        let day = min_day + (d as i32);
        let base = d * num_campaigns;
        for c in 0..num_campaigns {
            let v = merged[base + c];
            if v.sessions != 0 || v.assisted_sessions != 0 {
                push_plain_str(&mut cid_plain, camp_names[c].as_bytes());
                push_plain_str(&mut ch_plain, camp_channels[c].as_bytes());
                cday_plain.extend_from_slice(&day.to_le_bytes());
                csess_plain.extend_from_slice(&v.sessions.to_le_bytes());
                cpsess_plain.extend_from_slice(&v.purchasing_sessions.to_le_bytes());
                crev_plain.extend_from_slice(&v.revenue_cents.to_le_bytes());
                cass_plain.extend_from_slice(&v.assisted_sessions.to_le_bytes());
                camp_rows += 1;
            }
        }
    }

    let camp_cols = [
        ColSpec {
            name: "campaign_id",
            kind: ColKind::Utf8Req,
            plain: &cid_plain,
            def_bits: None,
        },
        ColSpec {
            name: "channel",
            kind: ColKind::Utf8Req,
            plain: &ch_plain,
            def_bits: None,
        },
        ColSpec {
            name: "day",
            kind: ColKind::Date32,
            plain: &cday_plain,
            def_bits: None,
        },
        ColSpec {
            name: "sessions",
            kind: ColKind::Int32,
            plain: &csess_plain,
            def_bits: None,
        },
        ColSpec {
            name: "purchasing_sessions",
            kind: ColKind::Int32,
            plain: &cpsess_plain,
            def_bits: None,
        },
        ColSpec {
            name: "revenue_cents",
            kind: ColKind::Int64,
            plain: &crev_plain,
            def_bits: None,
        },
        ColSpec {
            name: "assisted_sessions",
            kind: ColKind::Int32,
            plain: &cass_plain,
            def_bits: None,
        },
    ];
    write_parquet_file(
        &out_dir.join("campaign_daily/part-0.parquet"),
        camp_rows,
        &camp_cols,
    );

    std::process::exit(0);
}