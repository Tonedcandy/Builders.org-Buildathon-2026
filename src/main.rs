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
// Shard counts: enough that a shard's working set (records, tables, scratch)
// stays cache-sized with 8 threads at once on ~60M-event inputs.
const DEDUP_BITS: u32 = 9;
const DEDUP_SHARDS: usize = 1 << DEDUP_BITS;
const PERSON_BITS: u32 = 8;
const PERSON_SHARDS: usize = 1 << PERSON_BITS;
const CHUNK_SIZE: usize = 2 * 1024 * 1024;

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
        // Reserve the file's pages plus one spare zero page, then map the file
        // over the start of the reservation. Every byte from any position in the
        // file to 4 KiB past its end is then readable, so field parsers can load
        // 16 bytes at a time without bounds checks (see pack64_try_inline).
        let page = 4096usize;
        let rem = len % page;
        if rem != 0 && rem <= page - 16 {
            // The last page already has >= 16 zero bytes past the end of the
            // file, enough for every over-read: one plain mapping suffices.
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
            return Self {
                ptr: ptr as *const u8,
                len,
            };
        }
        let total = len.div_ceil(page) * page + page;
        let ptr = unsafe {
            let base = libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert!(base != libc::MAP_FAILED, "mmap reserve failed");
            libc::mmap(
                base,
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_FIXED,
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

/// Packs an ID of up to 10 chars from [0-9A-Za-z_] inline; None otherwise.
/// On x86-64 this reads 16 bytes from the start of `b`, so `b` must lie inside
/// a MappedFile (which keeps a readable page past the end of the file). Every
/// caller passes a CSV field from a mapped file.
#[inline(always)]
fn pack64_try_inline(b: &[u8]) -> Option<u64> {
    let len = b.len();
    if len > 10 {
        return None;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
    unsafe {
        return pack64_inline_simd(b.as_ptr(), len);
    }
    #[allow(unreachable_code)]
    pack64_inline_scalar(b)
}

/// Branch-free inline packing of `len` (<= 10) bytes: classify 16 bytes at once,
/// map each char to its 6-bit code, then combine codes pairwise with multiply-adds
/// (pairs -> 12 bits, quads -> 24 bits) and assemble the three 24-bit groups.
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[inline(always)]
unsafe fn pack64_inline_simd(p: *const u8, len: usize) -> Option<u64> {
    use std::arch::x86_64::*;
    let v = _mm_loadu_si128(p as *const __m128i);
    let iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let in_len = _mm_cmplt_epi8(iota, _mm_set1_epi8(len as i8));
    // Signed byte compares: bytes >= 0x80 are negative and fall in no class.
    let range = |lo: u8, hi: u8| {
        _mm_and_si128(
            _mm_cmpgt_epi8(v, _mm_set1_epi8(lo as i8 - 1)),
            _mm_cmplt_epi8(v, _mm_set1_epi8(hi as i8 + 1)),
        )
    };
    let dig = range(b'0', b'9');
    let up = range(b'A', b'Z');
    let low = range(b'a', b'z');
    let us = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'_' as i8));
    let valid = _mm_or_si128(_mm_or_si128(dig, up), _mm_or_si128(low, us));
    if _mm_movemask_epi8(_mm_andnot_si128(valid, in_len)) != 0 {
        return None;
    }
    // code = byte - offset: '0'->1, 'A'->11, '_'->37, 'a'->38 (as ENCODE_LUT).
    let off = _mm_or_si128(
        _mm_or_si128(
            _mm_and_si128(dig, _mm_set1_epi8(47)),
            _mm_and_si128(up, _mm_set1_epi8(54)),
        ),
        _mm_or_si128(
            _mm_and_si128(low, _mm_set1_epi8(59)),
            _mm_and_si128(us, _mm_set1_epi8(58)),
        ),
    );
    let code = _mm_and_si128(_mm_sub_epi8(v, off), in_len);
    // (c0*64 + c1) per 16-bit lane, then (w0*4096 + w1) per 32-bit lane.
    let w = _mm_maddubs_epi16(code, _mm_set1_epi16(0x0140));
    let d = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_1000));
    let lo = _mm_cvtsi128_si64(d) as u64;
    let d2 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(d)) as u32 as u64;
    let d0 = lo & 0xFFFF_FFFF;
    let d1 = lo >> 32;
    Some((d0 << 39) | (d1 << 15) | (d2 >> 9))
}

#[inline(always)]
fn pack64_inline_scalar(b: &[u8]) -> Option<u64> {
    let len = b.len();
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

/// Appends a short byte string without calling memcpy (musl's memcpy is slow
/// for small copies): two possibly-overlapping fixed-size copies cover any
/// length from 4 to 64 bytes. Reads and writes stay inside `src` and the newly
/// reserved tail of `dst`.
#[inline(always)]
fn append_small(dst: &mut Vec<u8>, src: &[u8]) {
    let n = src.len();
    if n > 64 {
        dst.extend_from_slice(src);
        return;
    }
    dst.reserve(n);
    unsafe {
        let len0 = dst.len();
        let d = dst.as_mut_ptr().add(len0);
        let s = src.as_ptr();
        if n >= 32 {
            let a = std::ptr::read_unaligned(s as *const [u8; 32]);
            let b = std::ptr::read_unaligned(s.add(n - 32) as *const [u8; 32]);
            std::ptr::write_unaligned(d as *mut [u8; 32], a);
            std::ptr::write_unaligned(d.add(n - 32) as *mut [u8; 32], b);
        } else if n >= 16 {
            let a = std::ptr::read_unaligned(s as *const u128);
            let b = std::ptr::read_unaligned(s.add(n - 16) as *const u128);
            std::ptr::write_unaligned(d as *mut u128, a);
            std::ptr::write_unaligned(d.add(n - 16) as *mut u128, b);
        } else if n >= 8 {
            let a = std::ptr::read_unaligned(s as *const u64);
            let b = std::ptr::read_unaligned(s.add(n - 8) as *const u64);
            std::ptr::write_unaligned(d as *mut u64, a);
            std::ptr::write_unaligned(d.add(n - 8) as *mut u64, b);
        } else if n >= 4 {
            let a = std::ptr::read_unaligned(s as *const u32);
            let b = std::ptr::read_unaligned(s.add(n - 4) as *const u32);
            std::ptr::write_unaligned(d as *mut u32, a);
            std::ptr::write_unaligned(d.add(n - 4) as *mut u32, b);
        } else {
            let mut i = 0;
            while i < n {
                *d.add(i) = *s.add(i);
                i += 1;
            }
        }
        dst.set_len(len0 + n);
    }
}

#[inline(always)]
fn pack64(b: &[u8], arena_id: u8, arena: &mut Vec<u8>) -> u64 {
    if let Some(v) = pack64_try_inline(b) {
        return v;
    }
    let off = arena.len() as u64;
    append_small(arena, b);
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

/// Byte-string equality without memcmp (musl's memcmp is a byte loop): 16, 8
/// or 4 bytes at a time, finishing with an overlapping word. Reads stay inside
/// both slices.
#[inline(always)]
fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() {
        return false;
    }
    unsafe {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        if n >= 16 {
            let mut i = 0;
            while i + 16 <= n {
                if std::ptr::read_unaligned(pa.add(i) as *const u128)
                    != std::ptr::read_unaligned(pb.add(i) as *const u128)
                {
                    return false;
                }
                i += 16;
            }
            if i < n {
                let t = n - 16;
                return std::ptr::read_unaligned(pa.add(t) as *const u128)
                    == std::ptr::read_unaligned(pb.add(t) as *const u128);
            }
            true
        } else if n >= 8 {
            std::ptr::read_unaligned(pa as *const u64) == std::ptr::read_unaligned(pb as *const u64)
                && std::ptr::read_unaligned(pa.add(n - 8) as *const u64)
                    == std::ptr::read_unaligned(pb.add(n - 8) as *const u64)
        } else if n >= 4 {
            std::ptr::read_unaligned(pa as *const u32) == std::ptr::read_unaligned(pb as *const u32)
                && std::ptr::read_unaligned(pa.add(n - 4) as *const u32)
                    == std::ptr::read_unaligned(pb.add(n - 4) as *const u32)
        } else {
            let mut i = 0;
            while i < n {
                if *pa.add(i) != *pb.add(i) {
                    return false;
                }
                i += 1;
            }
            true
        }
    }
}

#[inline(always)]
fn eq_pack64(a: u64, b: u64, arenas: &[Vec<u8>]) -> bool {
    // Identical values are always the same string (inline, or the same arena
    // bytes), which skips the byte compare for linked persons' shared root.
    if a == b {
        return true;
    }
    if ((a | b) >> 63) == 0 {
        a == b
    } else if (a >> 63) != (b >> 63) {
        false
    } else {
        bytes_eq(overflow_bytes(a, arenas), overflow_bytes(b, arenas))
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

/// A shard's records: a slice into its writer's slab, or (rarely) into a merged
/// copy when the shard overflowed its slab region. Never freed (exit(0) reclaims).
#[derive(Clone, Copy)]
struct ShardSlice<T> {
    ptr: *const T,
    len: usize,
}
unsafe impl<T: Send> Send for ShardSlice<T> {}
unsafe impl<T: Sync> Sync for ShardSlice<T> {}
impl<T> std::ops::Deref for ShardSlice<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Target block size for ShardWriter output.
const BLOCK_BYTES: usize = 64 * 1024;

thread_local! {
    /// This thread's free blocks (BLOCK_BYTES each, 64-byte aligned): blocks of
    /// shards it has finished reading. Its writers take these before mapping
    /// fresh memory, so later waves reuse earlier waves' memory instead of
    /// having the kernel zero new pages.
    static BLOCK_POOL: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Returns the blocks of fully consumed shard segments to this thread's pool.
/// Every segment starts at its block's base.
fn recycle_blocks<'a, T: 'a>(segs: impl Iterator<Item = &'a [T]>) {
    BLOCK_POOL.with(|p| {
        let mut p = p.borrow_mut();
        for seg in segs {
            p.push(seg.as_ptr() as usize);
        }
    });
}

/// Write-combining shard buffer. Records are staged per shard in a small
/// L1-resident buffer of B records, then streamed (non-temporal stores) into
/// the shard's current ~64 KiB block. Blocks are carved in order from large
/// slab chunks, so memory is touched (and zeroed by the kernel) only as it is
/// filled, no shard needs a size estimate, and nothing is ever copied twice.
/// A shard's output is the list of its blocks' filled parts.
struct ShardWriter<T: Copy, const S: usize, const B: usize> {
    bufs: Box<[[T; B]; S]>,
    lens: [u8; S],
    cur: Vec<*mut T>,   // next write position in each shard's current block
    end: Vec<*mut T>,   // end of each shard's current block
    start: Vec<*mut T>, // start of each shard's current block
    segs: Vec<Vec<ShardSlice<T>>>,
    bump: usize, // next free block in the current chunk (0 = none yet)
    bump_end: usize,
    chunk_bytes: usize,
    blk: usize, // records per block (a multiple of B)
}

impl<T: Copy, const S: usize, const B: usize> ShardWriter<T, S, B> {
    fn new(zero: T, est_per_shard: usize) -> Self {
        let sz = std::mem::size_of::<T>();
        // Full flushes use 16-byte streaming stores: block starts are 64-byte
        // aligned (chunks are) and B records are a multiple of 16 bytes.
        assert!((B * sz) % 16 == 0);
        let blk = (BLOCK_BYTES / sz) / B * B;
        assert!(blk > 0);
        let chunk_bytes = ((est_per_shard * S).div_ceil(blk) + S) * BLOCK_BYTES;
        Self {
            bufs: vec![[zero; B]; S].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!()),
            lens: [0u8; S],
            cur: vec![std::ptr::null_mut(); S],
            end: vec![std::ptr::null_mut(); S],
            start: vec![std::ptr::null_mut(); S],
            segs: (0..S).map(|_| Vec::new()).collect(),
            bump: 0,
            bump_end: 0,
            chunk_bytes,
            blk,
        }
    }

    /// Maps a fresh slab chunk (never freed; exit(0) reclaims it). Its pages
    /// are touched only as blocks are filled.
    fn new_chunk(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.chunk_bytes, 64).unwrap();
        let p = unsafe { std::alloc::alloc(layout) } as usize;
        assert!(p != 0);
        self.bump = p;
        self.bump_end = p + self.chunk_bytes;
    }

    /// Closes the shard's current block and starts a new one.
    #[cold]
    #[inline(never)]
    fn new_block(&mut self, shard: usize) {
        let st = self.start[shard];
        let c = self.cur[shard];
        if !st.is_null() && c > st {
            let len = (c as usize - st as usize) / std::mem::size_of::<T>();
            self.segs[shard].push(ShardSlice { ptr: st, len });
        }
        let b = match BLOCK_POOL.with(|p| p.borrow_mut().pop()) {
            Some(b) => b as *mut T,
            None => {
                if self.bump_end - self.bump < BLOCK_BYTES {
                    self.new_chunk();
                }
                let b = self.bump;
                self.bump += BLOCK_BYTES;
                b as *mut T
            }
        };
        self.start[shard] = b;
        self.cur[shard] = b;
        self.end[shard] = unsafe { b.add(self.blk) };
    }

    /// Streams one full staging buffer (B records) into the shard's block with
    /// non-temporal stores: the block is cold memory that is read only in the
    /// next wave, so streaming skips the read-for-ownership of every line and
    /// keeps the producer's working set in cache.
    #[inline(always)]
    fn append_full(&mut self, shard: usize, src: *const T) {
        unsafe {
            let bytes = B * std::mem::size_of::<T>();
            let mut c = *self.cur.get_unchecked(shard);
            if (c as usize) + bytes > (*self.end.get_unchecked(shard) as usize) {
                self.new_block(shard);
                c = *self.cur.get_unchecked(shard);
            }
            let dst = c as *mut u8;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
                let s8 = src as *const u8;
                let mut i = 0;
                while i < bytes {
                    _mm_stream_si128(
                        dst.add(i) as *mut __m128i,
                        _mm_loadu_si128(s8.add(i) as *const __m128i),
                    );
                    i += 16;
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            std::ptr::copy_nonoverlapping(src as *const u8, dst, bytes);
            *self.cur.get_unchecked_mut(shard) = c.add(B);
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
            let src = unsafe { self.bufs.get_unchecked(shard).as_ptr() };
            self.append_full(shard, src);
            unsafe {
                *self.lens.get_unchecked_mut(shard) = 0;
            }
        } else {
            unsafe {
                *self.lens.get_unchecked_mut(shard) = next as u8;
            }
        }
    }

    /// Flushes the staging buffers and returns each shard's segments.
    fn finish(mut self) -> Vec<Vec<ShardSlice<T>>> {
        for s in 0..S {
            let len = self.lens[s] as usize;
            if len > 0 {
                let sz = std::mem::size_of::<T>();
                if (self.cur[s] as usize) + len * sz > (self.end[s] as usize) {
                    self.new_block(s);
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(self.bufs[s].as_ptr(), self.cur[s], len);
                    self.cur[s] = self.cur[s].add(len);
                }
            }
            let st = self.start[s];
            let c = self.cur[s];
            if !st.is_null() && c > st {
                let len = (c as usize - st as usize) / std::mem::size_of::<T>();
                self.segs[s].push(ShardSlice { ptr: st, len });
            }
        }
        // Order the streaming stores before the barrier that hands the
        // segments to other threads.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
        std::mem::take(&mut self.segs)
    }
}

/// The segments of shard `sh` from every writer's output, in writer order.
#[inline(always)]
fn shard_segs<'a, T>(
    slots: &'a [SyncCell<Vec<Vec<ShardSlice<T>>>>],
    sh: usize,
) -> impl Iterator<Item = &'a [T]> + 'a {
    slots
        .iter()
        .flat_map(move |c| unsafe { c.get() }[sh].iter().map(|seg| &**seg))
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

/// Splits a line into its F = N + 1 fields using the comma offsets from the
/// scanner, and strips quotes. `q` is the number of `"` bytes in the line, also
/// from the scanner. A field is valid if it has no quotes, or is wrapped in a
/// pair of quotes (len >= 2) with none inside. Each wrapped field accounts for
/// exactly 2 quotes, so the line is valid iff the wrapped fields account for
/// all `q` quotes. Same result as unquoting every field and rejecting stray
/// quotes, without rescanning the fields.
#[inline(always)]
fn split_fields<'a, const N: usize, const F: usize>(
    line: &'a [u8],
    c: &[usize; N],
    q: u32,
) -> Option<[&'a [u8]; F]> {
    debug_assert!(F == N + 1);
    let empty: &'a [u8] = &[];
    let mut out: [&'a [u8]; F] = [empty; F];
    let mut start = 0usize;
    let mut i = 0usize;
    while i < F {
        let end = if i < N { c[i] } else { line.len() };
        out[i] = &line[start..end];
        start = end + 1;
        i += 1;
    }
    if q != 0 {
        let mut explained = 0u32;
        let mut i = 0usize;
        while i < F {
            let f = out[i];
            let len = f.len();
            let c0 = f.first().copied().unwrap_or(0);
            let c1 = f.last().copied().unwrap_or(0);
            let w = ((len >= 2) & (c0 == b'"') & (c1 == b'"')) as usize;
            out[i] = unsafe { f.get_unchecked(w..len - w) };
            explained += 2 * w as u32;
            i += 1;
        }
        if explained != q {
            return None;
        }
    }
    Some(out)
}

/// Software prefetch hint (no-op off x86-64). Never dereferences `p`.
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Prefetches the bytes of an overflow (arena) Pack64 value; nothing for inline ones.
#[inline(always)]
fn prefetch_pack64(v: u64, arenas: &[Vec<u8>]) {
    if (v >> 63) != 0 {
        let aid = ((v >> 55) & 0xFF) as usize;
        let off = ((v >> 23) & 0xFFFF_FFFF) as usize;
        let len = (v & 0x7F_FFFF) as usize;
        if let Some(a) = arenas.get(aid) {
            // Both ends: a 33-byte ID straddles two cache lines half the time.
            prefetch(a.as_ptr().wrapping_add(off));
            prefetch(a.as_ptr().wrapping_add(off + len.max(1) - 1));
        }
    }
}

/// Records the positions of the separators in `m` (bit k set = separator at
/// `base + k / stride`) as the next field ends of the current line. Only the
/// first N field ends of a line are used, so it writes exactly N slots,
/// branch-free, starting at min(count, N): slots past the real separators get
/// junk that a later write or the count check makes irrelevant. `buf` has room
/// for 2N slots.
#[inline(always)]
fn record_separators<const N: usize, const SHIFT: u32>(
    buf: &mut [usize; 16],
    c_cnt: &mut usize,
    mut m: u64,
    base: usize,
) {
    let start = (*c_cnt).min(N);
    *c_cnt += m.count_ones() as usize;
    let mut k = 0;
    while k < N {
        unsafe {
            *buf.get_unchecked_mut(start + k) =
                base.wrapping_add((m.trailing_zeros() >> SHIFT) as usize);
        }
        m &= m.wrapping_sub(1);
        k += 1;
    }
}

#[inline(always)]
fn scan_slice_scalar<const N: usize, F: FnMut(&[u8], &[usize; N], usize, u32)>(
    buf: &[u8],
    mut on_line: F,
) {
    assert!(N <= 8);
    let len = buf.len();
    let mut line_start = 0usize;
    let mut cbuf = [0usize; 16];
    let mut c_cnt = 0usize;
    let mut q_cnt = 0u32;
    let mut i = 0usize;

    while i + 8 <= len {
        let w = u64::from_le_bytes(unsafe { *(buf.as_ptr().add(i) as *const [u8; 8]) });
        let mut nl = has_byte_mask_swar(w, b'\n');
        let mut cm = has_byte_mask_swar(w, b',');
        let mut qt = has_byte_mask_swar(w, b'"');

        while nl != 0 {
            let nl_bit = nl.trailing_zeros();
            let before = (1u64 << nl_bit) - 1;
            record_separators::<N, 3>(&mut cbuf, &mut c_cnt, cm & before, i.wrapping_sub(line_start));
            q_cnt += (qt & before).count_ones();
            let nl_pos = i + (nl_bit as usize >> 3);
            if nl_pos > line_start {
                let commas = unsafe { &*(cbuf.as_ptr() as *const [usize; N]) };
                on_line(&buf[line_start..nl_pos], commas, c_cnt, q_cnt);
            }
            let clear = (!0u64).checked_shl(nl_bit + 1).unwrap_or(0);
            nl &= clear;
            cm &= clear;
            qt &= clear;
            line_start = nl_pos + 1;
            c_cnt = 0;
            q_cnt = 0;
        }
        if cm != 0 {
            record_separators::<N, 3>(&mut cbuf, &mut c_cnt, cm, i.wrapping_sub(line_start));
        }
        q_cnt += qt.count_ones();
        i += 8;
    }
    let mut commas = [0usize; N];
    commas.copy_from_slice(&cbuf[..N]);

    while i < len {
        let b = unsafe { *buf.get_unchecked(i) };
        if b == b'\n' {
            if i > line_start {
                on_line(&buf[line_start..i], &commas, c_cnt, q_cnt);
            }
            line_start = i + 1;
            c_cnt = 0;
            q_cnt = 0;
        } else if b == b',' {
            if c_cnt < N {
                commas[c_cnt] = i - line_start;
            }
            c_cnt += 1;
        } else if b == b'"' {
            q_cnt += 1;
        }
        i += 1;
    }
    if line_start < len {
        on_line(&buf[line_start..len], &commas, c_cnt, q_cnt);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_slice_avx2<const N: usize, F: FnMut(&[u8], &[usize; N], usize, u32)>(
    buf: &[u8],
    mut on_line: F,
) {
    use std::arch::x86_64::*;
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);

    assert!(N <= 8);
    let mut line_start = 0usize;
    let mut cbuf = [0usize; 16];
    let mut c_cnt = 0usize;
    let mut q_cnt = 0u32;
    let mut i = 0usize;

    // 64 bytes per step (lines average ~80 bytes): half the loop overhead and
    // separator-recording passes of 32-byte steps.
    while i + 64 <= len {
        let v0 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(ptr.add(i + 32) as *const __m256i);
        let mask = |t: __m256i| -> u64 {
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, t)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, t)) as u32 as u64) << 32)
        };
        let mut nl = mask(v_nl);
        let mut cm = mask(v_cm);
        let mut qt = mask(v_qt);

        while nl != 0 {
            let nl_bit = nl.trailing_zeros();
            let before = (1u64 << nl_bit) - 1;
            record_separators::<N, 0>(&mut cbuf, &mut c_cnt, cm & before, i.wrapping_sub(line_start));
            q_cnt += (qt & before).count_ones();
            let nl_pos = i + (nl_bit as usize);
            if nl_pos > line_start {
                let commas = &*(cbuf.as_ptr() as *const [usize; N]);
                on_line(&buf[line_start..nl_pos], commas, c_cnt, q_cnt);
            }
            let clear = (!0u64).checked_shl(nl_bit + 1).unwrap_or(0);
            nl &= clear;
            cm &= clear;
            qt &= clear;
            line_start = nl_pos + 1;
            c_cnt = 0;
            q_cnt = 0;
        }
        if cm != 0 {
            record_separators::<N, 0>(&mut cbuf, &mut c_cnt, cm, i.wrapping_sub(line_start));
        }
        q_cnt += qt.count_ones();
        i += 64;
    }

    while i + 32 <= len {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let mut nl = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_nl)) as u32;
        let mut cm = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_cm)) as u32;
        let mut qt = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_qt)) as u32;

        while nl != 0 {
            let nl_bit = nl.trailing_zeros();
            let before = (1u32 << nl_bit) - 1;
            record_separators::<N, 0>(&mut cbuf, &mut c_cnt, (cm & before) as u64, i.wrapping_sub(line_start));
            q_cnt += (qt & before).count_ones();
            let nl_pos = i + (nl_bit as usize);
            if nl_pos > line_start {
                let commas = &*(cbuf.as_ptr() as *const [usize; N]);
                on_line(&buf[line_start..nl_pos], commas, c_cnt, q_cnt);
            }
            let clear = (!0u32).checked_shl(nl_bit + 1).unwrap_or(0);
            nl &= clear;
            cm &= clear;
            qt &= clear;
            line_start = nl_pos + 1;
            c_cnt = 0;
            q_cnt = 0;
        }
        if cm != 0 {
            record_separators::<N, 0>(&mut cbuf, &mut c_cnt, cm as u64, i.wrapping_sub(line_start));
        }
        q_cnt += qt.count_ones();
        i += 32;
    }
    let mut commas = [0usize; N];
    commas.copy_from_slice(&cbuf[..N]);

    while i < len {
        let b = *buf.get_unchecked(i);
        if b == b'\n' {
            if i > line_start {
                on_line(&buf[line_start..i], &commas, c_cnt, q_cnt);
            }
            line_start = i + 1;
            c_cnt = 0;
            q_cnt = 0;
        } else if b == b',' {
            if c_cnt < N {
                commas[c_cnt] = i - line_start;
            }
            c_cnt += 1;
        } else if b == b'"' {
            q_cnt += 1;
        }
        i += 1;
    }
    if line_start < len {
        on_line(&buf[line_start..len], &commas, c_cnt, q_cnt);
    }
}

#[inline(always)]
fn scan_slice<const N: usize, F: FnMut(&[u8], &[usize; N], usize, u32)>(
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

/// True iff all 8 bytes of `v` (little-endian load) are ASCII '0'..='9'.
#[inline(always)]
fn swar_is_8_digits(v: u64) -> bool {
    ((v & 0xF0F0_F0F0_F0F0_F0F0)
        | ((v.wrapping_add(0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0) >> 4))
        == 0x3333_3333_3333_3333
}

/// Value of 8 ASCII digits (first digit in the lowest byte). Input must pass
/// swar_is_8_digits.
#[inline(always)]
fn swar_parse_8_digits(v: u64) -> u64 {
    let v = v.wrapping_sub(0x3030_3030_3030_3030);
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let v = ((v & 0x00FF_00FF_00FF_00FF).wrapping_mul(1 + (100 << 16))) >> 16;
    ((v & 0x0000_FFFF_0000_FFFF).wrapping_mul(1 + (10_000 << 32))) >> 32
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
        // "HH:MM:SS" as one little-endian word (the colons at bytes 2 and 5 were
        // checked above): validate the six digits and combine each pair at once.
        let t = u64::from_le_bytes(unsafe { *(p.add(11) as *const [u8; 8]) });
        const COLONS: u64 = 0x0000_FF00_00FF_0000;
        let digits = (t & !COLONS) | (0x3030_3030_3030_3030 & COLONS);
        if !swar_is_8_digits(digits) {
            return None;
        }
        let d = digits.wrapping_sub(0x3030_3030_3030_3030);
        let pairs = d.wrapping_mul(10).wrapping_add(d >> 8); // bytes 0, 3, 6
        let hh = pairs & 0xFF;
        let mm = (pairs >> 24) & 0xFF;
        let ss = (pairs >> 48) & 0xFF;
        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        Some(day_ms + ((hh * 3600 + mm * 60 + ss) as i64) * 1000)
    }

    #[inline(always)]
    fn parse_ts(&mut self, b: &[u8]) -> Option<i64> {
        let len = b.len();
        if len == 20 && unsafe { *b.get_unchecked(19) } == b'Z' {
            return self.parse_datetime_prefix(b);
        }
        if len == 13 {
            // Two overlapping 8-byte loads cover all 13 bytes; SWAR validate + parse.
            let a = u64::from_le_bytes(unsafe { *(b.as_ptr() as *const [u8; 8]) });
            let c = u64::from_le_bytes(unsafe { *(b.as_ptr().add(5) as *const [u8; 8]) });
            if !swar_is_8_digits(a) || !swar_is_8_digits(c) {
                return None;
            }
            let hi = swar_parse_8_digits(a); // digits 0..8
            let lo = swar_parse_8_digits(c) % 100_000; // digits 8..13
            return Some((hi * 100_000 + lo) as i64);
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

/// One linked (non-root) ID and its person's name, with both hashes stored so a
/// probe touches one 32-byte slot and never rehashes or reads arena bytes on a
/// mismatch: [key, key_hash, root, root_hash]. Plain u64 arrays so that
/// vec![[0; 4]; n] gets zeroed pages from the OS instead of writing every slot.
/// key == 0 means empty (an ID is never empty, so its Pack64 is never 0).
type LinkEntry = [u64; 4];

/// Interning slot: [key, (tag << 32) | (node index + 1)]; second word 0 = empty.
type NodeSlot = [u64; 2];

/// How far ahead the prefetch pipelines run (in items).
const PF_DIST: usize = 16;
const DEDUP_PF: usize = 12;

#[inline(always)]
fn dsu_find(parent: &mut [u32], mut x: u32) -> u32 {
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

/// Region of an ID hash: its top 6 bits. Lookups are batched by region (wave
/// 2b), and the map and bloom filter are indexed by the top bits of the hash,
/// so one region's lookups touch only 1/64 of each (cache-resident) instead of
/// missing to DRAM all over a map of tens of MB.
const ID_REGIONS: usize = 64;

#[inline(always)]
fn id_region(h: u64) -> usize {
    (h >> 58) as usize
}

struct IdLookup {
    has_links: bool,
    bloom: Vec<u64>, // one word per 16 map slots
    entries: Vec<LinkEntry>,
    mask: usize,
    shift: u32, // slot of h = h >> shift (top log2(capacity) bits)
}

impl IdLookup {
    fn empty() -> Self {
        Self {
            has_links: false,
            bloom: Vec::new(),
            entries: Vec::new(),
            mask: 0,
            shift: 63,
        }
    }

    #[inline(always)]
    fn slot_of(&self, h: u64) -> usize {
        (h >> self.shift) as usize
    }

    /// The two bloom bits of h, within the word of its slot group (low hash
    /// bits, independent of the slot bits).
    #[inline(always)]
    fn bloom_bits(h: u64) -> u64 {
        (1u64 << (h & 63)) | (1u64 << ((h >> 6) & 63))
    }

    fn from_pairs(all_pairs: &[Vec<(u64, u64)>], arenas: &[Vec<u8>]) -> Self {
        let total_pairs: usize = all_pairs.iter().map(|v| v.len()).sum();
        if total_pairs == 0 {
            return Self::empty();
        }

        // 1) Flatten the endpoints and hash them in one sequential pass (each
        //    link arena is read in the order it was written).
        let n_ends = total_pairs * 2;
        let mut ends: Vec<u64> = Vec::with_capacity(n_ends);
        let mut end_h: Vec<u64> = Vec::with_capacity(n_ends);
        for pv in all_pairs {
            for &(a, b) in pv {
                ends.push(a);
                end_h.push(hash_pack64(a, arenas));
                ends.push(b);
                end_h.push(hash_pack64(b, arenas));
            }
        }

        // 2) Intern endpoints to dense node ids, prefetching the slot of the
        //    endpoint PF_DIST ahead so the cache misses overlap.
        let mut cap = n_ends.next_power_of_two().max(64);
        if n_ends * 4 > cap * 3 {
            cap *= 2;
        }
        let tmask = cap - 1;
        let mut tab: Vec<NodeSlot> = vec![[0u64; 2]; cap];
        let mut node_keys: Vec<u64> = Vec::with_capacity(n_ends);
        let mut node_h: Vec<u64> = Vec::with_capacity(n_ends);
        let mut ids: Vec<u32> = Vec::with_capacity(n_ends);
        for i in 0..n_ends {
            if i + PF_DIST < n_ends {
                let hp = unsafe { *end_h.get_unchecked(i + PF_DIST) };
                prefetch(tab.as_ptr().wrapping_add((hp as usize) & tmask));
            }
            let k = unsafe { *ends.get_unchecked(i) };
            let h = unsafe { *end_h.get_unchecked(i) };
            let tag = h >> 32;
            let mut p = (h as usize) & tmask;
            let id = loop {
                let s = unsafe { *tab.get_unchecked(p) };
                if s[1] == 0 {
                    let id = node_keys.len() as u32;
                    node_keys.push(k);
                    node_h.push(h);
                    unsafe {
                        *tab.get_unchecked_mut(p) = [k, (tag << 32) | (id as u64 + 1)];
                    }
                    break id;
                }
                if (s[1] >> 32) == tag && eq_pack64(s[0], k, arenas) {
                    break ((s[1] & 0xFFFF_FFFF) - 1) as u32;
                }
                p = (p + 1) & tmask;
            };
            ids.push(id);
        }
        std::mem::forget(tab);
        std::mem::forget(ends);
        std::mem::forget(end_h);

        // 3) Union by smallest name, as before. Each successful union turns
        //    exactly one root into a non-root for good, so the union count is
        //    the number of IDs that need a map entry (no separate counting pass).
        let n = node_keys.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut unions = 0usize;
        for j in 0..total_pairs {
            if j + PF_DIST < total_pairs {
                let a = unsafe { *ids.get_unchecked(2 * (j + PF_DIST)) } as usize;
                let b = unsafe { *ids.get_unchecked(2 * (j + PF_DIST) + 1) } as usize;
                prefetch(parent.as_ptr().wrapping_add(a));
                prefetch(parent.as_ptr().wrapping_add(b));
            }
            let ia = unsafe { *ids.get_unchecked(2 * j) };
            let ib = unsafe { *ids.get_unchecked(2 * j + 1) };
            let ra = dsu_find(&mut parent, ia);
            let rb = dsu_find(&mut parent, ib);
            if ra != rb {
                if cmp_pack64(node_keys[ra as usize], node_keys[rb as usize], arenas)
                    == Ordering::Less
                {
                    parent[rb as usize] = ra;
                } else {
                    parent[ra as usize] = rb;
                }
                unions += 1;
            }
        }
        std::mem::forget(ids);
        if unions == 0 {
            return Self::empty();
        }

        // 4) One 32-byte entry per non-root ID, at the slot given by the top
        //    hash bits; a blocked bloom filter (one 64-bit word per 16 slots,
        //    4-8 bits per entry) screens out IDs that are not in the map.
        let map_cap = (unions * 2).next_power_of_two().max(64);
        let mask = map_cap - 1;
        let shift = 64 - map_cap.trailing_zeros();
        let mut entries: Vec<LinkEntry> = vec![[0u64; 4]; map_cap];
        let mut bloom = vec![0u64; map_cap / 16];
        for i in 0..n {
            if i + PF_DIST < n {
                let hp = unsafe { *node_h.get_unchecked(i + PF_DIST) };
                prefetch(entries.as_ptr().wrapping_add((hp >> shift) as usize));
            }
            let r = dsu_find(&mut parent, i as u32) as usize;
            if r != i {
                let h = node_h[i];
                let mut pos = (h >> shift) as usize;
                bloom[pos >> 4] |= Self::bloom_bits(h);
                while entries[pos][0] != 0 {
                    pos = (pos + 1) & mask;
                }
                entries[pos] = [node_keys[i], h, node_keys[r], node_h[r]];
            }
        }
        std::mem::forget(parent);
        std::mem::forget(node_keys);
        std::mem::forget(node_h);

        Self {
            has_links: true,
            bloom,
            entries,
            mask,
            shift,
        }
    }

    #[inline(always)]
    fn maybe_linked(&self, h: u64) -> bool {
        let b = Self::bloom_bits(h);
        (unsafe { *self.bloom.get_unchecked(self.slot_of(h) >> 4) } & b) == b
    }

    /// Pipeline stage 1 (far ahead): pull in the bloom word.
    #[inline(always)]
    fn prefetch_bloom(&self, h: u64) {
        if self.has_links {
            prefetch(self.bloom.as_ptr().wrapping_add(self.slot_of(h) >> 4));
        }
    }

    /// Pipeline stage 2 (nearer): pull in the map slot if the ID may be
    /// linked; returns whether it may be.
    #[inline(always)]
    fn prefetch_slot(&self, h: u64) -> bool {
        if self.has_links && self.maybe_linked(h) {
            prefetch(self.entries.as_ptr().wrapping_add(self.slot_of(h)));
            true
        } else {
            false
        }
    }

    /// Person name and its hash for a raw user ID whose hash `h` is known.
    #[inline(always)]
    fn resolve_hashed(&self, p: u64, h: u64, person_arenas: &[Vec<u8>]) -> (u64, u64) {
        if self.has_links && self.maybe_linked(h) {
            let mut pos = self.slot_of(h);
            loop {
                let e = unsafe { self.entries.get_unchecked(pos) };
                if e[0] == 0 {
                    break;
                }
                if e[1] == h && eq_pack64(e[0], p, person_arenas) {
                    return (e[2], e[3]);
                }
                pos = (pos + 1) & self.mask;
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
    append_small(out, s);
}

// ============================================================================
// Core Pipeline Structs
// ============================================================================

#[derive(Clone, Copy)]
struct RawEvent {
    eid: u64,
    raw_uid: u64,
    // Hash of the user ID, computed while parsing (its bytes are hot then) so
    // the person lookup never re-reads long IDs from the arenas.
    uh: u64,
    // ingest_ts in whole epoch seconds. rules.md: "ingest_ts is always valid F1",
    // and F1 has whole seconds, so this keeps the exact dedup order in 4 bytes.
    ingest_s: u32,
    // Low 32 bits of the event_id hash computed while parsing (the top bits chose
    // the dedup shard), so dedup never rehashes or re-reads long IDs.
    eid_h: u32,
    ts_off: u32,    // u32::MAX = invalid (kept for dedup shadowing only)
    price_cat: u32, // low 24 bits: price_cents (0 = not purchase), high 8 bits: cat_idx
}

const ZERO_RAW_EVENT: RawEvent = RawEvent {
    eid: 0,
    raw_uid: 0,
    uh: 0,
    ingest_s: 0,
    eid_h: 0,
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

/// A deduplicated valid event before its person lookup, filed by the region
/// of its user ID hash.
#[derive(Clone, Copy)]
struct UEvent {
    uid: u64,
    uh: u64,
    ts_off: u32,
    price_cat: u32,
}

const ZERO_UEVENT: UEvent = UEvent {
    uid: 0,
    uh: 0,
    ts_off: 0,
    price_cat: 0,
};

/// A parsed valid click before its person lookup, filed by user-hash region.
#[derive(Clone, Copy)]
struct UClick {
    uid: u64,
    uh: u64,
    cid: u64,
    ts_off: u32,
    camp: u16,
}

const ZERO_UCLICK: UClick = UClick {
    uid: 0,
    uh: 0,
    cid: 0,
    ts_off: 0,
    camp: 0,
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

fn list_csv_dir(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
        .collect();
    paths.sort();
    paths
}

/// Opens and maps every CSV in the given directories, spread over `ncpu`
/// threads: on container bind mounts each open/mmap can cost a fraction of a
/// millisecond, which adds up over hundreds of part files if done serially.
fn mmap_csv_dirs(dirs: &[PathBuf], ncpu: usize) -> Vec<Vec<MappedFile>> {
    let lists: Vec<Vec<PathBuf>> = dirs.iter().map(|d| list_csv_dir(d)).collect();
    let jobs: Vec<(usize, usize)> = lists
        .iter()
        .enumerate()
        .flat_map(|(d, l)| (0..l.len()).map(move |i| (d, i)))
        .collect();
    let cells: Vec<OnceLock<MappedFile>> = (0..jobs.len()).map(|_| OnceLock::new()).collect();
    let next = AtomicUsize::new(0);
    let nthreads = ncpu.min(jobs.len()).max(1);
    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| loop {
                let j = next.fetch_add(1, AtomicOrd::Relaxed);
                if j >= jobs.len() {
                    break;
                }
                let (d, i) = jobs[j];
                let _ = cells[j].set(MappedFile::open(&lists[d][i]));
            });
        }
    });
    let mut out: Vec<Vec<MappedFile>> = lists.iter().map(|l| Vec::with_capacity(l.len())).collect();
    for (j, cell) in cells.into_iter().enumerate() {
        out[jobs[j].0].push(cell.into_inner().unwrap());
    }
    out
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
    // SS_TRACE=1 prints phase timings to stderr (off by default).
    let t_start = std::time::Instant::now();
    let trace = std::env::var_os("SS_TRACE").is_some();
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
        scan_slice::<1, _>(&b[start..], use_avx2, |line, c, cnt, q| {
            if cnt != 1 {
                return;
            }
            let Some([p, cat]) = split_fields::<1, 2>(line, c, q) else {
                return;
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
        scan_slice::<1, _>(&b[start..], use_avx2, |line, c, cnt, q| {
            if cnt != 1 {
                return;
            }
            let Some([cp, ch]) = split_fields::<1, 2>(line, c, q) else {
                return;
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

    let mut mapped = mmap_csv_dirs(
        &[data_dir.join("links"), data_dir.join("events"), data_dir.join("clicks")],
        ncpu,
    );
    let click_files = mapped.pop().unwrap();
    let event_files = mapped.pop().unwrap();
    let link_files = mapped.pop().unwrap();

    let link_chunks = build_chunks(&link_files);
    let event_chunks = build_chunks(&event_files);
    let click_chunks = build_chunks(&click_files);

    let total_event_bytes: usize = event_files.iter().map(|f| f.len).sum();
    let total_click_bytes: usize = click_files.iter().map(|f| f.len).sum();
    let total_link_bytes: usize = link_files.iter().map(|f| f.len).sum();
    let ev_arena_est = total_event_bytes / ncpu / 12 + 4096;
    let cl_arena_est = total_click_bytes / ncpu / 8 + 4096;
    let ln_arena_est = total_link_bytes / ncpu / 3 + 4096;
    // Shard regions are sized from v5 line lengths (events ~83 bytes, clicks ~53)
    // plus 15% for uneven thread shares. Slab regions are touched (and zeroed by
    // the kernel) in full, so over-reserving costs real time; a shard that
    // outgrows its region spills safely (see ShardWriter).
    let est_ev_dedup = ((total_event_bytes / 80) / (ncpu * DEDUP_SHARDS) * 115 / 100).max(64);
    let est_ev_person = ((total_event_bytes / 84) / (ncpu * PERSON_SHARDS) * 115 / 100).max(64);
    let est_cl_person = ((total_click_bytes / 52) / (ncpu * PERSON_SHARDS) * 115 / 100).max(32);
    let est_uev = ((total_event_bytes / 84) / (ncpu * ID_REGIONS) * 115 / 100).max(64);
    let est_ucl = ((total_click_bytes / 52) / (ncpu * ID_REGIONS) * 115 / 100).max(32);

    fs::create_dir_all(out_dir.join("user_daily")).unwrap();
    fs::create_dir_all(out_dir.join("campaign_daily")).unwrap();

    // Shared synchronization state for single persistent thread pool
    let barrier = Barrier::new(ncpu);
    let next_link = AtomicUsize::new(0);
    let next_ev_chunk = AtomicUsize::new(0);
    let next_cl_chunk = AtomicUsize::new(0);
    let next_dedup_ev = AtomicUsize::new(0);
    let next_person_shard = AtomicUsize::new(0);
    let next_region = AtomicUsize::new(0);

    let link_pairs_slots: Vec<SyncCell<Vec<(u64, u64)>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let link_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let id_lookup_cell: OnceLock<IdLookup> = OnceLock::new();
    // Link arenas, published by worker 0 together with the lookup (click parsing
    // reads them), then taken by worker 0 to start person_arenas.
    let link_arenas_cell: SyncCell<Vec<Vec<u8>>> = SyncCell::new();

    let raw_ev_slots: Vec<SyncCell<Vec<Vec<ShardSlice<RawEvent>>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let eid_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let ev_parena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();

    let cid_arena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();
    let cl_parena_slots: Vec<SyncCell<Vec<u8>>> = (0..ncpu).map(|_| SyncCell::new()).collect();

    let person_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let eid_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let cid_arenas_cell: OnceLock<Vec<Vec<u8>>> = OnceLock::new();

    let ev_person_slots: Vec<SyncCell<Vec<Vec<ShardSlice<ValidEvent>>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let cl_person_slots: Vec<SyncCell<Vec<Vec<ShardSlice<ValidClick>>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let shard_order_cell: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    let uev_slots: Vec<SyncCell<Vec<Vec<ShardSlice<UEvent>>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();
    let ucl_slots: Vec<SyncCell<Vec<Vec<ShardSlice<UClick>>>>> =
        (0..ncpu).map(|_| SyncCell::new()).collect();

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
                let next_person_shard = &next_person_shard;
                let next_region = &next_region;
                let uev_slots = &uev_slots;
                let ucl_slots = &ucl_slots;
                let link_files = &link_files;
                let event_files = &event_files;
                let click_files = &click_files;
                let link_chunks = &link_chunks;
                let event_chunks = &event_chunks;
                let click_chunks = &click_chunks;
                let link_pairs_slots = &link_pairs_slots;
                let link_arena_slots = &link_arena_slots;
                let id_lookup_cell = &id_lookup_cell;
                let link_arenas_cell = &link_arenas_cell;
                let raw_ev_slots = &raw_ev_slots;
                let eid_arena_slots = &eid_arena_slots;
                let ev_parena_slots = &ev_parena_slots;
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
                    let mut l_pairs = Vec::with_capacity(total_link_bytes / ncpu / 12 + 64);
                    let mut l_arena = Vec::with_capacity(ln_arena_est);
                    loop {
                        let ci = next_link.fetch_add(1, AtomicOrd::Relaxed);
                        if ci >= link_chunks.len() {
                            break;
                        }
                        let (fi, s0, e0) = link_chunks[ci];
                        let slice = &link_files[fi].as_slice()[s0..e0];
                        scan_slice::<1, _>(slice, use_avx2, |line, c, cnt, q| {
                            if cnt != 1 {
                                return;
                            }
                            let Some([a, b]) = split_fields::<1, 2>(line, c, q) else {
                                return;
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
                    if trace && tid == 0 {
                        eprintln!("[{:7.3} s] links parsed", t_start.elapsed().as_secs_f64());
                    }

                    // ---- WAVE 1B: Thread 0 builds DSU while ALL threads parse events & clicks ----
                    if tid == 0 {
                        // Move the link arenas out (no clone) and publish them with the
                        // lookup: click parsing reads them, worker 0 takes them back
                        // after the parse barrier to build person_arenas.
                        let mut all_pairs = Vec::with_capacity(ncpu);
                        let mut l_arenas = Vec::with_capacity(ncpu);
                        for t in 0..ncpu {
                            all_pairs.push(unsafe { link_pairs_slots[t].take() });
                            l_arenas.push(unsafe { link_arena_slots[t].take() });
                        }
                        let lookup = IdLookup::from_pairs(&all_pairs, &l_arenas);
                        if trace {
                            eprintln!(
                                "[{:7.3} s] union-find built on worker 0",
                                t_start.elapsed().as_secs_f64()
                            );
                        }
                        std::mem::forget(all_pairs);
                        unsafe { link_arenas_cell.set(l_arenas) };
                        let _ = id_lookup_cell.set(lookup);
                    }

                    // Parse events into L1-buffered 128 shards
                    let mut ev_writer =
                        ShardWriter::<RawEvent, DEDUP_SHARDS, 8>::new(ZERO_RAW_EVENT, est_ev_dedup);
                    let mut eid_arena = Vec::with_capacity(ev_arena_est);
                    let mut ev_parena = Vec::with_capacity(ev_arena_est);
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
                        scan_slice::<6, _>(slice, use_avx2, |line, c, cnt, q| {
                            if cnt != 6 {
                                return;
                            }
                            let Some([f1, f2, f3, f4, f5, f6, f7]) =
                                split_fields::<6, 7>(line, c, q)
                            else {
                                return;
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
                            let mut uh = 0u64;
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
                                                    (raw_uid, uh) = match pack64_try_inline(f2) {
                                                        Some(p) => (p, mix64(p)),
                                                        None => (
                                                            pack64(f2, ep_aid, &mut ev_parena),
                                                            hash_bytes(f2),
                                                        ),
                                                    };
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
                                                (raw_uid, uh) = match pack64_try_inline(f2) {
                                                    Some(p) => (p, mix64(p)),
                                                    None => (
                                                        pack64(f2, ep_aid, &mut ev_parena),
                                                        hash_bytes(f2),
                                                    ),
                                                };
                                                ts_off = (ts - win_start) as u32;
                                            }
                                        }
                                    }
                                }
                            }

                            ev_writer.push(
                                (eid_h >> (64 - DEDUP_BITS)) as usize,
                                RawEvent {
                                    eid,
                                    ingest_s: ingest_ts.div_euclid(1000).clamp(0, u32::MAX as i64)
                                        as u32,
                                    eid_h: eid_h as u32,
                                    raw_uid,
                                    uh,
                                    ts_off,
                                    price_cat,
                                },
                            );
                        });
                    }

                    // Clicks need no dedup: copies of a click have identical values
                    // (rules.md), so a duplicate changes neither the crediting click
                    // nor the set of assisting campaigns. Hash each click's user ID
                    // here, while it is hot, and file the click by ID region for the
                    // batched person lookup in wave 2b.
                    let mut cl_u_writer =
                        ShardWriter::<UClick, ID_REGIONS, 8>::new(ZERO_UCLICK, est_ucl);
                    let mut cid_arena = Vec::with_capacity(cl_arena_est);
                    let mut cl_parena = Vec::with_capacity(cl_arena_est);
                    let c_aid = tid as u8;
                    let cp_aid = (2 * ncpu + tid) as u8;

                    loop {
                        let ci = next_cl_chunk.fetch_add(1, AtomicOrd::Relaxed);
                        if ci >= click_chunks.len() {
                            break;
                        }
                        let (fi, s0, e0) = click_chunks[ci];
                        let slice = &click_files[fi].as_slice()[s0..e0];
                        scan_slice::<3, _>(slice, use_avx2, |line, c, cnt, q| {
                            if cnt != 3 {
                                return;
                            }
                            let Some([f1, f2, f3, f4]) = split_fields::<3, 4>(line, c, q) else {
                                return;
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
                            let cid = pack64(f1, c_aid, &mut cid_arena);
                            let (uid, uh) = match pack64_try_inline(f2) {
                                Some(p) => (p, mix64(p)),
                                None => (pack64(f2, cp_aid, &mut cl_parena), hash_bytes(f2)),
                            };
                            cl_u_writer.push(
                                id_region(uh),
                                UClick {
                                    uid,
                                    uh,
                                    cid,
                                    ts_off: (ts - click_win_start) as u32,
                                    camp: camp_idx,
                                },
                            );
                        });
                    }
                    unsafe {
                        raw_ev_slots[tid].set(ev_writer.finish());
                        eid_arena_slots[tid].set(eid_arena);
                        ev_parena_slots[tid].set(ev_parena);
                        ucl_slots[tid].set(cl_u_writer.finish());
                        cid_arena_slots[tid].set(cid_arena);
                        cl_parena_slots[tid].set(cl_parena);
                    }
                    barrier.wait();

                    if tid == 0 {
                        let mut p_arenas = Vec::with_capacity(ncpu * 3);
                        p_arenas.extend(unsafe { link_arenas_cell.take() });
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
                    if trace && tid == 0 {
                        eprintln!("[{:7.3} s] parse done (events + clicks), arenas assembled", t_start.elapsed().as_secs_f64());
                    }

                    // ---- WAVE 2: Dedup events & clicks + resolve IDs + L1-buffered person scatter ----
                    let id_lookup = id_lookup_cell.get().unwrap();
                    let person_arenas = person_arenas_cell.get().unwrap();
                    let eid_arenas = eid_arenas_cell.get().unwrap();
                    let cid_arenas = cid_arenas_cell.get().unwrap();

                    let mut uev_writer =
                        ShardWriter::<UEvent, ID_REGIONS, 16>::new(ZERO_UEVENT, est_uev);
                    let mut tab32: Vec<u32> = Vec::new();
                    let mut tab64: Vec<u64> = Vec::new();
                    let mut unique_ev: Vec<RawEvent> = Vec::new();

                    loop {
                        let sh = next_dedup_ev.fetch_add(1, AtomicOrd::Relaxed);
                        if sh >= DEDUP_SHARDS {
                            break;
                        }
                        let mut n = 0usize;
                        {
                            n += shard_segs(raw_ev_slots, sh).map(|x| x.len()).sum::<usize>();
                        }
                        if n == 0 {
                            continue;
                        }
                        let cap = ((n * 3) / 2 + 16).next_power_of_two().max(64);
                        let mask = cap - 1;
                        unique_ev.clear();
                        unique_ev.reserve(n);

                        if n < 1_000_000 {
                            tab32.clear();
                            tab32.resize(cap, 0);
                            for slice in shard_segs(raw_ev_slots, sh) {
                                let slice: &[RawEvent] = slice;
                                for (j, &ev) in slice.iter().enumerate() {
                                    // Large shards' tables outgrow L2: fetch the slot
                                    // of an event a few iterations ahead.
                                    if let Some(nx) = slice.get(j + DEDUP_PF) {
                                        prefetch(unsafe {
                                            tab32.as_ptr().add((nx.eid_h as usize) & mask)
                                        });
                                    }
                                    let h = ev.eid_h as u64;
                                    let tag = ((h >> 20) as u32) & 0xFFF;
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
                                                if ev.ingest_s < ex.ingest_s {
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
                            for slice in shard_segs(raw_ev_slots, sh) {
                                let slice: &[RawEvent] = slice;
                                for (j, &ev) in slice.iter().enumerate() {
                                    if let Some(nx) = slice.get(j + DEDUP_PF) {
                                        prefetch(unsafe {
                                            tab64.as_ptr().add((nx.eid_h as usize) & mask)
                                        });
                                    }
                                    let h = ev.eid_h as u64;
                                    let tag = h as u32;
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
                                                if ev.ingest_s < ex.ingest_s {
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

                        // The shard's raw records are all in unique_ev now.
                        recycle_blocks(shard_segs(raw_ev_slots, sh));

                        // File every surviving valid event by the region of its
                        // user ID hash.
                        for ev in unique_ev.iter() {
                            if ev.ts_off != u32::MAX {
                                let uh = ev.uh;
                                uev_writer.push(
                                    id_region(uh),
                                    UEvent {
                                        uid: ev.raw_uid,
                                        uh,
                                        ts_off: ev.ts_off,
                                        price_cat: ev.price_cat,
                                    },
                                );
                            }
                        }
                    }

                    std::mem::forget(unique_ev);
                    std::mem::forget(tab32);
                    std::mem::forget(tab64);
                    unsafe {
                        uev_slots[tid].set(uev_writer.finish());
                    }
                    barrier.wait();
                    if trace && tid == 0 {
                        eprintln!("[{:7.3} s] wave 2 done (dedup)", t_start.elapsed().as_secs_f64());
                    }

                    // ---- WAVE 2B: person lookups, one ID region at a time ----
                    // A region's events and clicks probe only its 1/64 of the link
                    // map and bloom filter, so the probes mostly hit cache.
                    let mut ev_p_writer = ShardWriter::<ValidEvent, PERSON_SHARDS, 16>::new(
                        ZERO_VALID_EVENT,
                        est_ev_person,
                    );
                    let mut cl_p_writer =
                        ShardWriter::<ValidClick, PERSON_SHARDS, 8>::new(ZERO_VALID_CLICK, est_cl_person);
                    loop {
                        let r = next_region.fetch_add(1, AtomicOrd::Relaxed);
                        if r >= ID_REGIONS {
                            break;
                        }
                        for slice in shard_segs(uev_slots, r) {
                            let slice: &[UEvent] = slice;
                            for (j, ev) in slice.iter().enumerate() {
                                if let Some(nx) = slice.get(j + PF_DIST) {
                                    id_lookup.prefetch_bloom(nx.uh);
                                }
                                if let Some(nx) = slice.get(j + PF_DIST / 2) {
                                    if id_lookup.prefetch_slot(nx.uh) {
                                        prefetch_pack64(nx.uid, person_arenas);
                                    }
                                }
                                let (person, ph) =
                                    id_lookup.resolve_hashed(ev.uid, ev.uh, person_arenas);
                                ev_p_writer.push(
                                    (ph >> (64 - PERSON_BITS)) as usize,
                                    ValidEvent {
                                        person,
                                        ts_off: ev.ts_off,
                                        price_cat: ev.price_cat,
                                    },
                                );
                            }
                        }
                        for slice in shard_segs(ucl_slots, r) {
                            let slice: &[UClick] = slice;
                            for (j, cl) in slice.iter().enumerate() {
                                if let Some(nx) = slice.get(j + PF_DIST) {
                                    id_lookup.prefetch_bloom(nx.uh);
                                }
                                if let Some(nx) = slice.get(j + PF_DIST / 2) {
                                    if id_lookup.prefetch_slot(nx.uh) {
                                        prefetch_pack64(nx.uid, person_arenas);
                                    }
                                }
                                let (person, ph) =
                                    id_lookup.resolve_hashed(cl.uid, cl.uh, person_arenas);
                                cl_p_writer.push(
                                    (ph >> (64 - PERSON_BITS)) as usize,
                                    ValidClick {
                                        person,
                                        cid: cl.cid,
                                        ts_off: cl.ts_off,
                                        campaign_idx: cl.camp,
                                    },
                                );
                            }
                        }
                        recycle_blocks(shard_segs(uev_slots, r));
                        recycle_blocks(shard_segs(ucl_slots, r));
                    }
                    unsafe {
                        ev_person_slots[tid].set(ev_p_writer.finish());
                        cl_person_slots[tid].set(cl_p_writer.finish());
                    }
                    barrier.wait();

                    if tid == 0 {
                        let mut order: Vec<(usize, usize)> = (0..PERSON_SHARDS)
                            .map(|ps| {
                                let mut cnt = 0usize;
                                {
                                    cnt += shard_segs(ev_person_slots, ps).map(|x| x.len()).sum::<usize>();
                                }
                                (ps, cnt)
                            })
                            .collect();
                        order.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                        let _ = shard_order_cell.set(order);
                    }
                    barrier.wait();
                    if trace && tid == 0 {
                        eprintln!("[{:7.3} s] wave 2 done (dedup + resolve); shard order ready", t_start.elapsed().as_secs_f64());
                    }

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

                        // ~30 events per person; /16 leaves ~2x headroom and the table
                        // still grows if needed. A compact table keeps probes cache-local.
                        let est_persons = (ev_total / 16).max(64);
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

                        for slice in shard_segs(ev_person_slots, ps) {
                            let slice: &[ValidEvent] = slice;
                            for (j, &ev) in slice.iter().enumerate() {
                                if let Some(nx) = slice.get(j + PF_DIST) {
                                    prefetch_pack64(nx.person, person_arenas);
                                }
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
                        for slice in shard_segs(ev_person_slots, ps) {
                            let slice: &[ValidEvent] = slice;
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
                        for slice in shard_segs(cl_person_slots, ps) {
                            let slice: &[ValidClick] = slice;
                            for (j, &cl) in slice.iter().enumerate() {
                                if let Some(nx) = slice.get(j + PF_DIST) {
                                    prefetch_pack64(nx.person, person_arenas);
                                }
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

    if trace {
        eprintln!("[{:7.3} s] all output written", t_start.elapsed().as_secs_f64());
    }
    std::process::exit(0);
}