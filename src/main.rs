use parquet::basic::Compression;
use parquet::file::properties::EnabledStatistics;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Date32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use memchr::memchr;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use serde::Deserialize;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const SESSION_GAP_MS: i64 = 1_800_000;
const ASSIST_WINDOW_MS: i64 = 7 * MS_PER_DAY;
const SHARDS: usize = 64;
const OUT_PARTS: usize = 8;

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

#[inline(always)]
fn hash_bytes(b: &[u8]) -> usize {
    let mut h = FxHasher::default();
    b.hash(&mut h);
    (h.finish() as usize) & (SHARDS - 1)
}

#[inline(always)]
fn for_each_line<'a>(buf: &'a [u8], mut f: impl FnMut(&'a [u8])) {
    let Some(first_nl) = memchr(b'\n', buf) else {
        return;
    };
    let mut pos = first_nl + 1;
    while pos < buf.len() {
        let end = match memchr(b'\n', &buf[pos..]) {
            Some(rel) => pos + rel,
            None => buf.len(),
        };
        let line = &buf[pos..end];
        if !line.is_empty() {
            f(line);
        }
        pos = end + 1;
    }
}

#[inline(always)]
fn split_2(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let c1 = memchr(b',', line)?;
    let rest = &line[c1 + 1..];
    if memchr(b',', rest).is_some() {
        return None;
    }
    Some((&line[..c1], rest))
}

#[inline(always)]
fn split_4(line: &[u8]) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
    let c1 = memchr(b',', line)?;
    let r1 = &line[c1 + 1..];
    let c2 = memchr(b',', r1)?;
    let r2 = &r1[c2 + 1..];
    let c3 = memchr(b',', r2)?;
    let r3 = &r2[c3 + 1..];
    if memchr(b',', r3).is_some() {
        return None;
    }
    Some((&line[..c1], &r1[..c2], &r2[..c3], r3))
}

#[inline(always)]
fn split_7(line: &[u8]) -> Option<(&[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8])> {
    let c1 = memchr(b',', line)?;
    let r1 = &line[c1 + 1..];
    let c2 = memchr(b',', r1)?;
    let r2 = &r1[c2 + 1..];
    let c3 = memchr(b',', r2)?;
    let r3 = &r2[c3 + 1..];
    let c4 = memchr(b',', r3)?;
    let r4 = &r3[c4 + 1..];
    let c5 = memchr(b',', r4)?;
    let r5 = &r4[c5 + 1..];
    let c6 = memchr(b',', r5)?;
    let r6 = &r5[c6 + 1..];
    if memchr(b',', r6).is_some() {
        return None;
    }
    Some((
        &line[..c1],
        &r1[..c2],
        &r2[..c3],
        &r3[..c4],
        &r4[..c5],
        &r5[..c6],
        r6,
    ))
}

#[inline(always)]
fn unquote(f: &[u8]) -> Option<&[u8]> {
    if memchr(b'"', f).is_none() {
        return Some(f);
    }
    if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' {
        let inner = &f[1..f.len() - 1];
        if memchr(b'"', inner).is_none() {
            return Some(inner);
        }
    }
    None
}

#[inline(always)]
fn parse_2_digits(b: &[u8]) -> Option<i32> {
    if b.len() != 2 || !b[0].is_ascii_digit() || !b[1].is_ascii_digit() {
        return None;
    }
    Some(((b[0] - b'0') as i32) * 10 + (b[1] - b'0') as i32)
}

#[inline(always)]
fn parse_4_digits(b: &[u8]) -> Option<i32> {
    if b.len() != 4
        || !b[0].is_ascii_digit()
        || !b[1].is_ascii_digit()
        || !b[2].is_ascii_digit()
        || !b[3].is_ascii_digit()
    {
        return None;
    }
    Some(
        ((b[0] - b'0') as i32) * 1000
            + ((b[1] - b'0') as i32) * 100
            + ((b[2] - b'0') as i32) * 10
            + (b[3] - b'0') as i32,
    )
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

#[inline(always)]
fn parse_datetime_prefix(b: &[u8]) -> Option<i64> {
    if b.len() < 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let y = parse_4_digits(&b[0..4])?;
    let m = parse_2_digits(&b[5..7])?;
    let d = parse_2_digits(&b[8..10])?;
    let hh = parse_2_digits(&b[11..13])?;
    let mm = parse_2_digits(&b[14..16])?;
    let ss = parse_2_digits(&b[17..19])?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let days = days_from_civil(y, m, d)?;
    Some((days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
}

#[inline(always)]
fn parse_ts(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.len() == 20 && b[19] == b'Z' {
        return parse_datetime_prefix(b);
    }
    if b.len() == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let base = parse_datetime_prefix(b)?;
        let oh = parse_2_digits(&b[20..22])?;
        let om = parse_2_digits(&b[23..25])?;
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
    let mut val: i64 = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        val = val.checked_mul(10)?.checked_add((c - b'0') as i64)?;
    }
    Some(val)
}

#[inline(always)]
fn parse_price(b: &[u8]) -> Option<i32> {
    if b.is_empty() {
        return None;
    }
    let mut val: i64 = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        val = val.saturating_mul(10).saturating_add((c - b'0') as i64);
        if val > 10_000_000 {
            return None;
        }
    }
    if (1..=10_000_000).contains(&val) {
        Some(val as i32)
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

fn read_all_parallel(paths: &[PathBuf]) -> Vec<Vec<u8>> {
    paths.par_iter().map(|p| fs::read(p).unwrap()).collect()
}

struct Dsu<'a> {
    parent: Vec<u32>,
    names: Vec<&'a [u8]>,
    id_to_idx: FxHashMap<&'a [u8], u32>,
}

impl<'a> Dsu<'a> {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            names: Vec::new(),
            id_to_idx: FxHashMap::default(),
        }
    }

    #[inline(always)]
    fn get_or_insert(&mut self, s: &'a [u8]) -> u32 {
        if let Some(&idx) = self.id_to_idx.get(s) {
            return idx;
        }
        let idx = self.names.len() as u32;
        self.names.push(s);
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
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            if self.names[ra as usize] < self.names[rb as usize] {
                self.parent[rb as usize] = ra;
            } else {
                self.parent[ra as usize] = rb;
            }
        }
    }
}

fn build_id_map<'a>(link_buffers: &'a [Vec<u8>]) -> FxHashMap<&'a [u8], &'a [u8]> {
    let mut dsu = Dsu::new();
    for buf in link_buffers {
        for_each_line(buf, |line| {
            let Some((a, b)) = split_2(line) else { return };
            let (Some(a), Some(b)) = (unquote(a), unquote(b)) else {
                return;
            };
            if a.is_empty() || b.is_empty() {
                return;
            }
            let ia = dsu.get_or_insert(a);
            let ib = dsu.get_or_insert(b);
            dsu.union(ia, ib);
        });
    }
    let mut map = FxHashMap::default();
    let n = dsu.names.len() as u32;
    map.reserve(n as usize);
    for i in 0..n {
        let r = dsu.find(i);
        if r != i {
            map.insert(dsu.names[i as usize], dsu.names[r as usize]);
        }
    }
    map
}

#[derive(Clone, Copy)]
struct ValidEvent<'a> {
    person_id: &'a [u8],
    ts: i64,
    price_cents: i32,
    cat_idx: u16,
    is_purchase: bool,
}

#[derive(Clone, Copy)]
struct ValidClick<'a> {
    click_id: &'a [u8],
    person_id: &'a [u8],
    campaign_idx: u16,
    ts: i64,
}

struct UserDayRow<'a> {
    user_id: &'a [u8],
    day: i32,
    sessions: i64,
    events: i64,
    revenue_cents: i64,
    top_category: Option<u16>,
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

    let manifest: Manifest =
        serde_json::from_slice(&fs::read(data_dir.join("manifest.json")).unwrap()).unwrap();
    let win_start = parse_ts(manifest.window_start.as_bytes()).unwrap();
    let win_end = parse_ts(manifest.window_end.as_bytes()).unwrap();
    let click_win_start = win_start - ASSIST_WINDOW_MS;

    // Read all input files in parallel and keep buffers alive for zero-copy slices
    let prod_buf = fs::read(data_dir.join("products.csv")).unwrap();
    let camp_buf = fs::read(data_dir.join("campaigns.csv")).unwrap();
    let link_buffers = read_all_parallel(&list_csvs(&data_dir.join("links")));
    let event_buffers = read_all_parallel(&list_csvs(&data_dir.join("events")));
    let click_buffers = read_all_parallel(&list_csvs(&data_dir.join("clicks")));

    // 1. Load products.csv -> category index (0 = "unknown")
    let mut categories: Vec<&str> = vec!["unknown"];
    let mut cat_to_idx: FxHashMap<&[u8], u16> = FxHashMap::default();
    cat_to_idx.insert(b"unknown", 0);
    let mut prod_to_cat: FxHashMap<&[u8], u16> = FxHashMap::default();
    for_each_line(&prod_buf, |line| {
        if let Some((p, c)) = split_2(line) {
            if let (Some(p), Some(c)) = (unquote(p), unquote(c)) {
                let idx = *cat_to_idx.entry(c).or_insert_with(|| {
                    let i = categories.len() as u16;
                    categories.push(std::str::from_utf8(c).unwrap());
                    i
                });
                prod_to_cat.insert(p, idx);
            }
        }
    });

    // 2. Load campaigns.csv -> campaign index (0 = "organic")
    let mut camp_names: Vec<&str> = vec!["organic"];
    let mut camp_channels: Vec<&str> = vec!["organic"];
    let mut camp_to_idx: FxHashMap<&[u8], u16> = FxHashMap::default();
    for_each_line(&camp_buf, |line| {
        if let Some((c, ch)) = split_2(line) {
            if let (Some(c), Some(ch)) = (unquote(c), unquote(ch)) {
                let idx = camp_names.len() as u16;
                camp_names.push(std::str::from_utf8(c).unwrap());
                camp_channels.push(std::str::from_utf8(ch).unwrap());
                camp_to_idx.insert(c, idx);
            }
        }
    });

    // 3. Build Linked IDs map (zero-copy)
    let id_map = build_id_map(&link_buffers);

    // 4. Parse & sharded-dedup Events in parallel (zero heap allocations per row)
    let per_file_events: Vec<Vec<Vec<(&[u8], i64, Option<ValidEvent>)>>> = event_buffers
        .par_iter()
        .map(|buf| {
            let mut shards: Vec<Vec<(&[u8], i64, Option<ValidEvent>)>> =
                (0..SHARDS).map(|_| Vec::new()).collect();
            for_each_line(buf, |line| {
                let Some((f1, f2, f3, f4, f5, f6, f7)) = split_7(line) else {
                    return;
                };
                let (
                    Some(eid),
                    Some(uid),
                    Some(ts_b),
                    Some(ing_b),
                    Some(et),
                    Some(pid),
                    Some(pr),
                ) = (
                    unquote(f1),
                    unquote(f2),
                    unquote(f3),
                    unquote(f4),
                    unquote(f5),
                    unquote(f6),
                    unquote(f7),
                )
                else {
                    return;
                };
                if eid.is_empty() {
                    return;
                }
                let Some(ingest_ts) = parse_ts(ing_b) else {
                    return;
                };
                let valid = if !uid.is_empty() && !pid.is_empty() {
                    let is_purchase = match et {
                        b"view" | b"cart" => Some(false),
                        b"purchase" => Some(true),
                        _ => None,
                    };
                    if let (Some(is_p), Some(ts)) = (is_purchase, parse_ts(ts_b)) {
                        if ts >= win_start && ts < win_end && (ingest_ts - ts) <= MS_PER_DAY {
                            let price_ok = if is_p { parse_price(pr) } else { Some(0) };
                            if let Some(price_cents) = price_ok {
                                let person_id = id_map.get(uid).copied().unwrap_or(uid);
                                let cat_idx = *prod_to_cat.get(pid).unwrap_or(&0);
                                Some(ValidEvent {
                                    person_id,
                                    ts,
                                    price_cents,
                                    cat_idx,
                                    is_purchase: is_p,
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                shards[hash_bytes(eid)].push((eid, ingest_ts, valid));
            });
            shards
        })
        .collect();

    let event_person_shards: Vec<Vec<Vec<ValidEvent>>> = (0..SHARDS)
        .into_par_iter()
        .map(|s| {
            let est: usize = per_file_events.iter().map(|f| f[s].len()).sum();
            let mut dedup: FxHashMap<&[u8], (i64, Option<ValidEvent>)> =
                FxHashMap::with_capacity_and_hasher(est, Default::default());
            for f in &per_file_events {
                for &(eid, ing, ev) in &f[s] {
                    match dedup.get_mut(eid) {
                        Some(entry) => {
                            if ing < entry.0 {
                                entry.0 = ing;
                                entry.1 = ev;
                            }
                        }
                        None => {
                            dedup.insert(eid, (ing, ev));
                        }
                    }
                }
            }
            let mut out: Vec<Vec<ValidEvent>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for (_, (_, ev_opt)) in dedup {
                if let Some(ev) = ev_opt {
                    out[hash_bytes(ev.person_id)].push(ev);
                }
            }
            out
        })
        .collect();
    drop(per_file_events);

    // 5. Parse & sharded-dedup Clicks in parallel
    let per_file_clicks: Vec<Vec<Vec<ValidClick>>> = click_buffers
        .par_iter()
        .map(|buf| {
            let mut shards: Vec<Vec<ValidClick>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for_each_line(buf, |line| {
                let Some((f1, f2, f3, f4)) = split_4(line) else {
                    return;
                };
                let (Some(cid), Some(uid), Some(camp), Some(ts_b)) =
                    (unquote(f1), unquote(f2), unquote(f3), unquote(f4))
                else {
                    return;
                };
                if cid.is_empty() || uid.is_empty() {
                    return;
                }
                let Some(ts) = parse_ts(ts_b) else { return };
                if ts < click_win_start || ts >= win_end {
                    return;
                }
                let Some(&campaign_idx) = camp_to_idx.get(camp) else {
                    return;
                };
                let person_id = id_map.get(uid).copied().unwrap_or(uid);
                shards[hash_bytes(cid)].push(ValidClick {
                    click_id: cid,
                    person_id,
                    campaign_idx,
                    ts,
                });
            });
            shards
        })
        .collect();

    let click_person_shards: Vec<Vec<Vec<ValidClick>>> = (0..SHARDS)
        .into_par_iter()
        .map(|s| {
            let est: usize = per_file_clicks.iter().map(|f| f[s].len()).sum();
            let mut dedup: FxHashMap<&[u8], ValidClick> =
                FxHashMap::with_capacity_and_hasher(est, Default::default());
            for f in &per_file_clicks {
                for &cl in &f[s] {
                    dedup.entry(cl.click_id).or_insert(cl);
                }
            }
            let mut out: Vec<Vec<ValidClick>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for (_, cl) in dedup {
                out[hash_bytes(cl.person_id)].push(cl);
            }
            out
        })
        .collect();
    drop(per_file_clicks);

    // 6. Per-person-shard sessionization + linear two-pointer attribution & assists
    let num_campaigns = camp_names.len();
    let shard_results: Vec<(Vec<UserDayRow>, FxHashMap<(u16, i32), CampaignDayAgg>)> = (0..SHARDS)
        .into_par_iter()
        .map(|ps| {
            let mut by_person_events: FxHashMap<&[u8], Vec<ValidEvent>> = FxHashMap::default();
            for src in &event_person_shards {
                for &ev in &src[ps] {
                    by_person_events.entry(ev.person_id).or_default().push(ev);
                }
            }
            let mut by_person_clicks: FxHashMap<&[u8], Vec<ValidClick>> = FxHashMap::default();
            for src in &click_person_shards {
                for &cl in &src[ps] {
                    by_person_clicks.entry(cl.person_id).or_default().push(cl);
                }
            }

            let mut user_rows = Vec::with_capacity(by_person_events.len() * 2);
            let mut camp_agg: FxHashMap<(u16, i32), CampaignDayAgg> = FxHashMap::default();
            let mut camp_counts = vec![0u32; num_campaigns];
            let mut active_camps: Vec<u16> = Vec::new();

            for (person_id, mut events) in by_person_events {
                events.sort_unstable_by_key(|e| e.ts);

                let mut clicks = by_person_clicks.remove(person_id).unwrap_or_default();
                clicks.sort_unstable_by(|a, b| {
                    a.ts.cmp(&b.ts).then_with(|| b.click_id.cmp(a.click_id))
                });

                for &c in &active_camps {
                    camp_counts[c as usize] = 0;
                }
                active_camps.clear();

                let mut left = 0usize;
                let mut right = 0usize;

                let mut day_map: FxHashMap<i32, (i64, i64, i64, FxHashMap<u16, i64>)> =
                    FxHashMap::default();

                let mut i = 0usize;
                while i < events.len() {
                    let sess_start = events[i].ts;
                    let day = sess_start.div_euclid(MS_PER_DAY) as i32;
                    let mut sess_events = 0i64;
                    let mut sess_rev = 0i64;
                    let mut has_purchase = false;
                    let mut prev_ts = sess_start;

                    let d_entry = day_map
                        .entry(day)
                        .or_insert_with(|| (0, 0, 0, FxHashMap::default()));
                    d_entry.0 += 1;

                    while i < events.len() {
                        let ev = &events[i];
                        if ev.ts - prev_ts > SESSION_GAP_MS {
                            break;
                        }
                        prev_ts = ev.ts;
                        sess_events += 1;
                        if ev.is_purchase {
                            has_purchase = true;
                            sess_rev += ev.price_cents as i64;
                            *d_entry.3.entry(ev.cat_idx).or_insert(0) += ev.price_cents as i64;
                        }
                        i += 1;
                    }
                    d_entry.1 += sess_events;
                    d_entry.2 += sess_rev;

                    while right < clicks.len() && clicks[right].ts <= sess_start {
                        let c = clicks[right].campaign_idx;
                        if camp_counts[c as usize] == 0 {
                            active_camps.push(c);
                        }
                        camp_counts[c as usize] += 1;
                        right += 1;
                    }
                    let min_click_ts = sess_start - ASSIST_WINDOW_MS;
                    while left < right && clicks[left].ts < min_click_ts {
                        let c = clicks[left].campaign_idx;
                        camp_counts[c as usize] -= 1;
                        left += 1;
                    }

                    let credited_camp = if left < right {
                        clicks[right - 1].campaign_idx
                    } else {
                        0
                    };
                    let cr = camp_agg.entry((credited_camp, day)).or_default();
                    cr.sessions += 1;
                    if has_purchase {
                        cr.purchasing_sessions += 1;
                        cr.revenue_cents += sess_rev;
                    }

                    if left < right {
                        active_camps.retain(|&c| camp_counts[c as usize] > 0);
                        for &c in &active_camps {
                            camp_agg.entry((c, day)).or_default().assisted_sessions += 1;
                        }
                    } else {
                        active_camps.clear();
                    }
                }

                for (day, (sessions, ev_cnt, rev, cat_revs)) in day_map {
                    let top_category = cat_revs
                        .into_iter()
                        .max_by(|(c1, r1), (c2, r2)| {
                            r1.cmp(r2).then_with(|| {
                                categories[*c2 as usize].cmp(categories[*c1 as usize])
                            })
                        })
                        .map(|(c, _)| c);
                    user_rows.push(UserDayRow {
                        user_id: person_id,
                        day,
                        sessions,
                        events: ev_cnt,
                        revenue_cents: rev,
                        top_category,
                    });
                }
            }

            (user_rows, camp_agg)
        })
        .collect();

    // 7. Write user_daily in parallel (8 part files) and campaign_daily
    fs::create_dir_all(out_dir.join("user_daily")).unwrap();
    fs::create_dir_all(out_dir.join("campaign_daily")).unwrap();

    let user_schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let shards_per_part = SHARDS / OUT_PARTS;
    (0..OUT_PARTS).into_par_iter().for_each(|part| {
        let start = part * shards_per_part;
        let end = start + shards_per_part;
        let total: usize = shard_results[start..end].iter().map(|(u, _)| u.len()).sum();
        if total == 0 && part > 0 {
            return;
        }
        let mut uids = Vec::with_capacity(total);
        let mut days = Vec::with_capacity(total);
        let mut sessions = Vec::with_capacity(total);
        let mut events = Vec::with_capacity(total);
        let mut revs = Vec::with_capacity(total);
        let mut top_cats: Vec<Option<&str>> = Vec::with_capacity(total);

        for (u_rows, _) in &shard_results[start..end] {
            for r in u_rows {
                uids.push(std::str::from_utf8(r.user_id).unwrap());
                days.push(r.day);
                sessions.push(r.sessions);
                events.push(r.events);
                revs.push(r.revenue_cents);
                top_cats.push(r.top_category.map(|c| categories[c as usize]));
            }
        }

        let batch = RecordBatch::try_new(
            user_schema.clone(),
            vec![
                Arc::new(StringArray::from(uids)),
                Arc::new(Date32Array::from(days)),
                Arc::new(Int64Array::from(sessions)),
                Arc::new(Int64Array::from(events)),
                Arc::new(Int64Array::from(revs)),
                Arc::new(StringArray::from(top_cats)),
            ],
        )
        .unwrap();

        let file = File::create(out_dir.join(format!("user_daily/part-{part}.parquet"))).unwrap();
        let props = WriterProperties::builder()
    .set_compression(Compression::UNCOMPRESSED)
    .set_dictionary_enabled(false)
    .set_statistics_enabled(EnabledStatistics::None)
    .build();
        let mut writer = ArrowWriter::try_new(file, user_schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    });

    let mut final_camp: FxHashMap<(u16, i32), CampaignDayAgg> = FxHashMap::default();
    for (_, c_map) in &shard_results {
        for (&k, &v) in c_map {
            let e = final_camp.entry(k).or_default();
            e.sessions += v.sessions;
            e.purchasing_sessions += v.purchasing_sessions;
            e.revenue_cents += v.revenue_cents;
            e.assisted_sessions += v.assisted_sessions;
        }
    }

    let n = final_camp.len();
    let mut cids = Vec::with_capacity(n);
    let mut chs = Vec::with_capacity(n);
    let mut days = Vec::with_capacity(n);
    let mut sessions = Vec::with_capacity(n);
    let mut p_sessions = Vec::with_capacity(n);
    let mut revs = Vec::with_capacity(n);
    let mut assists = Vec::with_capacity(n);

    for ((c_idx, day), v) in final_camp {
        cids.push(camp_names[c_idx as usize]);
        chs.push(camp_channels[c_idx as usize]);
        days.push(day);
        sessions.push(v.sessions);
        p_sessions.push(v.purchasing_sessions);
        revs.push(v.revenue_cents);
        assists.push(v.assisted_sessions);
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
            Arc::new(StringArray::from(cids)),
            Arc::new(StringArray::from(chs)),
            Arc::new(Date32Array::from(days)),
            Arc::new(Int64Array::from(sessions)),
            Arc::new(Int64Array::from(p_sessions)),
            Arc::new(Int64Array::from(revs)),
            Arc::new(Int64Array::from(assists)),
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