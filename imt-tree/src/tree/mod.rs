use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use ff::PrimeField as _;
use pasta_curves::Fp;
use rayon::prelude::*;
use tracing::info;

pub(crate) use crate::hasher::SinsemillaHasher;
pub use crate::proof::ImtProofData;

mod nullifier_tree;
pub use nullifier_tree::*;

#[cfg(test)]
mod tests;

/// Depth of the nullifier range Merkle tree.
///
/// With paired leaves (each range occupies 2 leaf slots), the tree supports
/// up to 2^28 ranges (~268M). This is sufficient for Zcash mainnet which
/// currently has under 64M Orchard nullifiers.
pub const TREE_DEPTH: usize = 29;

/// Byte size of one serialized `Fp` field element (Pallas base field).
const FP_BYTES: usize = 32;

/// Byte size of one serialized `Range` (`[low, width]` = 2 `Fp` values).
const RANGE_BYTES: usize = FP_BYTES * 2;

/// A gap range `[low, width]` representing an interval between two adjacent
/// on-chain nullifiers. `low` is the interval start and `width = high - low`
/// where `high` is the inclusive upper bound.
///
/// **Paired-leaf model**: each range becomes two adjacent leaves in the
/// Merkle tree — `low` at position `2k` and `high = low + width` at
/// position `2k + 1`. This matches the Orchard nullifier tree convention
/// where `nf_start` is always the left leaf and `nf_end` is its sibling.
///
/// **Exclusion proof**: to prove a value `x` is not a nullifier, the prover
/// reveals a range `[low, width]` where `x - low <= width` plus a Merkle path
/// proving that range is committed in the tree. Field subtraction handles the
/// lower bound check: if `x < low`, the result wraps to a value larger than
/// any valid width.
///
/// Every on-chain nullifier `n` acts as a boundary between two adjacent ranges:
/// the range before it has upper bound `n - 1` and the range after has `low = n + 1`.
/// Because the bounds are `n +/- 1`, the nullifier `n` itself falls outside every
/// range -- so the membership check can only succeed for non-nullifier values.
///
/// Example with sorted nullifiers `[n1, n2]`:
/// ```text
///   Range 0: [0,    n1-1-0]       <- gap before n1
///   Range 1: [n1+1, n2-1-(n1+1)]  <- gap between n1 and n2
///   Range 2: [n2+1, MAX-(n2+1)]   <- gap after n2
/// ```
/// `n1` is the boundary of ranges 0 and 1; `n2` is the boundary of ranges 1
/// and 2. Neither `n1` nor `n2` is contained in any range.
///
/// ## Tree structure and padding
///
/// The tree has a fixed depth of [`TREE_DEPTH`]. With `n` on-chain nullifiers
/// the tree contains `n + 1` ranges, each producing 2 leaves (total `2(n+1)`).
/// The remaining leaf slots are filled with the Orchard "uncommitted" value
/// `Fp::from(2)`. At each level, the empty hash is computed by self-hashing
/// the level below using Sinsemilla MerkleCRH with level domain separation:
/// `empty[0] = Sinsemilla(l=0, 2, 2)`, `empty[i+1] = Sinsemilla(l=i+1, empty[i], empty[i])`.
///
/// This matches the Orchard `EMPTY_ROOTS` exactly, so the IMT tree can be
/// extended to depth 32 by padding with `EMPTY_ROOTS[29..32]`.
pub type Range = [Fp; 2];

/// Return type for [`load_full_tree`]: `(ranges, levels, root, height)`.
pub type FullTreeData = (Vec<Range>, Vec<Vec<Fp>>, Fp, Option<u64>);

/// Build gap ranges from a sorted nullifier set.
///
/// For each consecutive pair of nullifiers, the gap `[low, width]` is emitted
/// where `width = high - low` (inclusive upper bound minus lower bound).
/// A final range `[last_nf + 1, Fp::MAX - (last_nf + 1)]` closes the space.
pub fn build_nf_ranges(nfs: impl IntoIterator<Item = Fp>) -> Vec<Range> {
    let mut prev = Fp::zero();
    let mut ranges = vec![];
    for r in nfs {
        if prev < r {
            let high = r - Fp::one();
            ranges.push([prev, high - prev]);
        }
        prev = r + Fp::one();
    }
    if prev != Fp::zero() {
        let high = Fp::one().neg();
        ranges.push([prev, high - prev]);
    }
    ranges
}

/// Expand ranges into paired leaves: `[low, high, low, high, ...]`.
///
/// Each `[low, width]` range produces two adjacent leaves:
/// - Position `2k`: `low`
/// - Position `2k+1`: `high = low + width`
///
/// This matches the Orchard nullifier tree convention where `nf_start` is
/// always at an even position and `nf_end` is the odd sibling.
pub fn expand_ranges(ranges: &[Range]) -> Vec<Fp> {
    let mut leaves = Vec::with_capacity(ranges.len() * 2);
    for &[low, width] in ranges {
        leaves.push(low);
        leaves.push(low + width);
    }
    leaves
}

/// Hash each `(low, width)` range pair into a single leaf commitment using
/// Sinsemilla MerkleCRH at level 0.
///
/// The leaf hash is `Sinsemilla(l=0, low, high)` where `high = low + width`.
/// This is used by the PIR tree, which stores pre-hashed leaves and starts
/// its internal hashing at level 1.
pub fn commit_ranges(ranges: &[Range]) -> Vec<Fp> {
    ranges
        .par_iter()
        .map_init(SinsemillaHasher::new, |hasher, [low, width]| {
            hasher.hash(0, *low, *low + *width)
        })
        .collect()
}

/// Pre-compute the empty subtree hash at each tree level using Sinsemilla.
///
/// `empty[0] = Sinsemilla(l=0, UNCOMMITTED, UNCOMMITTED)` — the hash of two
/// uncommitted (Fp::from(2)) leaves at level 0.
/// `empty[i] = Sinsemilla(l=i, empty[i-1], empty[i-1])` — the hash of a
/// fully-empty subtree of height `i`.
///
/// These values are identical to the Orchard `EMPTY_ROOTS[1..TREE_DEPTH+1]`
/// since they use the same hash function and empty leaf value.
pub fn precompute_empty_hashes() -> [Fp; TREE_DEPTH] {
    let hasher = SinsemillaHasher::new();
    let empty_leaf = Fp::from(crate::hasher::UNCOMMITTED_ORCHARD);
    let mut empty = [Fp::default(); TREE_DEPTH];
    empty[0] = hasher.hash(0, empty_leaf, empty_leaf);
    for i in 1..TREE_DEPTH {
        empty[i] = hasher.hash(i, empty[i - 1], empty[i - 1]);
    }
    empty
}

/// Build the Merkle tree bottom-up, retaining all intermediate levels.
///
/// Returns `(root, levels)` where `levels[i]` contains the node hashes at
/// tree level `i` (level 0 = input leaves, padded to even length).
/// Each level is padded using the pre-computed empty hash so that pair-wise
/// hashing produces the next level cleanly.
///
/// `level_offset` controls the Sinsemilla level domain separation:
/// - For the full paired-leaf tree (NullifierTree), use `level_offset = 0`.
///   Level 0 contains raw `(low, high)` values and the first hash at level 0
///   combines them with `Sinsemilla(l=0, low, high)`.
/// - For the PIR tree, use `level_offset = 1`. Level 0 contains pre-hashed
///   leaves (`Sinsemilla(l=0, low, high)`) computed by `commit_ranges()`, so
///   the first internal hash should use `Sinsemilla(l=1, ...)`.
///
/// Uses Sinsemilla MerkleCRH with level-based domain separation at each
/// level, matching the Orchard Merkle tree hash exactly.
pub fn build_levels(
    mut leaves: Vec<Fp>,
    empty: &[Fp; TREE_DEPTH],
    depth: usize,
    level_offset: usize,
) -> (Fp, Vec<Vec<Fp>>) {
    let hasher = SinsemillaHasher::new();
    let empty_leaf = Fp::from(crate::hasher::UNCOMMITTED_ORCHARD);
    let mut levels: Vec<Vec<Fp>> = Vec::with_capacity(depth);

    // Level 0 padding value depends on the level_offset:
    // - offset=0: raw leaves, pad with the uncommitted leaf value (Fp::from(2))
    // - offset=1: pre-hashed leaves (commit_ranges), pad with empty[0] = hash(0, 2, 2)
    let level0_pad = if level_offset == 0 {
        empty_leaf
    } else {
        empty[level_offset - 1]
    };
    if leaves.is_empty() {
        leaves.push(level0_pad);
    }
    if leaves.len() & 1 == 1 {
        leaves.push(level0_pad);
    }
    levels.push(leaves);

    const PAR_THRESHOLD: usize = 1024;

    for i in 0..depth - 1 {
        let prev = &levels[i];
        let pairs = prev.len() / 2;
        let sinsemilla_level = i + level_offset;
        let mut next: Vec<Fp> = if pairs >= PAR_THRESHOLD {
            prev.par_chunks_exact(2)
                .map_init(SinsemillaHasher::new, |h, pair| h.hash(sinsemilla_level, pair[0], pair[1]))
                .collect()
        } else {
            (0..pairs)
                .map(|j| hasher.hash(sinsemilla_level, prev[j * 2], prev[j * 2 + 1]))
                .collect()
        };
        if next.len() & 1 == 1 {
            // next is at tree level i+1 (result of hashing level-i pairs).
            // The empty hash for level i+1 nodes is empty[i + level_offset].
            // empty[k] = hash of a fully-empty subtree at Sinsemilla level k,
            // which represents a node at tree level k+1.
            // For level_offset=0: tree level i+1 → empty[i]
            // For level_offset=1: tree level i+1 but Sinsemilla level i+1 → empty[i+1]
            next.push(empty[i + level_offset]);
        }
        levels.push(next);
    }

    let top = &levels[depth - 1];
    let root = hasher.hash(depth - 1 + level_offset, top[0], top[1]);

    (root, levels)
}

/// Find the gap-range index that contains `value`.
///
/// Returns `Some(i)` where `ranges[i]` is `[low, width]`,
/// or `None` if the value is an existing nullifier.
///
/// Membership check: `value - low <= width` (field subtraction). If `value < low`
/// the result wraps to a huge value > width, so this single comparison covers
/// both bounds.
///
/// Uses binary search (`partition_point`) on the sorted, non-overlapping
/// ranges for O(log n) lookup instead of a linear scan.
pub fn find_range_for_value(ranges: &[Range], value: Fp) -> Option<usize> {
    // Find the first range whose `low` is greater than `value`.
    // All ranges before that index have `low <= value`.
    let i = ranges.partition_point(|[low, _]| *low <= value);
    if i == 0 {
        return None;
    }
    let idx = i - 1;
    let [low, width] = ranges[idx];
    // value - low <= width: if value < low, field subtraction wraps to a huge
    // value that exceeds any valid width, so the check fails correctly.
    let offset = value - low;
    if offset <= width {
        Some(idx)
    } else {
        None
    }
}

/// Serialize gap ranges to a binary file.
///
/// Format: `[8-byte LE count][count x 2 x 32-byte Fp representations]`
pub fn save_tree(path: &Path, ranges: &[Range]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    let count = ranges.len() as u64;
    f.write_all(&count.to_le_bytes())?;
    for [low, width] in ranges {
        f.write_all(&low.to_repr())?;
        f.write_all(&width.to_repr())?;
    }
    Ok(())
}

/// Deserialize gap ranges from a binary file written by [`save_tree`].
///
/// Uses a single `read` syscall followed by parallel parsing for speed.
pub fn load_tree(path: &Path) -> Result<Vec<Range>> {
    let t0 = Instant::now();
    let buf = std::fs::read(path)?;
    anyhow::ensure!(buf.len() >= 8, "tree file too small");
    let count = u64::from_le_bytes(
        buf[..8].try_into().expect("load_tree: header slice is exactly 8 bytes"),
    ) as usize;
    let expected = 8 + count * RANGE_BYTES;
    anyhow::ensure!(
        buf.len() >= expected,
        "tree file truncated: expected {} bytes, got {}",
        expected,
        buf.len()
    );
    let ranges: Vec<Range> = buf[8..8 + count * RANGE_BYTES]
        .par_chunks_exact(RANGE_BYTES)
        .map(|chunk| {
            let low_arr: [u8; FP_BYTES] = chunk[..FP_BYTES].try_into().expect("chunk is exactly FP_BYTES");
            let width_arr: [u8; FP_BYTES] = chunk[FP_BYTES..RANGE_BYTES].try_into().expect("chunk is exactly FP_BYTES");
            let low = Option::from(Fp::from_repr(low_arr)).expect("non-canonical Fp in tree file");
            let width = Option::from(Fp::from_repr(width_arr)).expect("non-canonical Fp in tree file");
            [low, width]
        })
        .collect();
    info!(
        count = ranges.len(),
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        "ranges loaded from file"
    );
    Ok(ranges)
}

/// Serialize a full Merkle tree (ranges + all levels + root + height) to a binary file.
///
/// Format:
/// ```text
/// [8-byte LE range_count]
/// [range_count x 2 x 32-byte Fp]        -- ranges
/// [for each of TREE_DEPTH levels:
///     [8-byte LE level_len]
///     [level_len x 32-byte Fp]           -- node hashes at this level
/// ]
/// [32-byte Fp root]
/// [8-byte LE height]                     -- optional trailer (0 = unknown)
/// ```
///
/// On reload via [`load_full_tree`], zero hashing is required -- all data is
/// read directly from the file.
pub fn save_full_tree(
    path: &Path,
    ranges: &[Range],
    levels: &[Vec<Fp>],
    root: Fp,
    height: Option<u64>,
) -> Result<()> {
    let t0 = Instant::now();
    let mut f = std::fs::File::create(path)?;

    // Ranges
    let range_count = ranges.len() as u64;
    f.write_all(&range_count.to_le_bytes())?;
    for [low, width] in ranges {
        f.write_all(&low.to_repr())?;
        f.write_all(&width.to_repr())?;
    }

    // Levels
    for level in levels {
        let level_len = level.len() as u64;
        f.write_all(&level_len.to_le_bytes())?;
        for node in level {
            f.write_all(&node.to_repr())?;
        }
    }

    // Root
    f.write_all(&root.to_repr())?;

    // Height trailer (0 means unknown)
    f.write_all(&height.unwrap_or(0).to_le_bytes())?;

    info!(
        range_count = ranges.len(),
        level_count = levels.len(),
        ?height,
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        "full tree saved"
    );
    Ok(())
}

/// Deserialize a full Merkle tree from a binary file written by [`save_full_tree`].
///
/// Returns `(ranges, levels, root, height)` with zero hashing -- all data is
/// read directly from the file using bulk I/O and parallel parsing.
///
/// Backwards-compatible: files written without the height trailer return `None`.
pub fn load_full_tree(path: &Path) -> Result<FullTreeData> {
    let t0 = Instant::now();
    let buf = std::fs::read(path)?;
    info!(
        size_mb = format!("{:.1}", buf.len() as f64 / (1024.0 * 1024.0)),
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        "full tree file read"
    );

    let t1 = Instant::now();
    let mut pos = 0usize;

    // Helper: read N bytes from buf
    macro_rules! read_bytes {
        ($n:expr) => {{
            let end = pos + $n;
            anyhow::ensure!(end <= buf.len(), "unexpected EOF in full tree file");
            let slice = &buf[pos..end];
            pos = end;
            slice
        }};
    }

    // Ranges
    let range_count = u64::from_le_bytes(
        read_bytes!(8).try_into().expect("header slice is exactly 8 bytes"),
    ) as usize;
    let range_bytes = &buf[pos..pos + range_count * RANGE_BYTES];
    pos += range_count * RANGE_BYTES;
    let ranges: Vec<Range> = range_bytes
        .par_chunks_exact(RANGE_BYTES)
        .map(|chunk| {
            let low_arr: [u8; FP_BYTES] = chunk[..FP_BYTES].try_into().expect("chunk is exactly FP_BYTES");
            let width_arr: [u8; FP_BYTES] = chunk[FP_BYTES..RANGE_BYTES].try_into().expect("chunk is exactly FP_BYTES");
            let low = Option::from(Fp::from_repr(low_arr)).expect("non-canonical Fp in full tree file");
            let width = Option::from(Fp::from_repr(width_arr)).expect("non-canonical Fp in full tree file");
            [low, width]
        })
        .collect();

    // Levels
    let mut levels: Vec<Vec<Fp>> = Vec::with_capacity(TREE_DEPTH);
    for _ in 0..TREE_DEPTH {
        let level_len = u64::from_le_bytes(
            read_bytes!(8).try_into().expect("level header is exactly 8 bytes"),
        ) as usize;
        let level_bytes = &buf[pos..pos + level_len * FP_BYTES];
        pos += level_len * FP_BYTES;
        let level: Vec<Fp> = level_bytes
            .par_chunks_exact(FP_BYTES)
            .map(|chunk| {
                let arr: [u8; FP_BYTES] = chunk.try_into().expect("chunk is exactly FP_BYTES");
                Option::from(Fp::from_repr(arr)).expect("non-canonical Fp in level data")
            })
            .collect();
        levels.push(level);
    }

    // Root
    let root_bytes: [u8; FP_BYTES] = buf[pos..pos + FP_BYTES].try_into()
        .map_err(|_| anyhow::anyhow!("unexpected EOF reading root"))?;
    let root = Option::from(Fp::from_repr(root_bytes))
        .expect("non-canonical Fp for root in full tree file");
    pos += FP_BYTES;

    // Height trailer (optional, backwards-compatible with old files)
    let height = if pos + 8 <= buf.len() {
        let h = u64::from_le_bytes(
            buf[pos..pos + 8].try_into().expect("height trailer is exactly 8 bytes"),
        );
        if h > 0 { Some(h) } else { None }
    } else {
        None
    };

    info!(
        range_count = ranges.len(),
        level_count = levels.len(),
        ?height,
        elapsed_s = format!("{:.1}", t1.elapsed().as_secs_f64()),
        "full tree parsed"
    );

    Ok((ranges, levels, root, height))
}
