//! Integration test: full PIR round-trip without YPIR.
//!
//! Builds a depth-26 tree from synthetic nullifiers, exports tier data,
//! parses it back, constructs proofs, and verifies them.

use ff::{Field, PrimeField as _};
use pasta_curves::Fp;

use imt_tree::hasher::SinsemillaHasher;
use imt_tree::tree::{build_sentinel_tree, TREE_DEPTH};
use imt_tree::ImtProofData;

use pir_export::tier0::Tier0Data;
use pir_export::tier1::Tier1Row;
use pir_export::tier2::Tier2Row;
use pir_export::{
    build_pir_tree, build_ranges_with_sentinels, PIR_DEPTH, TIER0_LAYERS, TIER1_LAYERS,
    TIER1_LEAVES, TIER1_ROW_BYTES, TIER2_LEAVES, TIER2_ROW_BYTES,
};

/// Perform local proof construction from tier data (mirrors pir_client::fetch_proof_local).
fn construct_proof(
    tier0_data: &[u8],
    tier1_data: &[u8],
    tier2_data: &[u8],
    num_ranges: usize,
    value: Fp,
    empty_hashes: &[Fp; TREE_DEPTH],
    root29: Fp,
) -> Option<ImtProofData> {
    let hasher = SinsemillaHasher::new();
    let tier0 = Tier0Data::from_bytes(tier0_data.to_vec()).ok()?;

    let s1 = tier0.find_subtree(value)?;

    let mut path = [Fp::default(); TREE_DEPTH];

    // +1 shift: PIR level k = full tree level k+1 (paired-leaf model)
    let tier0_siblings = tier0.extract_siblings(s1);
    for (i, &sib) in tier0_siblings.iter().enumerate() {
        path[PIR_DEPTH - TIER0_LAYERS + 1 + i] = sib;
    }

    // Tier 1: direct row lookup
    let t1_offset = s1 * TIER1_ROW_BYTES;
    let tier1_row = &tier1_data[t1_offset..t1_offset + TIER1_ROW_BYTES];
    let tier1 = Tier1Row::from_bytes(tier1_row).ok()?;

    let s2 = tier1.find_sub_subtree(value)?;

    let tier1_siblings = tier1.extract_siblings(s2);
    for (i, &sib) in tier1_siblings.iter().enumerate() {
        path[PIR_DEPTH - TIER0_LAYERS - TIER1_LAYERS + 1 + i] = sib;
    }

    // Tier 2: direct row lookup
    let t2_row_idx = s1 * TIER1_LEAVES + s2;
    let t2_offset = t2_row_idx * TIER2_ROW_BYTES;
    let tier2_row = &tier2_data[t2_offset..t2_offset + TIER2_ROW_BYTES];
    let tier2 = Tier2Row::from_bytes(tier2_row).ok()?;
    let valid_leaves = num_ranges.saturating_sub(t2_row_idx * TIER2_LEAVES).min(TIER2_LEAVES);

    let leaf_idx = tier2.find_leaf(value, valid_leaves)?;

    // path[0]: the paired-leaf sibling = high = low + width of the SAME range.
    // In the paired-leaf model, low is at position 2k and high is at 2k+1.
    // path[0] is always the level-0 sibling of the low leaf.
    let (low, width) = tier2.leaf_record(leaf_idx);
    path[0] = low + width;

    // path[1..9]: tier-2 internal node siblings (shifted by 1)
    let tier2_siblings = tier2.extract_siblings(leaf_idx, valid_leaves, &hasher);
    for (i, &sib) in tier2_siblings.iter().enumerate() {
        path[1 + i] = sib;
    }

    // Path padding (depth 26 → 29), shifted by 1
    // path[level] is a sibling at full tree level `level`.
    // Empty subtree at tree level L = empty_hashes[L-1].
    for level in (PIR_DEPTH + 1)..TREE_DEPTH {
        path[level] = empty_hashes[level - 1];
    }

    // Global leaf position in full paired-leaf tree: range k → pos 2k
    let global_range_idx = t2_row_idx * TIER2_LEAVES + leaf_idx;
    let leaf_pos = (global_range_idx * 2) as u32;
    let (low, width) = tier2.leaf_record(leaf_idx);

    Some(ImtProofData {
        root: root29,
        low,
        width,
        leaf_pos,
        path,
    })
}

#[test]
fn test_small_tree_round_trip() {
    // Build a small tree with 100 nullifiers
    let mut rng = rand::thread_rng();
    let raw_nfs: Vec<Fp> = (0..100).map(|_| Fp::random(&mut rng)).collect();
    let ranges = build_ranges_with_sentinels(&raw_nfs);

    eprintln!("  Ranges: {}", ranges.len());

    let tree = build_pir_tree(ranges.clone()).unwrap();
    eprintln!("  Root26: {}", hex::encode(tree.root26.to_repr()));
    eprintln!("  Root29: {}", hex::encode(tree.root29.to_repr()));

    // Export tier data
    let tier0_data =
        pir_export::tier0::export(&tree.root26, &tree.levels, &tree.ranges, &tree.empty_hashes);

    let mut tier1_data = Vec::new();
    pir_export::tier1::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier1_data,
    )
    .unwrap();

    let mut tier2_data = Vec::new();
    pir_export::tier2::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier2_data,
    )
    .unwrap();

    eprintln!("  Tier sizes: {} / {} / {}", tier0_data.len(), tier1_data.len(), tier2_data.len());

    // Test multiple values
    let mut passed = 0;
    for &[low, _width] in ranges.iter().take(20) {
        // Query with the low value of each range (guaranteed to be in-range)
        let value = low;
        let proof = construct_proof(
            &tier0_data,
            &tier1_data,
            &tier2_data,
            ranges.len(),
            value,
            &tree.empty_hashes,
            tree.root29,
        );

        match proof {
            Some(p) => {
                assert!(
                    p.verify(value),
                    "Proof failed verification for value {}",
                    hex::encode(value.to_repr())
                );
                passed += 1;
            }
            None => {
                panic!(
                    "Failed to construct proof for value {} (low of a valid range)",
                    hex::encode(value.to_repr())
                );
            }
        }
    }

    eprintln!("  {} proofs passed", passed);
}

#[test]
fn test_root_extension_matches_depth29() {
    // Build trees with the same nullifiers and verify roots match
    let mut rng = rand::thread_rng();
    let raw_nfs: Vec<Fp> = (0..50).map(|_| Fp::random(&mut rng)).collect();

    // Depth-29 tree via existing system
    let tree29 = build_sentinel_tree(&raw_nfs).unwrap();
    let root29_existing = tree29.root();

    // Depth-26 PIR tree with root extension
    let ranges = build_ranges_with_sentinels(&raw_nfs);
    let tree26 = build_pir_tree(ranges).unwrap();
    let root29_extended = tree26.root29;

    eprintln!(
        "  Existing root29:  {}",
        hex::encode(root29_existing.to_repr())
    );
    eprintln!(
        "  Extended root29:  {}",
        hex::encode(root29_extended.to_repr())
    );

    assert_eq!(
        root29_existing, root29_extended,
        "Depth-29 root from existing tree must match PIR extended root"
    );
}

#[test]
fn test_pir_proof_matches_existing_prove() {
    // Verify PIR proofs produce the same (low, width) as NullifierTree::prove()
    let mut rng = rand::thread_rng();
    let raw_nfs: Vec<Fp> = (0..200).map(|_| Fp::random(&mut rng)).collect();

    let tree29 = build_sentinel_tree(&raw_nfs).unwrap();

    let ranges = build_ranges_with_sentinels(&raw_nfs);
    let tree = build_pir_tree(ranges.clone()).unwrap();

    let tier0_data =
        pir_export::tier0::export(&tree.root26, &tree.levels, &tree.ranges, &tree.empty_hashes);
    let mut tier1_data = Vec::new();
    pir_export::tier1::export(&tree.levels, &tree.ranges, &tree.empty_hashes, &mut tier1_data)
        .unwrap();
    let mut tier2_data = Vec::new();
    pir_export::tier2::export(&tree.levels, &tree.ranges, &tree.empty_hashes, &mut tier2_data)
        .unwrap();

    for (range_i, &[low, _]) in ranges.iter().take(50).enumerate() {
        let value = low;

        let proof_existing = tree29.prove(value).expect("existing prove failed");
        let proof_pir = construct_proof(
            &tier0_data,
            &tier1_data,
            &tier2_data,
            ranges.len(),
            value,
            &tree.empty_hashes,
            tree.root29,
        )
        .expect("PIR proof construction failed");

        assert_eq!(proof_existing.low, proof_pir.low, "low mismatch at range {}", range_i);
        assert_eq!(proof_existing.width, proof_pir.width, "width mismatch at range {}", range_i);
        assert_eq!(proof_existing.leaf_pos, proof_pir.leaf_pos, "leaf_pos mismatch at range {}", range_i);
        assert_eq!(proof_existing.root, proof_pir.root, "root mismatch at range {}", range_i);

        // Compare paths element by element
        for (level, (e, p)) in proof_existing.path.iter().zip(proof_pir.path.iter()).enumerate() {
            if e != p {
                eprintln!(
                    "  range {}: path[{}] mismatch: existing={} pir={}",
                    range_i, level, hex::encode(e.to_repr()), hex::encode(p.to_repr())
                );
            }
        }
        assert_eq!(proof_existing.path, proof_pir.path, "path mismatch at range {}", range_i);

        assert!(proof_pir.verify(value), "PIR proof verification failed at range {}", range_i);
    }
}

/// Test the `build_and_export` convenience function (used by the serve rebuild path).
///
/// This exercises the full pipeline: sort, sentinel injection, tree build, and
/// tier file export to disk. Verifies the output files exist and the metadata
/// records the correct height.
#[test]
fn test_build_and_export_writes_files() {
    let dir = std::env::temp_dir().join(format!(
        "pir_build_export_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let nfs: Vec<Fp> = (1u64..=50).map(|i| Fp::from(i * 997)).collect();
    let tree = pir_export::build_and_export(nfs, &dir, Some(2_800_000)).unwrap();

    // Verify files exist
    assert!(dir.join("tier0.bin").exists());
    assert!(dir.join("tier1.bin").exists());
    assert!(dir.join("tier2.bin").exists());
    assert!(dir.join("pir_root.json").exists());

    // Verify metadata
    let meta: pir_export::PirMetadata =
        serde_json::from_str(&std::fs::read_to_string(dir.join("pir_root.json")).unwrap())
            .unwrap();
    assert_eq!(meta.height, Some(2_800_000));
    assert_eq!(meta.pir_depth, pir_export::PIR_DEPTH);
    assert_eq!(meta.root29, hex::encode(tree.root29.to_repr()));
    assert!(meta.num_ranges > 50); // sentinels add extra ranges

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test that `build_and_export` with a subset of nullifiers produces a different
/// root than the full set, and that the exported tier files are valid.
///
/// This verifies the target-height export path: when the export pipeline operates
/// on a subset of nullifiers (as it does with `--target-height`), the resulting
/// tree has a distinct, self-consistent root.
#[test]
fn test_subset_export_produces_different_root() {
    let raw_nfs: Vec<Fp> = (1u64..=200).map(|i| Fp::from(i * 7919)).collect();

    // Build tree from full set
    let full_ranges = build_ranges_with_sentinels(&raw_nfs);
    let full_tree = build_pir_tree(full_ranges.clone()).unwrap();

    // Build tree from first half (simulates export at a lower target height)
    let half = raw_nfs.len() / 2;
    let subset_ranges = build_ranges_with_sentinels(&raw_nfs[..half]);
    let subset_tree = build_pir_tree(subset_ranges.clone()).unwrap();

    // Roots must differ (different nullifier sets produce different trees)
    assert_ne!(
        full_tree.root29, subset_tree.root29,
        "subset root must differ from full root"
    );

    // Export the subset tree and verify it round-trips correctly
    let tier0_data = pir_export::tier0::export(
        &subset_tree.root26,
        &subset_tree.levels,
        &subset_tree.ranges,
        &subset_tree.empty_hashes,
    );
    let mut tier1_data = Vec::new();
    pir_export::tier1::export(
        &subset_tree.levels,
        &subset_tree.ranges,
        &subset_tree.empty_hashes,
        &mut tier1_data,
    )
    .unwrap();
    let mut tier2_data = Vec::new();
    pir_export::tier2::export(
        &subset_tree.levels,
        &subset_tree.ranges,
        &subset_tree.empty_hashes,
        &mut tier2_data,
    )
    .unwrap();

    // Verify proofs for the subset tree work
    for &[low, _] in subset_ranges.iter().take(20) {
        let proof = construct_proof(
            &tier0_data,
            &tier1_data,
            &tier2_data,
            subset_ranges.len(),
            low,
            &subset_tree.empty_hashes,
            subset_tree.root29,
        )
        .expect("subset proof construction failed");
        assert!(proof.verify(low), "subset proof verification failed");
    }
}

/// Test that tier export is deterministic: exporting the same tree
/// twice produces byte-identical output.
#[test]
fn test_export_deterministic() {
    let raw_nfs: Vec<Fp> = (1u64..=100).map(|i| Fp::from(i * 1013)).collect();
    let ranges = build_ranges_with_sentinels(&raw_nfs);
    let tree = build_pir_tree(ranges).unwrap();

    // Export tier1 twice
    let mut tier1_a = Vec::new();
    pir_export::tier1::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier1_a,
    )
    .unwrap();

    let mut tier1_b = Vec::new();
    pir_export::tier1::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier1_b,
    )
    .unwrap();

    assert_eq!(
        tier1_a, tier1_b,
        "tier1 parallel export must be deterministic"
    );

    // Export tier2 twice
    let mut tier2_a = Vec::new();
    pir_export::tier2::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier2_a,
    )
    .unwrap();

    let mut tier2_b = Vec::new();
    pir_export::tier2::export(
        &tree.levels,
        &tree.ranges,
        &tree.empty_hashes,
        &mut tier2_b,
    )
    .unwrap();

    assert_eq!(
        tier2_a, tier2_b,
        "tier2 parallel export must be deterministic"
    );
}

#[test]
fn test_tier0_binary_search() {
    let raw_nfs: Vec<Fp> = (1u64..=50).map(|i| Fp::from(i * 1000)).collect();
    let ranges = build_ranges_with_sentinels(&raw_nfs);
    let tree = build_pir_tree(ranges.clone()).unwrap();

    let tier0_data =
        pir_export::tier0::export(&tree.root26, &tree.levels, &tree.ranges, &tree.empty_hashes);
    let tier0 = Tier0Data::from_bytes(tier0_data).unwrap();

    // Test that values within ranges are found
    for &[low, _width] in ranges.iter().take(10) {
        let result = tier0.find_subtree(low);
        assert!(result.is_some(), "find_subtree failed for low={:?}", low);
    }
}
