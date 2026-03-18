use super::*;
use crate::test_helpers::{fp, four_nullifiers};
use crate::hasher::SinsemillaHasher;

#[test]
fn test_build_ranges_from_4_nullifiers() {
    let ranges = build_nf_ranges(four_nullifiers());
    assert_eq!(ranges.len(), 5);

    assert_eq!(ranges[0], [fp(0), fp(9)]);       // width = 9 - 0 = 9
    assert_eq!(ranges[1], [fp(11), fp(8)]);      // width = 19 - 11 = 8
    assert_eq!(ranges[2], [fp(21), fp(8)]);      // width = 29 - 21 = 8
    assert_eq!(ranges[3], [fp(31), fp(8)]);      // width = 39 - 31 = 8
    // Last range: [41, MAX - 41]
    assert_eq!(ranges[4][0], fp(41));
    assert_eq!(ranges[4][1], Fp::one().neg() - fp(41));
}

#[test]
fn test_nullifiers_not_in_any_range() {
    let ranges = build_nf_ranges(four_nullifiers());
    for &nf in &four_nullifiers() {
        assert!(
            find_range_for_value(&ranges, nf).is_none(),
            "nullifier {:?} should not be in any gap range",
            nf
        );
    }
}

#[test]
fn test_non_nullifiers_found_in_ranges() {
    let ranges = build_nf_ranges(four_nullifiers());

    // Values in each gap
    assert_eq!(find_range_for_value(&ranges, fp(0)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(5)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(9)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(11)), Some(1));
    assert_eq!(find_range_for_value(&ranges, fp(15)), Some(1));
    assert_eq!(find_range_for_value(&ranges, fp(25)), Some(2));
    assert_eq!(find_range_for_value(&ranges, fp(35)), Some(3));
    assert_eq!(find_range_for_value(&ranges, fp(41)), Some(4));
    assert_eq!(find_range_for_value(&ranges, fp(1000)), Some(4));
}

#[test]
fn test_merkle_root_is_deterministic() {
    let tree1 = NullifierTree::build(four_nullifiers());
    let tree2 = NullifierTree::build(four_nullifiers());
    assert_eq!(tree1.root(), tree2.root());
}

#[test]
fn test_merkle_paths_verify_for_each_range() {
    let tree = NullifierTree::build(four_nullifiers());

    // Verify an exclusion proof for a value in every range
    let test_values = [fp(5), fp(15), fp(25), fp(35), fp(41)];
    for (i, &value) in test_values.iter().enumerate() {
        let proof = tree.prove(value).expect("should produce proof");
        assert_eq!(proof.leaf_pos, (i * 2) as u32); // paired leaves: 2*range_idx
        assert!(
            proof.verify(value),
            "exclusion proof for range {} does not verify",
            i
        );
    }
}

#[test]
fn test_exclusion_proof_end_to_end() {
    let tree = NullifierTree::build(four_nullifiers());

    // Prove that 15 is not a nullifier
    let value = fp(15);
    let proof = tree.prove(value).expect("should produce proof");
    assert_eq!(proof.leaf_pos, 2); // paired leaves: range 1 → pos 2*1=2

    assert_eq!(proof.low, fp(11));
    assert_eq!(proof.width, fp(8));   // width = 19 - 11 = 8
    assert!(proof.verify(value));
}

#[test]
fn test_nullifier_has_no_proof() {
    let tree = NullifierTree::build(four_nullifiers());
    for &nf in &four_nullifiers() {
        assert!(
            tree.prove(nf).is_none(),
            "nullifier {:?} should not have an exclusion proof",
            nf
        );
    }
}

#[test]
fn test_tree_len() {
    let tree = NullifierTree::build(four_nullifiers());
    assert_eq!(tree.len(), 5);
    assert!(!tree.is_empty());
}

#[test]
fn test_save_load_round_trip() {
    let tree = NullifierTree::build(four_nullifiers());
    let dir = std::env::temp_dir().join("imt_tree_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ranges.bin");

    tree.save(&path).unwrap();
    let loaded = NullifierTree::load(&path).unwrap();
    assert_eq!(tree.root(), loaded.root());
    assert_eq!(tree.ranges(), loaded.ranges());

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_save_load_full_round_trip() {
    let mut tree = NullifierTree::build(four_nullifiers());
    tree.set_height(2_000_000);
    let dir = std::env::temp_dir().join("imt_tree_test_full");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("full_tree.bin");

    tree.save_full(&path).unwrap();
    let loaded = NullifierTree::load_full(&path).unwrap();

    assert_eq!(tree.root(), loaded.root());
    assert_eq!(tree.ranges(), loaded.ranges());
    assert_eq!(tree.len(), loaded.len());
    assert_eq!(loaded.height(), Some(2_000_000));

    // Verify all level hashes match
    let original_leaves = tree.leaves();
    let loaded_leaves = loaded.leaves();
    assert_eq!(original_leaves, loaded_leaves);

    // Verify proofs still work on the loaded tree
    let value = fp(15);
    let proof = loaded.prove(value).unwrap();
    assert!(proof.verify(value));

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_save_load_full_no_height() {
    let tree = NullifierTree::build(four_nullifiers());
    assert_eq!(tree.height(), None);

    let dir = std::env::temp_dir().join("imt_tree_test_full_no_height");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("full_tree.bin");

    tree.save_full(&path).unwrap();
    let loaded = NullifierTree::load_full(&path).unwrap();
    assert_eq!(loaded.height(), None);
    assert_eq!(tree.root(), loaded.root());

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_unsorted_input_produces_same_tree() {
    let sorted = NullifierTree::build(four_nullifiers());
    let unsorted = NullifierTree::build(vec![fp(30), fp(10), fp(40), fp(20)]);
    assert_eq!(sorted.root(), unsorted.root());
}

#[test]
fn test_precompute_empty_hashes_chain() {
    let hasher = SinsemillaHasher::new();
    let empty_leaf = Fp::from(crate::hasher::UNCOMMITTED_ORCHARD);
    let empty = precompute_empty_hashes();

    assert_eq!(empty[0], hasher.hash(0, empty_leaf, empty_leaf));

    for i in 1..TREE_DEPTH {
        let expected = hasher.hash(i, empty[i - 1], empty[i - 1]);
        assert_eq!(
            empty[i], expected,
            "empty hash mismatch at level {}",
            i
        );
    }
}

#[test]
fn test_build_levels_consistency() {
    let hasher = SinsemillaHasher::new();
    let tree = NullifierTree::build(four_nullifiers());

    for i in 0..TREE_DEPTH - 1 {
        let prev = &tree.levels[i];
        let next = &tree.levels[i + 1];
        let pairs = prev.len() / 2;
        for j in 0..pairs {
            let expected = hasher.hash(i, prev[j * 2], prev[j * 2 + 1]);
            assert_eq!(
                next[j], expected,
                "level {} node {} does not match hash of level {} children",
                i + 1, j, i
            );
        }
    }

    let top = &tree.levels[TREE_DEPTH - 1];
    let expected_root = hasher.hash(TREE_DEPTH - 1, top[0], top[1]);
    assert_eq!(tree.root(), expected_root);
}

#[test]
fn test_leaves_accessor() {
    let tree = NullifierTree::build(four_nullifiers());
    let leaves = tree.leaves();
    // Paired leaves: 5 ranges × 2 leaves each = 10
    assert_eq!(leaves.len(), 10);
    let expected = expand_ranges(tree.ranges());
    assert_eq!(leaves, expected.as_slice());
}

#[test]
fn test_find_range_empty_ranges() {
    let ranges: Vec<Range> = vec![];
    assert_eq!(find_range_for_value(&ranges, fp(0)), None);
    assert_eq!(find_range_for_value(&ranges, fp(42)), None);
}

#[test]
fn test_find_range_single_range() {
    let ranges = build_nf_ranges(vec![fp(100)]);
    assert_eq!(ranges.len(), 2);

    assert_eq!(find_range_for_value(&ranges, fp(0)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(99)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(100)), None);
    assert_eq!(find_range_for_value(&ranges, fp(101)), Some(1));
    assert_eq!(find_range_for_value(&ranges, fp(999)), Some(1));
}

#[test]
fn test_find_range_exact_boundaries() {
    let ranges = build_nf_ranges(four_nullifiers());
    assert_eq!(find_range_for_value(&ranges, fp(0)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(11)), Some(1));
    assert_eq!(find_range_for_value(&ranges, fp(21)), Some(2));
    assert_eq!(find_range_for_value(&ranges, fp(31)), Some(3));
    assert_eq!(find_range_for_value(&ranges, fp(41)), Some(4));

    assert_eq!(find_range_for_value(&ranges, fp(9)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(19)), Some(1));
    assert_eq!(find_range_for_value(&ranges, fp(29)), Some(2));
    assert_eq!(find_range_for_value(&ranges, fp(39)), Some(3));
}

#[test]
fn test_find_range_consecutive_nullifiers() {
    let ranges = build_nf_ranges(vec![fp(10), fp(11), fp(12)]);
    assert_eq!(ranges.len(), 2);

    assert_eq!(find_range_for_value(&ranges, fp(5)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(9)), Some(0));
    assert_eq!(find_range_for_value(&ranges, fp(10)), None);
    assert_eq!(find_range_for_value(&ranges, fp(11)), None);
    assert_eq!(find_range_for_value(&ranges, fp(12)), None);
    assert_eq!(find_range_for_value(&ranges, fp(13)), Some(1));
}

#[test]
fn test_find_range_binary_search_large_set() {
    let nullifiers: Vec<Fp> = (0..10_000u64).map(|i| fp(i * 3 + 1)).collect();
    let ranges = build_nf_ranges(nullifiers.clone());

    for nf in &nullifiers {
        assert!(find_range_for_value(&ranges, *nf).is_none());
    }

    for (i, window) in nullifiers.windows(2).enumerate() {
        let mid = window[0] + Fp::one();
        let result = find_range_for_value(&ranges, mid);
        assert!(
            result.is_some(),
            "mid-gap value between nf[{}] and nf[{}] not found",
            i,
            i + 1
        );
        let idx = result.unwrap();
        let [low, width] = ranges[idx];
        let offset = mid - low;
        assert!(
            offset <= width,
            "value not within returned range at index {}",
            idx
        );
    }
}

#[test]
fn test_find_range_agrees_with_linear_scan() {
    fn linear_find(ranges: &[Range], value: Fp) -> Option<usize> {
        for (i, [low, width]) in ranges.iter().enumerate() {
            let offset = value - *low;
            if offset <= *width {
                return Some(i);
            }
        }
        None
    }

    let nullifiers: Vec<Fp> = (0..500u64).map(|i| fp(i * 7 + 3)).collect();
    let ranges = build_nf_ranges(nullifiers);

    for v in 0..4000u64 {
        let val = fp(v);
        assert_eq!(
            find_range_for_value(&ranges, val),
            linear_find(&ranges, val),
            "disagreement at value {}",
            v
        );
    }
}

// -- Tree behavior at different scales ------------------------------------

#[test]
fn test_single_nullifier_tree() {
    let tree = NullifierTree::build(vec![fp(100)]);
    assert_eq!(tree.len(), 2);

    let ranges = tree.ranges();
    assert_eq!(ranges[0], [fp(0), fp(99)]);       // width = 99 - 0 = 99
    assert_eq!(ranges[1][0], fp(101));
    assert_eq!(ranges[1][1], Fp::one().neg() - fp(101)); // width = MAX - 101

    let proof_low = tree.prove(fp(50)).unwrap();
    assert_eq!(proof_low.leaf_pos, 0); // paired: range 0 → pos 0
    assert!(proof_low.verify(fp(50)));

    let proof_high = tree.prove(fp(200)).unwrap();
    assert_eq!(proof_high.leaf_pos, 2); // paired: range 1 → pos 2
    assert!(proof_high.verify(fp(200)));

    assert!(tree.prove(fp(100)).is_none());
}

#[test]
fn test_consecutive_nullifiers_collapse_gap() {
    let tree = NullifierTree::build(vec![fp(5), fp(6), fp(7)]);

    assert_eq!(tree.len(), 2);
    assert_eq!(tree.ranges()[0], [fp(0), fp(4)]);   // width = 4 - 0 = 4
    assert_eq!(tree.ranges()[1][0], fp(8));

    assert!(tree.prove(fp(2)).unwrap().verify(fp(2)));
    assert!(tree.prove(fp(100)).unwrap().verify(fp(100)));

    for nf in [5u64, 6, 7] {
        assert!(tree.prove(fp(nf)).is_none(), "nullifier {} should have no proof", nf);
    }
}

#[test]
fn test_adjacent_nullifiers_differ_by_one() {
    let tree = NullifierTree::build(vec![fp(5), fp(6)]);

    assert_eq!(tree.len(), 2);
    assert_eq!(tree.ranges()[0], [fp(0), fp(4)]);   // width = 4 - 0 = 4
    assert_eq!(tree.ranges()[1][0], fp(7));

    assert!(tree.prove(fp(4)).unwrap().verify(fp(4)));
    assert!(tree.prove(fp(7)).unwrap().verify(fp(7)));
    assert!(tree.prove(fp(5)).is_none());
    assert!(tree.prove(fp(6)).is_none());
}

#[test]
fn test_nullifier_at_zero() {
    let tree = NullifierTree::build(vec![Fp::zero()]);
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.ranges()[0][0], fp(1));
    assert_eq!(tree.ranges()[0][1], Fp::one().neg() - fp(1)); // width = MAX - 1

    assert!(tree.prove(Fp::zero()).is_none());
    assert!(tree.prove(fp(1)).unwrap().verify(fp(1)));
    assert!(tree.prove(fp(1000)).unwrap().verify(fp(1000)));
}

#[test]
fn test_nullifier_at_zero_and_one() {
    let tree = NullifierTree::build(vec![Fp::zero(), fp(1)]);
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.ranges()[0][0], fp(2));

    assert!(tree.prove(Fp::zero()).is_none());
    assert!(tree.prove(fp(1)).is_none());
    assert!(tree.prove(fp(2)).unwrap().verify(fp(2)));
}

#[test]
fn test_larger_tree_200_nullifiers() {
    let nullifiers: Vec<Fp> = (1..=200u64).map(|i| fp(i * 1000)).collect();
    let tree = NullifierTree::build(nullifiers.clone());

    assert_eq!(tree.len(), 201);

    let test_indices = [0usize, 1, 50, 100, 150, 199, 200];
    for &idx in &test_indices {
        let range = tree.ranges()[idx];
        let value = range[0];
        let proof = tree.prove(value).unwrap();
        assert_eq!(proof.leaf_pos, (idx * 2) as u32); // paired leaves
        assert!(proof.verify(value), "proof at leaf index {} does not verify", idx);
    }

    for nf in &nullifiers {
        assert!(tree.prove(*nf).is_none());
    }
}

#[test]
fn test_larger_tree_different_sizes_have_different_roots() {
    let tree_100 = NullifierTree::build((1..=100u64).map(fp));
    let tree_200 = NullifierTree::build((1..=200u64).map(fp));
    assert_ne!(tree_100.root(), tree_200.root());
}

#[test]
fn test_duplicate_nullifiers_produce_same_tree() {
    let with_dups = NullifierTree::build(vec![fp(10), fp(10), fp(20), fp(20), fp(30)]);
    let without_dups = NullifierTree::build(vec![fp(10), fp(20), fp(30)]);
    assert_eq!(with_dups.root(), without_dups.root());
    assert_eq!(with_dups.ranges(), without_dups.ranges());
}

// ================================================================
// End-to-end sentinel tree + circuit-compatible proof tests
// ================================================================

#[test]
fn test_sentinel_tree_all_ranges_under_2_250() {
    let tree = build_sentinel_tree(&[]).unwrap();

    // Directly check each width against the same constraint used
    // by verify_range_widths: byte 31 of the LE repr < 0x04 means
    // the value fits in 250 bits (bits 250-255 are zero).
    for (i, [low, width]) in tree.ranges().iter().enumerate() {
        let repr = width.to_repr();
        assert!(
            repr.as_ref()[31] < 0x04,
            "range {} has width >= 2^250: low={:?}, width={:?}",
            i, low, width
        );
    }

    // Also verify via the production method.
    tree.verify_range_widths().expect("sentinel tree should pass width verification");
}

#[test]
fn test_sentinel_tree_with_extra_nullifiers() {
    let extras = vec![fp(42), fp(1000000), fp(999999999)];
    let tree = build_sentinel_tree(&extras).unwrap();

    for nf in &extras {
        assert!(tree.prove(*nf).is_none(), "nullifier should be excluded");
    }

    let proof = tree.prove(fp(43)).unwrap();
    assert!(proof.verify(fp(43)));
}

#[test]
fn test_proof_fields_match_tree() {
    let tree = build_sentinel_tree(&[fp(42), fp(100)]).unwrap();
    let value = fp(50);

    let proof = tree.prove(value).expect("value should be in a gap");
    assert_eq!(proof.root, tree.root());
    assert_eq!(proof.path.len(), TREE_DEPTH);
    assert!(proof.verify(value));
}

#[test]
fn test_proof_rejects_wrong_value() {
    let tree = build_sentinel_tree(&[fp(42), fp(100)]).unwrap();
    let value = fp(50);
    let proof = tree.prove(value).expect("value should be in a gap");

    assert!(!proof.verify(fp(42)), "nullifier should not verify");
    assert!(!proof.verify(fp(100)), "nullifier should not verify");
}

#[test]
fn test_e2e_sentinel_tree_proof_gen_and_verify() {
    let extra_nfs = vec![fp(12345), fp(67890), fp(111111)];
    let tree = build_sentinel_tree(&extra_nfs).unwrap();

    let test_value = fp(50000);
    assert!(tree.prove(test_value).is_some(), "test value should be in a gap range");

    let proof = tree.prove(test_value).unwrap();
    assert!(proof.verify(test_value));
    assert_eq!(proof.path.len(), TREE_DEPTH);

    let tree2 = build_sentinel_tree(&extra_nfs).unwrap();
    assert_eq!(tree.root(), tree2.root());
}

#[test]
fn test_pir_tree_root_matches_full_tree() {
    // The PIR tree (pre-hashed leaves, depth 26, level_offset=1) extended
    // to depth 29 should produce the same root as the full paired-leaf tree
    // (raw leaves, depth 29, level_offset=0).
    let nfs = four_nullifiers();
    let mut sorted = nfs.clone();
    sorted.sort();
    let ranges = build_nf_ranges(sorted);

    // Full tree
    let leaves_full = expand_ranges(&ranges);
    let empty = precompute_empty_hashes();
    let (root29_full, levels_full) = build_levels(leaves_full, &empty, TREE_DEPTH, 0);

    // PIR tree
    let leaves_pir = commit_ranges(&ranges);
    let (root26_pir, levels_pir) = build_levels(leaves_pir, &empty, 26, 1);

    // PIR level k should match full level k+1
    let max_check = levels_pir.len().min(levels_full.len() - 1);
    for k in 0..max_check {
        let pir_len = levels_pir[k].len();
        let full_len = levels_full[k + 1].len();
        for i in 0..pir_len.min(full_len) {
            assert_eq!(
                levels_pir[k][i], levels_full[k + 1][i],
                "PIR level {} node {} != full level {} node {}",
                k, i, k + 1, i
            );
        }
        assert_eq!(
            pir_len, full_len,
            "PIR level {} len {} != full level {} len {}",
            k, pir_len, k + 1, full_len
        );
    }

    // Extend PIR root to depth 29. PIR root at Sinsemilla level 26 = full
    // tree level 27. Need 2 more hashes at Sinsemilla levels 27 and 28.
    let hasher = SinsemillaHasher::new();
    let mut root = root26_pir;
    for level in 27..TREE_DEPTH {
        root = hasher.hash(level, root, empty[level - 1]);
    }

    assert_eq!(
        root, root29_full,
        "PIR extended root must match full tree root"
    );
}

#[test]
fn test_empty_hashes_match_sinsemilla_convention() {
    let hasher = SinsemillaHasher::new();
    let empty_leaf = Fp::from(crate::hasher::UNCOMMITTED_ORCHARD);
    let empty = precompute_empty_hashes();
    assert_eq!(empty[0], hasher.hash(0, empty_leaf, empty_leaf));

    for i in 1..TREE_DEPTH {
        assert_eq!(empty[i], hasher.hash(i, empty[i - 1], empty[i - 1]));
    }
}
