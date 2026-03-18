use pasta_curves::Fp;

use crate::hasher::SinsemillaHasher;
use crate::tree::TREE_DEPTH;

/// Circuit-compatible IMT non-membership proof data.
///
/// Each field maps directly to a circuit witness:
///
/// - `root`: public input, checked against the IMT root in the instance column
/// - `low`, `width`: witnessed interval `(low, width)` pair
/// - `leaf_pos`: position of the `low` leaf (always even in paired-leaf model);
///   position bits determine swap ordering at each Merkle level
/// - `path`: sibling hashes for the 29-level Merkle authentication path
///
/// ## Paired-leaf model
///
/// In the paired-leaf tree, each range `[low, high]` occupies two adjacent
/// leaves: `low` at position `2k` and `high = low + width` at `2k + 1`.
/// Therefore:
/// - `path[0]` is always `high` (the level-0 sibling of the `low` leaf)
/// - `path[1..29]` are the standard Merkle siblings at levels 1-28
///
/// This matches the Orchard nullifier tree convention where `nf_start` is
/// always the left leaf and `nf_end` is extracted from `auth_path[0]`.
#[derive(Clone, Debug)]
pub struct ImtProofData {
    /// The Merkle root of the IMT.
    pub root: Fp,
    /// Interval start (low bound of the bracketing leaf).
    pub low: Fp,
    /// Interval width (`high - low`, pre-computed during tree construction).
    pub width: Fp,
    /// Position of the `low` leaf in the tree (always even).
    pub leaf_pos: u32,
    /// Sibling hashes along the 29-level Merkle path.
    /// `path[0]` = `high = low + width` (the paired sibling leaf).
    pub path: [Fp; TREE_DEPTH],
}

impl ImtProofData {
    /// Verify this proof out-of-circuit.
    ///
    /// Checks that `value` falls within `[low, low + width]` and that the
    /// Merkle path recomputes to `root` using Sinsemilla MerkleCRH.
    pub fn verify(&self, value: Fp) -> bool {
        // value - low <= width: if value < low, field subtraction wraps to a
        // huge value that exceeds any valid width, so the check fails correctly.
        let offset = value - self.low;
        if offset > self.width {
            return false;
        }
        let hasher = SinsemillaHasher::new();
        // In the paired-leaf model, the leaf is `low` directly (no hashing).
        let mut current = self.low;
        let mut pos = self.leaf_pos;
        for (level, sibling) in self.path.iter().enumerate() {
            let (l, r) = if pos & 1 == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = hasher.hash(level, l, r);
            pos >>= 1;
        }
        current == self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{fp, four_nullifiers};
    use crate::tree::{NullifierTree, TREE_DEPTH};

    #[test]
    fn test_proof_verify_rejects_wrong_value() {
        let tree = NullifierTree::build(four_nullifiers());

        let proof = tree.prove(fp(15)).unwrap();
        assert!(!proof.verify(fp(5)));
        assert!(!proof.verify(fp(10)));
    }

    #[test]
    fn test_proof_verify_rejects_wrong_root() {
        let tree = NullifierTree::build(four_nullifiers());

        let mut proof = tree.prove(fp(15)).unwrap();
        proof.root = Fp::zero();
        assert!(!proof.verify(fp(15)));
    }

    #[test]
    fn test_verify_rejects_tampered_auth_path_level_0() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let mut proof = tree.prove(value).unwrap();

        proof.path[0] += Fp::one();
        assert!(
            !proof.verify(value),
            "tampered auth_path[0] should fail verification"
        );
    }

    #[test]
    fn test_verify_rejects_tampered_auth_path_mid_level() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let mut proof = tree.prove(value).unwrap();

        let mid = TREE_DEPTH / 2;
        proof.path[mid] = Fp::zero();
        assert!(
            !proof.verify(value),
            "tampered auth_path[{}] should fail verification",
            mid
        );
    }

    #[test]
    fn test_verify_rejects_tampered_low() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let mut proof = tree.prove(value).unwrap();

        proof.low = Fp::from(999u64);
        assert!(
            !proof.verify(value),
            "tampered low bound should fail verification"
        );
    }

    #[test]
    fn test_verify_rejects_tampered_position() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let mut proof = tree.prove(value).unwrap();

        // In paired-leaf model, leaf_pos is 2*range_idx
        let original_pos = proof.leaf_pos;
        assert_eq!(original_pos % 2, 0, "leaf_pos should be even");

        proof.leaf_pos = original_pos + 2;
        assert!(!proof.verify(value), "wrong position should fail");

        proof.leaf_pos = original_pos.wrapping_sub(2);
        assert!(!proof.verify(value), "wrong position should fail");
    }

    #[test]
    fn test_verify_rejects_swapped_range_fields() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let mut proof = tree.prove(value).unwrap();

        let (low, width) = (proof.low, proof.width);
        proof.low = width;
        proof.width = low;
        assert!(
            !proof.verify(value),
            "swapped range fields should fail verification"
        );
    }

    #[test]
    fn test_path_0_equals_high() {
        let tree = NullifierTree::build(four_nullifiers());
        let value = fp(15);
        let proof = tree.prove(value).unwrap();

        // In paired-leaf model, path[0] should be high = low + width
        assert_eq!(
            proof.path[0],
            proof.low + proof.width,
            "path[0] should equal high (low + width)"
        );
    }

    #[test]
    fn test_leaf_pos_always_even() {
        let tree = NullifierTree::build(four_nullifiers());
        // Test proofs for values in different ranges
        for v in [fp(5), fp(15), fp(25), fp(35), fp(50)] {
            if let Some(proof) = tree.prove(v) {
                assert_eq!(
                    proof.leaf_pos % 2, 0,
                    "leaf_pos should always be even for value {:?}",
                    v
                );
            }
        }
    }
}
