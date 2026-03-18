pub mod hasher;
pub mod proof;
pub mod tree;
pub use proof::*;
pub use tree::*;

#[cfg(test)]
pub(crate) mod test_helpers;

use pasta_curves::Fp;

/// Convenience wrapper: Sinsemilla MerkleCRH hash of two field elements at a given level.
///
/// This is the same hash used for internal Merkle nodes in the IMT tree.
/// For leaf commitments, the tree uses paired raw values (low, high) rather
/// than explicit hashing — the level-0 Sinsemilla hash combines the pair.
pub fn sinsemilla_hash(level: usize, left: Fp, right: Fp) -> Fp {
    hasher::SinsemillaHasher::new().hash(level, left, right)
}
