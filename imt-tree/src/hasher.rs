use ff::PrimeFieldBits;
use pasta_curves::pallas;
use sinsemilla::HashDomain;

/// Personalization string for the Merkle CRH — matches Orchard exactly.
const MERKLE_CRH_PERSONALIZATION: &str = "z.cash:Orchard-MerkleCRH";

/// Number of bits per field element used in the Merkle hash input.
/// This is `pallas::Base::NUM_BITS` = 255.
const L_ORCHARD_MERKLE: usize = 255;

/// Number of bits used to encode the level in Sinsemilla MerkleCRH.
const K: usize = 10;

/// The "uncommitted" leaf value used for empty tree nodes.
/// Matches the Orchard protocol: <https://zips.z.cash/protocol/protocol.pdf#thmuncommittedorchard>
pub const UNCOMMITTED_ORCHARD: u64 = 2;

/// A Sinsemilla-based Merkle hasher compatible with Orchard's MerkleCRH.
///
/// Uses `sinsemilla::HashDomain` with the `"z.cash:Orchard-MerkleCRH"`
/// personalization. The hash input is `i2lebsp_k(level) || left || right`
/// where each field element contributes 255 bits.
///
/// This produces identical results to `MerkleHashOrchard::combine` from the
/// orchard crate, enabling the IMT tree to be verified by the existing vote
/// circuit without modification.
pub struct SinsemillaHasher {
    domain: HashDomain,
}

impl Default for SinsemillaHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl SinsemillaHasher {
    /// Create a new hasher with the Orchard MerkleCRH domain.
    pub fn new() -> Self {
        SinsemillaHasher {
            domain: HashDomain::new(MERKLE_CRH_PERSONALIZATION),
        }
    }

    /// Hash two field elements at a given tree level using Sinsemilla MerkleCRH.
    ///
    /// The input to the hash is:
    /// ```text
    /// i2lebsp_10(level) || le_bits(left)[0..255] || le_bits(right)[0..255]
    /// ```
    /// Total: 520 bits.
    ///
    /// Returns `Fp::zero()` if the hash produces the identity point (should
    /// never happen in practice with valid inputs).
    #[inline]
    pub fn hash(&self, level: usize, left: pallas::Base, right: pallas::Base) -> pallas::Base {
        let level_bits = i2lebsp_k(level);
        let left_bits = left.to_le_bits();
        let right_bits = right.to_le_bits();
        let bits = std::iter::empty()
            .chain(level_bits.iter().copied())
            .chain(left_bits.iter().by_vals().take(L_ORCHARD_MERKLE))
            .chain(right_bits.iter().by_vals().take(L_ORCHARD_MERKLE));
        self.domain.hash(bits).unwrap_or(pallas::Base::zero())
    }
}

/// Convert an integer to K bits in little-endian order.
///
/// Matches `i2lebsp_k` from the orchard crate's `constants::sinsemilla` module.
fn i2lebsp_k(int: usize) -> [bool; K] {
    assert!(int < (1 << K));
    let mut bits = [false; K];
    for (i, bit) in bits.iter_mut().enumerate() {
        *bit = (int >> i) & 1 == 1;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;

    #[test]
    fn test_sinsemilla_hasher_deterministic() {
        let h1 = SinsemillaHasher::new();
        let h2 = SinsemillaHasher::new();
        let a = pallas::Base::from(42u64);
        let b = pallas::Base::from(99u64);
        assert_eq!(h1.hash(0, a, b), h2.hash(0, a, b));
    }

    #[test]
    fn test_sinsemilla_hasher_level_matters() {
        let h = SinsemillaHasher::new();
        let a = pallas::Base::from(1u64);
        let b = pallas::Base::from(2u64);
        // Different levels should produce different hashes
        assert_ne!(h.hash(0, a, b), h.hash(1, a, b));
    }

    #[test]
    fn test_sinsemilla_hasher_order_matters() {
        let h = SinsemillaHasher::new();
        let a = pallas::Base::from(1u64);
        let b = pallas::Base::from(2u64);
        assert_ne!(h.hash(0, a, b), h.hash(0, b, a));
    }

    #[test]
    fn test_empty_leaf_hash() {
        let h = SinsemillaHasher::new();
        let empty = pallas::Base::from(UNCOMMITTED_ORCHARD);
        // Hashing two empty leaves at level 0 should be deterministic
        let result = h.hash(0, empty, empty);
        assert_ne!(result, pallas::Base::zero());
    }

    #[test]
    fn test_i2lebsp_k() {
        assert_eq!(i2lebsp_k(0), [false; 10]);
        assert_eq!(i2lebsp_k(1), {
            let mut bits = [false; 10];
            bits[0] = true;
            bits
        });
        assert_eq!(i2lebsp_k(1023), [true; 10]);
    }
}
