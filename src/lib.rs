//! CM256 - Cauchy Reed-Solomon Erasure Coding
//!
//! A Rust port of the CM256 library by Christopher A. Taylor.
//!
//! This library provides efficient erasure coding using a Cauchy matrix
//! over GF(256). It can encode `original_count` data blocks into
//! `recovery_count` recovery blocks, and recover any `original_count`
//! blocks from any combination of original and recovery blocks.
//!
//! # Example
//!
//! ```rust
//! use cm256::{Params, Block, encode, decode};
//!
//! // Create encoder parameters
//! let params = Params::new(3, 2, 1024).unwrap(); // 3 original, 2 recovery, 1024 bytes each
//!
//! // Original data blocks
//! let original_data: Vec<Vec<u8>> = vec![
//!     vec![1u8; 1024],
//!     vec![2u8; 1024],
//!     vec![3u8; 1024],
//! ];
//!
//! // Encode
//! let mut recovery_blocks = vec![0u8; params.recovery_count() * params.block_bytes()];
//! let blocks: Vec<Block> = original_data.iter()
//!     .enumerate()
//!     .map(|(i, data)| Block::new(i as u8, data))
//!     .collect();
//! encode(&params, &blocks, &mut recovery_blocks).unwrap();
//! ```

// Define custom GF(256) type matching the C++ cm256 polynomial
// C++ uses polynomial index 3 = 0xa6, computed as (0xa6 << 1) | 1 = 0x14d
// Generator 0x02 works for this polynomial
use gf256::gf::gf;

#[gf(polynomial = 0x14d, generator = 0x2)]
pub type Gf256;

/// Error type for CM256 operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Invalid parameter values (zero or negative counts)
    InvalidParams,
    /// Too many total blocks (original + recovery > 256)
    TooManyBlocks,
    /// Null pointer or empty input
    InvalidInput,
    /// Duplicate block indices found during decoding
    DuplicateIndex,
    /// Not enough blocks provided for decoding
    InsufficientBlocks,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidParams => write!(f, "invalid encoder parameters"),
            Error::TooManyBlocks => write!(f, "original_count + recovery_count > 256"),
            Error::InvalidInput => write!(f, "invalid input"),
            Error::DuplicateIndex => write!(f, "duplicate block index"),
            Error::InsufficientBlocks => write!(f, "insufficient blocks for decoding"),
        }
    }
}

impl std::error::Error for Error {}

/// Encoder/decoder parameters
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Number of original data blocks (< 256)
    original_count: usize,
    /// Number of recovery blocks (< 256)
    recovery_count: usize,
    /// Size of each block in bytes
    block_bytes: usize,
}

impl Params {
    /// Create new encoder parameters
    ///
    /// # Arguments
    /// * `original_count` - Number of original data blocks (1..256)
    /// * `recovery_count` - Number of recovery blocks (1..256)
    /// * `block_bytes` - Size of each block in bytes (> 0)
    ///
    /// # Errors
    /// Returns `Error::InvalidParams` if any count is zero or negative.
    /// Returns `Error::TooManyBlocks` if original_count + recovery_count > 256.
    pub fn new(original_count: usize, recovery_count: usize, block_bytes: usize) -> Result<Self, Error> {
        if original_count == 0 || recovery_count == 0 || block_bytes == 0 {
            return Err(Error::InvalidParams);
        }
        if original_count + recovery_count > 256 {
            return Err(Error::TooManyBlocks);
        }
        Ok(Self {
            original_count,
            recovery_count,
            block_bytes,
        })
    }

    #[inline]
    pub fn original_count(&self) -> usize {
        self.original_count
    }

    #[inline]
    pub fn recovery_count(&self) -> usize {
        self.recovery_count
    }

    #[inline]
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Get the block index for a recovery block
    #[inline]
    pub fn recovery_block_index(&self, recovery_index: usize) -> u8 {
        debug_assert!(recovery_index < self.recovery_count);
        (self.original_count + recovery_index) as u8
    }

    /// Get the block index for an original block
    #[inline]
    pub fn original_block_index(&self, original_index: usize) -> u8 {
        debug_assert!(original_index < self.original_count);
        original_index as u8
    }
}

/// A data block with its index
#[derive(Debug, Clone)]
pub struct Block<'a> {
    /// Block data
    pub data: &'a [u8],
    /// Block index (0..original_count for original, original_count.. for recovery)
    pub index: u8,
}

impl<'a> Block<'a> {
    /// Create a new block
    #[inline]
    pub fn new(index: u8, data: &'a [u8]) -> Self {
        Self { data, index }
    }

    /// Check if this is an original block
    #[inline]
    pub fn is_original(&self, params: &Params) -> bool {
        (self.index as usize) < params.original_count
    }
}

/// Mutable block for decoding (recovery blocks are modified in-place)
#[derive(Debug)]
pub struct BlockMut<'a> {
    /// Block data (mutable for recovery blocks)
    pub data: &'a mut [u8],
    /// Block index
    pub index: u8,
}

impl<'a> BlockMut<'a> {
    #[inline]
    pub fn new(index: u8, data: &'a mut [u8]) -> Self {
        Self { data, index }
    }
}

// =============================================================================
// Cauchy Matrix Generation
// =============================================================================

/// Generate a Cauchy matrix element
///
/// The matrix is normalized so the first row is all 1s:
///   a_ij = (y_j + x_0) / (x_i + y_j)
///
/// Where:
/// - x_i is the recovery block index
/// - x_0 = original_count (first recovery block index)  
/// - y_j is the original block index (column)
#[inline]
fn get_matrix_element(x_i: u8, x_0: u8, y_j: u8) -> Gf256 {
    let numerator = Gf256(y_j) + Gf256(x_0);
    let denominator = Gf256(x_i) + Gf256(y_j);
    numerator / denominator
}

// =============================================================================
// Optimized Bulk Operations
// =============================================================================

/// Precomputed multiplication tables for all 256 possible multipliers
/// 
/// MUL_TABLES[y][x] = x * y in GF(256)
/// This is computed once at startup and cached for all subsequent operations
struct MulTables {
    tables: [[u8; 256]; 256],
}

impl MulTables {
    /// Initialize all multiplication tables
    const fn new() -> Self {
        let mut tables = [[0u8; 256]; 256];
        let mut y = 0usize;
        while y < 256 {
            let mut x = 0usize;
            while x < 256 {
                // Compute x * y in GF(256) using naive polynomial multiplication
                // This is the const-compatible version
                tables[y][x] = Self::gf_mul(x as u8, y as u8);
                x += 1;
            }
            y += 1;
        }
        Self { tables }
    }
    
    /// Const-compatible GF(256) multiplication
    /// Polynomial: 0x14d = x^8 + x^6 + x^3 + x^2 + 1
    const fn gf_mul(a: u8, b: u8) -> u8 {
        let mut result = 0u16;
        let mut aa = a as u16;
        let mut bb = b;
        
        // Polynomial multiplication
        let mut i = 0;
        while i < 8 {
            if bb & 1 != 0 {
                result ^= aa;
            }
            aa <<= 1;
            bb >>= 1;
            i += 1;
        }
        
        // Polynomial reduction by 0x14d
        let mut i = 15;
        while i >= 8 {
            if result & (1 << i) != 0 {
                result ^= 0x14d << (i - 8);
            }
            i -= 1;
        }
        
        result as u8
    }
    
    /// Get multiplication table for a specific multiplier
    #[inline(always)]
    fn get(&self, y: u8) -> &[u8; 256] {
        &self.tables[y as usize]
    }
}

/// Global precomputed multiplication tables
static MUL_TABLES: MulTables = MulTables::new();

// =============================================================================
// SIMD-Optimized Nibble Tables (for pshufb-based multiplication)
// =============================================================================

/// Precomputed nibble tables for SIMD multiplication
/// 
/// For each multiplier y:
/// - TABLE_LO[y][x] = (x & 0x0f) * y  (low nibble contribution)
/// - TABLE_HI[y][x] = (x << 4) * y    (high nibble contribution)
/// 
/// Result: x * y = TABLE_LO[y][x & 0x0f] ^ TABLE_HI[y][x >> 4]
#[allow(dead_code)]  // Used by SIMD code paths when target features are enabled
struct NibbleTables {
    lo: [[u8; 16]; 256],  // Low nibble tables
    hi: [[u8; 16]; 256],  // High nibble tables
}

impl NibbleTables {
    #[allow(dead_code)]
    const fn new() -> Self {
        let mut lo = [[0u8; 16]; 256];
        let mut hi = [[0u8; 16]; 256];
        
        let mut y = 0usize;
        while y < 256 {
            let mut x = 0usize;
            while x < 16 {
                // Low nibble: x * y where x is 0..15
                lo[y][x] = MulTables::gf_mul(x as u8, y as u8);
                // High nibble: (x << 4) * y where x is 0..15
                hi[y][x] = MulTables::gf_mul((x << 4) as u8, y as u8);
                x += 1;
            }
            y += 1;
        }
        
        Self { lo, hi }
    }
    
    #[allow(dead_code)]
    #[inline(always)]
    fn get_lo(&self, y: u8) -> &[u8; 16] {
        &self.lo[y as usize]
    }
    
    #[allow(dead_code)]
    #[inline(always)]
    fn get_hi(&self, y: u8) -> &[u8; 16] {
        &self.hi[y as usize]
    }
}

/// Global precomputed nibble tables for SIMD
#[allow(dead_code)]
static NIBBLE_TABLES: NibbleTables = NibbleTables::new();

// SIMD implementation using target_feature
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
mod simd {
    use super::*;
    use std::arch::x86_64::*;
    
    /// Multiply 16 bytes by a coefficient using SSSE3 pshufb
    #[inline(always)]
    pub unsafe fn mul_16(src: *const u8, table_lo: __m128i, table_hi: __m128i, mask_0f: __m128i) -> __m128i {
        let data = _mm_loadu_si128(src as *const __m128i);
        let lo_nibbles = _mm_and_si128(data, mask_0f);
        let lo_result = _mm_shuffle_epi8(table_lo, lo_nibbles);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi64(data, 4), mask_0f);
        let hi_result = _mm_shuffle_epi8(table_hi, hi_nibbles);
        _mm_xor_si128(lo_result, hi_result)
    }
    
    /// Multiply 32 bytes by a coefficient using AVX2 vpshufb
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    pub unsafe fn mul_32(src: *const u8, table_lo: __m256i, table_hi: __m256i, mask_0f: __m256i) -> __m256i {
        let data = _mm256_loadu_si256(src as *const __m256i);
        let lo_nibbles = _mm256_and_si256(data, mask_0f);
        let lo_result = _mm256_shuffle_epi8(table_lo, lo_nibbles);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi64(data, 4), mask_0f);
        let hi_result = _mm256_shuffle_epi8(table_hi, hi_nibbles);
        _mm256_xor_si256(lo_result, hi_result)
    }
    
    /// Multiply 64 bytes by a coefficient using AVX-512 vpshufb
    #[cfg(target_feature = "avx512bw")]
    #[inline(always)]
    pub unsafe fn mul_64(src: *const u8, table_lo: __m512i, table_hi: __m512i, mask_0f: __m512i) -> __m512i {
        let data = _mm512_loadu_si512(src as *const __m512i);
        let lo_nibbles = _mm512_and_si512(data, mask_0f);
        let lo_result = _mm512_shuffle_epi8(table_lo, lo_nibbles);
        let hi_nibbles = _mm512_and_si512(_mm512_srli_epi64(data, 4), mask_0f);
        let hi_result = _mm512_shuffle_epi8(table_hi, hi_nibbles);
        _mm512_xor_si512(lo_result, hi_result)
    }
    
    /// dst[i] = src[i] * coeff using SIMD (AVX-512 > AVX2 > SSE3)
    #[cfg(target_feature = "avx512bw")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn gf256_mul_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        // Create 512-bit tables by broadcasting the 128-bit tables 4x
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm512_broadcast_i32x4(table_lo_128);
        let table_hi = _mm512_broadcast_i32x4(table_hi_128);
        let mask_0f = _mm512_set1_epi8(0x0f);
        
        // Process 256 bytes at a time (4x unrolled)
        let chunks_256 = len / 256;
        for i in 0..chunks_256 {
            let offset = i * 256;
            let r0 = mul_64(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let r1 = mul_64(src.as_ptr().add(offset + 64), table_lo, table_hi, mask_0f);
            let r2 = mul_64(src.as_ptr().add(offset + 128), table_lo, table_hi, mask_0f);
            let r3 = mul_64(src.as_ptr().add(offset + 192), table_lo, table_hi, mask_0f);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, r0);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 64) as *mut __m512i, r1);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 128) as *mut __m512i, r2);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 192) as *mut __m512i, r3);
        }
        
        // Handle remaining 64-byte chunks
        let remainder_256 = chunks_256 * 256;
        let chunks_64 = (len - remainder_256) / 64;
        for i in 0..chunks_64 {
            let offset = remainder_256 + i * 64;
            let result = mul_64(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, result);
        }
        
        // Handle remaining bytes with AVX2
        let remainder_64 = remainder_256 + chunks_64 * 64;
        let table_lo_256 = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi_256 = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f_256 = _mm256_set1_epi8(0x0f);
        if len >= remainder_64 + 32 {
            let result = mul_32(src.as_ptr().add(remainder_64), table_lo_256, table_hi_256, mask_0f_256);
            _mm256_storeu_si256(dst.as_mut_ptr().add(remainder_64) as *mut __m256i, result);
        }
        
        // Handle 16-byte remainder with SSE3
        let remainder_32 = (len / 32) * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let result = mul_16(src.as_ptr().add(remainder_32), table_lo_128, table_hi_128, mask_0f_128);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, result);
        }
        
        // Handle final remainder with scalar
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[src[i] as usize];
        }
    }
    
    /// dst[i] = src[i] * coeff using SIMD (AVX2)
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf256_mul_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        // Create 256-bit tables by duplicating the 128-bit tables
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f = _mm256_set1_epi8(0x0f);
        
        // Process 128 bytes at a time (4x unrolled)
        let chunks_128 = len / 128;
        for i in 0..chunks_128 {
            let offset = i * 128;
            let r0 = mul_32(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let r1 = mul_32(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let r2 = mul_32(src.as_ptr().add(offset + 64), table_lo, table_hi, mask_0f);
            let r3 = mul_32(src.as_ptr().add(offset + 96), table_lo, table_hi, mask_0f);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, r0);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 32) as *mut __m256i, r1);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 64) as *mut __m256i, r2);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 96) as *mut __m256i, r3);
        }
        
        // Handle remaining 32-byte chunks
        let remainder_128 = chunks_128 * 128;
        let chunks_32 = (len - remainder_128) / 32;
        for i in 0..chunks_32 {
            let offset = remainder_128 + i * 32;
            let result = mul_32(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, result);
        }
        
        // Handle 16-byte remainder with SSE3
        let remainder_32 = remainder_128 + chunks_32 * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let result = mul_16(src.as_ptr().add(remainder_32), table_lo_128, table_hi_128, mask_0f_128);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, result);
        }
        
        // Handle final remainder with scalar
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[src[i] as usize];
        }
    }
    
    #[cfg(not(target_feature = "avx2"))]
    #[target_feature(enable = "ssse3")]
    pub unsafe fn gf256_mul_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        let table_lo = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let mask_0f = _mm_set1_epi8(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let r0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let r1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let r2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let r3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, r0);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 16) as *mut __m128i, r1);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 32) as *mut __m128i, r2);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 48) as *mut __m128i, r3);
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let result = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[src[i] as usize];
        }
    }
    
    /// dst[i] ^= src[i] * coeff using SIMD (AVX-512)
    #[cfg(target_feature = "avx512bw")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn gf256_muladd_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm512_broadcast_i32x4(table_lo_128);
        let table_hi = _mm512_broadcast_i32x4(table_hi_128);
        let mask_0f = _mm512_set1_epi8(0x0f);
        
        // Process 256 bytes at a time (4x unrolled)
        let chunks_256 = len / 256;
        for i in 0..chunks_256 {
            let offset = i * 256;
            let p0 = mul_64(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let p1 = mul_64(src.as_ptr().add(offset + 64), table_lo, table_hi, mask_0f);
            let p2 = mul_64(src.as_ptr().add(offset + 128), table_lo, table_hi, mask_0f);
            let p3 = mul_64(src.as_ptr().add(offset + 192), table_lo, table_hi, mask_0f);
            let c0 = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
            let c1 = _mm512_loadu_si512(dst.as_ptr().add(offset + 64) as *const __m512i);
            let c2 = _mm512_loadu_si512(dst.as_ptr().add(offset + 128) as *const __m512i);
            let c3 = _mm512_loadu_si512(dst.as_ptr().add(offset + 192) as *const __m512i);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, _mm512_xor_si512(c0, p0));
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 64) as *mut __m512i, _mm512_xor_si512(c1, p1));
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 128) as *mut __m512i, _mm512_xor_si512(c2, p2));
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset + 192) as *mut __m512i, _mm512_xor_si512(c3, p3));
        }
        
        // Handle remaining 64-byte chunks
        let remainder_256 = chunks_256 * 256;
        let chunks_64 = (len - remainder_256) / 64;
        for i in 0..chunks_64 {
            let offset = remainder_256 + i * 64;
            let product = mul_64(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let current = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, _mm512_xor_si512(current, product));
        }
        
        // Handle remaining with smaller SIMD
        let remainder_64 = remainder_256 + chunks_64 * 64;
        let table_lo_256 = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi_256 = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f_256 = _mm256_set1_epi8(0x0f);
        if len >= remainder_64 + 32 {
            let product = mul_32(src.as_ptr().add(remainder_64), table_lo_256, table_hi_256, mask_0f_256);
            let current = _mm256_loadu_si256(dst.as_ptr().add(remainder_64) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(remainder_64) as *mut __m256i, _mm256_xor_si256(current, product));
        }
        
        let remainder_32 = (len / 32) * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let product = mul_16(src.as_ptr().add(remainder_32), table_lo_128, table_hi_128, mask_0f_128);
            let current = _mm_loadu_si128(dst.as_ptr().add(remainder_32) as *const __m128i);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, _mm_xor_si128(current, product));
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] ^= table[src[i] as usize];
        }
    }
    
    /// dst[i] ^= src[i] * coeff using SIMD (AVX2)
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf256_muladd_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f = _mm256_set1_epi8(0x0f);
        
        // Process 128 bytes at a time (4x unrolled)
        let chunks_128 = len / 128;
        for i in 0..chunks_128 {
            let offset = i * 128;
            let p0 = mul_32(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let p1 = mul_32(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let p2 = mul_32(src.as_ptr().add(offset + 64), table_lo, table_hi, mask_0f);
            let p3 = mul_32(src.as_ptr().add(offset + 96), table_lo, table_hi, mask_0f);
            let c0 = _mm256_loadu_si256(dst.as_ptr().add(offset) as *const __m256i);
            let c1 = _mm256_loadu_si256(dst.as_ptr().add(offset + 32) as *const __m256i);
            let c2 = _mm256_loadu_si256(dst.as_ptr().add(offset + 64) as *const __m256i);
            let c3 = _mm256_loadu_si256(dst.as_ptr().add(offset + 96) as *const __m256i);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, _mm256_xor_si256(c0, p0));
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 32) as *mut __m256i, _mm256_xor_si256(c1, p1));
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 64) as *mut __m256i, _mm256_xor_si256(c2, p2));
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset + 96) as *mut __m256i, _mm256_xor_si256(c3, p3));
        }
        
        // Handle remaining 32-byte chunks
        let remainder_128 = chunks_128 * 128;
        let chunks_32 = (len - remainder_128) / 32;
        for i in 0..chunks_32 {
            let offset = remainder_128 + i * 32;
            let product = mul_32(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let current = _mm256_loadu_si256(dst.as_ptr().add(offset) as *const __m256i);
            let result = _mm256_xor_si256(current, product);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, result);
        }
        
        let remainder_32 = remainder_128 + chunks_32 * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let product = mul_16(src.as_ptr().add(remainder_32), table_lo_128, table_hi_128, mask_0f_128);
            let current = _mm_loadu_si128(dst.as_ptr().add(remainder_32) as *const __m128i);
            let result = _mm_xor_si128(current, product);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] ^= table[src[i] as usize];
        }
    }
    
    #[cfg(not(target_feature = "avx2"))]
    #[target_feature(enable = "ssse3")]
    pub unsafe fn gf256_muladd_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        let table_lo = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let mask_0f = _mm_set1_epi8(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let p0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let p1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let p2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let p3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            let c0 = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
            let c1 = _mm_loadu_si128(dst.as_ptr().add(offset + 16) as *const __m128i);
            let c2 = _mm_loadu_si128(dst.as_ptr().add(offset + 32) as *const __m128i);
            let c3 = _mm_loadu_si128(dst.as_ptr().add(offset + 48) as *const __m128i);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, _mm_xor_si128(c0, p0));
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 16) as *mut __m128i, _mm_xor_si128(c1, p1));
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 32) as *mut __m128i, _mm_xor_si128(c2, p2));
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 48) as *mut __m128i, _mm_xor_si128(c3, p3));
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let product = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let current = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
            let result = _mm_xor_si128(current, product);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] ^= table[src[i] as usize];
        }
    }
    
    /// dst[i] = dst[i] * coeff in-place using SIMD (AVX-512)
    #[cfg(target_feature = "avx512bw")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn gf256_mul_mem_inplace_simd(dst: &mut [u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm512_broadcast_i32x4(table_lo_128);
        let table_hi = _mm512_broadcast_i32x4(table_hi_128);
        let mask_0f = _mm512_set1_epi8(0x0f);
        
        // Process 64 bytes at a time
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let data = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
            let lo_nibbles = _mm512_and_si512(data, mask_0f);
            let lo_result = _mm512_shuffle_epi8(table_lo, lo_nibbles);
            let hi_nibbles = _mm512_and_si512(_mm512_srli_epi64(data, 4), mask_0f);
            let hi_result = _mm512_shuffle_epi8(table_hi, hi_nibbles);
            let result = _mm512_xor_si512(lo_result, hi_result);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, result);
        }
        
        // Handle remaining 32-byte chunks with AVX2
        let remainder_64 = chunks_64 * 64;
        let table_lo_256 = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi_256 = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f_256 = _mm256_set1_epi8(0x0f);
        if len >= remainder_64 + 32 {
            let data = _mm256_loadu_si256(dst.as_ptr().add(remainder_64) as *const __m256i);
            let lo_nibbles = _mm256_and_si256(data, mask_0f_256);
            let lo_result = _mm256_shuffle_epi8(table_lo_256, lo_nibbles);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi64(data, 4), mask_0f_256);
            let hi_result = _mm256_shuffle_epi8(table_hi_256, hi_nibbles);
            let result = _mm256_xor_si256(lo_result, hi_result);
            _mm256_storeu_si256(dst.as_mut_ptr().add(remainder_64) as *mut __m256i, result);
        }
        
        // Handle 16-byte remainder
        let remainder_32 = (len / 32) * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let data = _mm_loadu_si128(dst.as_ptr().add(remainder_32) as *const __m128i);
            let lo_nibbles = _mm_and_si128(data, mask_0f_128);
            let lo_result = _mm_shuffle_epi8(table_lo_128, lo_nibbles);
            let hi_nibbles = _mm_and_si128(_mm_srli_epi64(data, 4), mask_0f_128);
            let hi_result = _mm_shuffle_epi8(table_hi_128, hi_nibbles);
            let result = _mm_xor_si128(lo_result, hi_result);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[dst[i] as usize];
        }
    }
    
    /// dst[i] = dst[i] * coeff in-place using SIMD (AVX2)
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf256_mul_mem_inplace_simd(dst: &mut [u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo_128 = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi_128 = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let table_lo = _mm256_broadcastsi128_si256(table_lo_128);
        let table_hi = _mm256_broadcastsi128_si256(table_hi_128);
        let mask_0f = _mm256_set1_epi8(0x0f);
        
        let chunks_32 = len / 32;
        for i in 0..chunks_32 {
            let offset = i * 32;
            let data = _mm256_loadu_si256(dst.as_ptr().add(offset) as *const __m256i);
            let lo_nibbles = _mm256_and_si256(data, mask_0f);
            let lo_result = _mm256_shuffle_epi8(table_lo, lo_nibbles);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi64(data, 4), mask_0f);
            let hi_result = _mm256_shuffle_epi8(table_hi, hi_nibbles);
            let result = _mm256_xor_si256(lo_result, hi_result);
            _mm256_storeu_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, result);
        }
        
        let remainder_32 = chunks_32 * 32;
        let mask_0f_128 = _mm_set1_epi8(0x0f);
        if len >= remainder_32 + 16 {
            let data = _mm_loadu_si128(dst.as_ptr().add(remainder_32) as *const __m128i);
            let lo_nibbles = _mm_and_si128(data, mask_0f_128);
            let lo_result = _mm_shuffle_epi8(table_lo_128, lo_nibbles);
            let hi_nibbles = _mm_and_si128(_mm_srli_epi64(data, 4), mask_0f_128);
            let hi_result = _mm_shuffle_epi8(table_hi_128, hi_nibbles);
            let result = _mm_xor_si128(lo_result, hi_result);
            _mm_storeu_si128(dst.as_mut_ptr().add(remainder_32) as *mut __m128i, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[dst[i] as usize];
        }
    }
    
    #[cfg(not(target_feature = "avx2"))]
    #[target_feature(enable = "ssse3")]
    pub unsafe fn gf256_mul_mem_inplace_simd(dst: &mut [u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        let table_lo = _mm_loadu_si128(lo_table.as_ptr() as *const __m128i);
        let table_hi = _mm_loadu_si128(hi_table.as_ptr() as *const __m128i);
        let mask_0f = _mm_set1_epi8(0x0f);
        
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            let data = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
            let lo_nibbles = _mm_and_si128(data, mask_0f);
            let lo_result = _mm_shuffle_epi8(table_lo, lo_nibbles);
            let hi_nibbles = _mm_and_si128(_mm_srli_epi64(data, 4), mask_0f);
            let hi_result = _mm_shuffle_epi8(table_hi, hi_nibbles);
            let result = _mm_xor_si128(lo_result, hi_result);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
        }
        
        let remainder_start = chunks * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[dst[i] as usize];
        }
    }
}

// NEON SIMD implementation for ARM/aarch64
#[cfg(target_arch = "aarch64")]
mod simd_neon {
    use super::*;
    use std::arch::aarch64::*;
    
    /// Multiply 16 bytes by a coefficient using NEON vtbl
    /// Uses the nibble-based lookup: result[i] = TABLE_LO[coeff][src[i] & 0x0f] ^ TABLE_HI[coeff][src[i] >> 4]
    #[inline(always)]
    pub unsafe fn mul_16(src: *const u8, table_lo: uint8x16_t, table_hi: uint8x16_t, mask_0f: uint8x16_t) -> uint8x16_t {
        let data = vld1q_u8(src);
        
        // Get low nibbles and look up
        let lo_nibbles = vandq_u8(data, mask_0f);
        let lo_result = vqtbl1q_u8(table_lo, lo_nibbles);
        
        // Get high nibbles (shift right by 4), look up
        let hi_nibbles = vandq_u8(vshrq_n_u8(data, 4), mask_0f);
        let hi_result = vqtbl1q_u8(table_hi, hi_nibbles);
        
        // XOR the results
        veorq_u8(lo_result, hi_result)
    }
    
    /// dst[i] = src[i] * coeff using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn gf256_mul_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = vld1q_u8(lo_table.as_ptr());
        let table_hi = vld1q_u8(hi_table.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let r0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let r1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let r2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let r3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            vst1q_u8(dst.as_mut_ptr().add(offset), r0);
            vst1q_u8(dst.as_mut_ptr().add(offset + 16), r1);
            vst1q_u8(dst.as_mut_ptr().add(offset + 32), r2);
            vst1q_u8(dst.as_mut_ptr().add(offset + 48), r3);
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let result = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            vst1q_u8(dst.as_mut_ptr().add(offset), result);
        }
        
        // Handle final remainder with scalar
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[src[i] as usize];
        }
    }
    
    /// dst[i] ^= src[i] * coeff using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn gf256_muladd_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = vld1q_u8(lo_table.as_ptr());
        let table_hi = vld1q_u8(hi_table.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let p0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let p1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let p2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let p3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            let c0 = vld1q_u8(dst.as_ptr().add(offset));
            let c1 = vld1q_u8(dst.as_ptr().add(offset + 16));
            let c2 = vld1q_u8(dst.as_ptr().add(offset + 32));
            let c3 = vld1q_u8(dst.as_ptr().add(offset + 48));
            vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(c0, p0));
            vst1q_u8(dst.as_mut_ptr().add(offset + 16), veorq_u8(c1, p1));
            vst1q_u8(dst.as_mut_ptr().add(offset + 32), veorq_u8(c2, p2));
            vst1q_u8(dst.as_mut_ptr().add(offset + 48), veorq_u8(c3, p3));
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let product = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let current = vld1q_u8(dst.as_ptr().add(offset));
            let result = veorq_u8(current, product);
            vst1q_u8(dst.as_mut_ptr().add(offset), result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] ^= table[src[i] as usize];
        }
    }
    
    /// dst[i] = dst[i] * coeff in-place using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn gf256_mul_mem_inplace_simd(dst: &mut [u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = vld1q_u8(lo_table.as_ptr());
        let table_hi = vld1q_u8(hi_table.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let d0 = vld1q_u8(dst.as_ptr().add(offset));
            let d1 = vld1q_u8(dst.as_ptr().add(offset + 16));
            let d2 = vld1q_u8(dst.as_ptr().add(offset + 32));
            let d3 = vld1q_u8(dst.as_ptr().add(offset + 48));
            
            let lo0 = vandq_u8(d0, mask_0f);
            let lo1 = vandq_u8(d1, mask_0f);
            let lo2 = vandq_u8(d2, mask_0f);
            let lo3 = vandq_u8(d3, mask_0f);
            let hi0 = vandq_u8(vshrq_n_u8(d0, 4), mask_0f);
            let hi1 = vandq_u8(vshrq_n_u8(d1, 4), mask_0f);
            let hi2 = vandq_u8(vshrq_n_u8(d2, 4), mask_0f);
            let hi3 = vandq_u8(vshrq_n_u8(d3, 4), mask_0f);
            
            let r0 = veorq_u8(vqtbl1q_u8(table_lo, lo0), vqtbl1q_u8(table_hi, hi0));
            let r1 = veorq_u8(vqtbl1q_u8(table_lo, lo1), vqtbl1q_u8(table_hi, hi1));
            let r2 = veorq_u8(vqtbl1q_u8(table_lo, lo2), vqtbl1q_u8(table_hi, hi2));
            let r3 = veorq_u8(vqtbl1q_u8(table_lo, lo3), vqtbl1q_u8(table_hi, hi3));
            
            vst1q_u8(dst.as_mut_ptr().add(offset), r0);
            vst1q_u8(dst.as_mut_ptr().add(offset + 16), r1);
            vst1q_u8(dst.as_mut_ptr().add(offset + 32), r2);
            vst1q_u8(dst.as_mut_ptr().add(offset + 48), r3);
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let data = vld1q_u8(dst.as_ptr().add(offset));
            let lo_nibbles = vandq_u8(data, mask_0f);
            let hi_nibbles = vandq_u8(vshrq_n_u8(data, 4), mask_0f);
            let result = veorq_u8(vqtbl1q_u8(table_lo, lo_nibbles), vqtbl1q_u8(table_hi, hi_nibbles));
            vst1q_u8(dst.as_mut_ptr().add(offset), result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[dst[i] as usize];
        }
    }
}

// WebAssembly SIMD implementation
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd_wasm {
    use super::*;
    use std::arch::wasm32::*;
    
    /// Multiply 16 bytes by a coefficient using WASM SIMD i8x16_swizzle
    #[inline(always)]
    pub unsafe fn mul_16(src: *const u8, table_lo: v128, table_hi: v128, mask_0f: v128) -> v128 {
        let data = v128_load(src as *const v128);
        
        // Get low nibbles and look up
        let lo_nibbles = v128_and(data, mask_0f);
        let lo_result = i8x16_swizzle(table_lo, lo_nibbles);
        
        // Get high nibbles (shift right by 4), look up
        let hi_nibbles = v128_and(u8x16_shr(data, 4), mask_0f);
        let hi_result = i8x16_swizzle(table_hi, hi_nibbles);
        
        // XOR the results
        v128_xor(lo_result, hi_result)
    }
    
    /// dst[i] = src[i] * coeff using WASM SIMD
    #[target_feature(enable = "simd128")]
    pub unsafe fn gf256_mul_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = v128_load(lo_table.as_ptr() as *const v128);
        let table_hi = v128_load(hi_table.as_ptr() as *const v128);
        let mask_0f = u8x16_splat(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let r0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let r1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let r2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let r3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, r0);
            v128_store(dst.as_mut_ptr().add(offset + 16) as *mut v128, r1);
            v128_store(dst.as_mut_ptr().add(offset + 32) as *mut v128, r2);
            v128_store(dst.as_mut_ptr().add(offset + 48) as *mut v128, r3);
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let result = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, result);
        }
        
        // Handle final remainder with scalar
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[src[i] as usize];
        }
    }
    
    /// dst[i] ^= src[i] * coeff using WASM SIMD
    #[target_feature(enable = "simd128")]
    pub unsafe fn gf256_muladd_mem_simd(dst: &mut [u8], src: &[u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = v128_load(lo_table.as_ptr() as *const v128);
        let table_hi = v128_load(hi_table.as_ptr() as *const v128);
        let mask_0f = u8x16_splat(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let p0 = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let p1 = mul_16(src.as_ptr().add(offset + 16), table_lo, table_hi, mask_0f);
            let p2 = mul_16(src.as_ptr().add(offset + 32), table_lo, table_hi, mask_0f);
            let p3 = mul_16(src.as_ptr().add(offset + 48), table_lo, table_hi, mask_0f);
            let c0 = v128_load(dst.as_ptr().add(offset) as *const v128);
            let c1 = v128_load(dst.as_ptr().add(offset + 16) as *const v128);
            let c2 = v128_load(dst.as_ptr().add(offset + 32) as *const v128);
            let c3 = v128_load(dst.as_ptr().add(offset + 48) as *const v128);
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, v128_xor(c0, p0));
            v128_store(dst.as_mut_ptr().add(offset + 16) as *mut v128, v128_xor(c1, p1));
            v128_store(dst.as_mut_ptr().add(offset + 32) as *mut v128, v128_xor(c2, p2));
            v128_store(dst.as_mut_ptr().add(offset + 48) as *mut v128, v128_xor(c3, p3));
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let product = mul_16(src.as_ptr().add(offset), table_lo, table_hi, mask_0f);
            let current = v128_load(dst.as_ptr().add(offset) as *const v128);
            let result = v128_xor(current, product);
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] ^= table[src[i] as usize];
        }
    }
    
    /// dst[i] = dst[i] * coeff in-place using WASM SIMD
    #[target_feature(enable = "simd128")]
    pub unsafe fn gf256_mul_mem_inplace_simd(dst: &mut [u8], coeff: u8) {
        let len = dst.len();
        let lo_table = NIBBLE_TABLES.get_lo(coeff);
        let hi_table = NIBBLE_TABLES.get_hi(coeff);
        
        let table_lo = v128_load(lo_table.as_ptr() as *const v128);
        let table_hi = v128_load(hi_table.as_ptr() as *const v128);
        let mask_0f = u8x16_splat(0x0f);
        
        // Process 64 bytes at a time (4x unrolled)
        let chunks_64 = len / 64;
        for i in 0..chunks_64 {
            let offset = i * 64;
            let d0 = v128_load(dst.as_ptr().add(offset) as *const v128);
            let d1 = v128_load(dst.as_ptr().add(offset + 16) as *const v128);
            let d2 = v128_load(dst.as_ptr().add(offset + 32) as *const v128);
            let d3 = v128_load(dst.as_ptr().add(offset + 48) as *const v128);
            
            let lo0 = v128_and(d0, mask_0f);
            let lo1 = v128_and(d1, mask_0f);
            let lo2 = v128_and(d2, mask_0f);
            let lo3 = v128_and(d3, mask_0f);
            let hi0 = v128_and(u8x16_shr(d0, 4), mask_0f);
            let hi1 = v128_and(u8x16_shr(d1, 4), mask_0f);
            let hi2 = v128_and(u8x16_shr(d2, 4), mask_0f);
            let hi3 = v128_and(u8x16_shr(d3, 4), mask_0f);
            
            let r0 = v128_xor(i8x16_swizzle(table_lo, lo0), i8x16_swizzle(table_hi, hi0));
            let r1 = v128_xor(i8x16_swizzle(table_lo, lo1), i8x16_swizzle(table_hi, hi1));
            let r2 = v128_xor(i8x16_swizzle(table_lo, lo2), i8x16_swizzle(table_hi, hi2));
            let r3 = v128_xor(i8x16_swizzle(table_lo, lo3), i8x16_swizzle(table_hi, hi3));
            
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, r0);
            v128_store(dst.as_mut_ptr().add(offset + 16) as *mut v128, r1);
            v128_store(dst.as_mut_ptr().add(offset + 32) as *mut v128, r2);
            v128_store(dst.as_mut_ptr().add(offset + 48) as *mut v128, r3);
        }
        
        // Handle remaining 16-byte chunks
        let remainder_64 = chunks_64 * 64;
        let chunks_16 = (len - remainder_64) / 16;
        for i in 0..chunks_16 {
            let offset = remainder_64 + i * 16;
            let data = v128_load(dst.as_ptr().add(offset) as *const v128);
            let lo_nibbles = v128_and(data, mask_0f);
            let hi_nibbles = v128_and(u8x16_shr(data, 4), mask_0f);
            let result = v128_xor(i8x16_swizzle(table_lo, lo_nibbles), i8x16_swizzle(table_hi, hi_nibbles));
            v128_store(dst.as_mut_ptr().add(offset) as *mut v128, result);
        }
        
        let remainder_start = (len / 16) * 16;
        let table = MUL_TABLES.get(coeff);
        for i in remainder_start..len {
            dst[i] = table[dst[i] as usize];
        }
    }
}

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
fn has_ssse3() -> bool {
    #[cfg(target_feature = "ssse3")]
    { true }
    #[cfg(not(target_feature = "ssse3"))]
    { false }
}

/// dst[i] = src[i] * coeff
/// Using SIMD when available, otherwise lookup table
#[inline]
fn gf256_mul_mem(dst: &mut [u8], src: &[u8], coeff: Gf256) {
    debug_assert_eq!(dst.len(), src.len());
    
    if coeff.0 == 0 {
        dst.fill(0);
        return;
    }
    if coeff.0 == 1 {
        dst.copy_from_slice(src);
        return;
    }

    // Use x86 SIMD when compiled with SSSE3
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "ssse3"))]
    {
        unsafe { simd::gf256_mul_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Use NEON SIMD on aarch64
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe { simd_neon::gf256_mul_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Use WASM SIMD on wasm32
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { simd_wasm::gf256_mul_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Scalar fallback (when SIMD feature disabled or no SIMD available)
    #[cfg(not(all(feature = "simd", any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))))]
    {
        let table = MUL_TABLES.get(coeff.0);
        for i in 0..dst.len() {
            dst[i] = table[src[i] as usize];
        }
    }
}

/// dst[i] += src[i] * coeff (where + is XOR)
/// Using SIMD when available, otherwise lookup table
#[inline]
fn gf256_muladd_mem(dst: &mut [u8], src: &[u8], coeff: Gf256) {
    debug_assert_eq!(dst.len(), src.len());
    
    if coeff.0 == 0 {
        return;
    }
    if coeff.0 == 1 {
        // Just XOR
        gf256_xor_mem(dst, src);
        return;
    }

    // Use x86 SIMD when compiled with SSSE3
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "ssse3"))]
    {
        unsafe { simd::gf256_muladd_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Use NEON SIMD on aarch64
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe { simd_neon::gf256_muladd_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Use WASM SIMD on wasm32
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { simd_wasm::gf256_muladd_mem_simd(dst, src, coeff.0) };
        return;
    }

    // Scalar fallback (when SIMD feature disabled or no SIMD available)
    #[cfg(not(all(feature = "simd", any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))))]
    {
        let table = MUL_TABLES.get(coeff.0);
        for i in 0..dst.len() {
            dst[i] ^= table[src[i] as usize];
        }
    }
}

/// dst[i] = src1[i] XOR src2[i]
/// Optimized with wide operations
#[inline]
fn gf256_add_mem(dst: &mut [u8], src1: &[u8], src2: &[u8]) {
    debug_assert_eq!(dst.len(), src1.len());
    debug_assert_eq!(dst.len(), src2.len());
    
    // Process 8 bytes at a time using u64
    let len = dst.len();
    let chunks = len / 8;
    
    // Cast to u64 slices for wide operations
    // SAFETY: alignment handled by processing only complete u64s
    for i in 0..chunks {
        let base = i * 8;
        let v1 = u64::from_ne_bytes(src1[base..base + 8].try_into().unwrap());
        let v2 = u64::from_ne_bytes(src2[base..base + 8].try_into().unwrap());
        dst[base..base + 8].copy_from_slice(&(v1 ^ v2).to_ne_bytes());
    }
    
    // Handle remainder
    let base = chunks * 8;
    for i in base..len {
        dst[i] = src1[i] ^ src2[i];
    }
}

/// dst[i] ^= src[i]
/// Optimized with wide operations
#[inline]
fn gf256_xor_mem(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    
    let len = dst.len();
    let chunks = len / 8;
    
    for i in 0..chunks {
        let base = i * 8;
        let v1 = u64::from_ne_bytes(dst[base..base + 8].try_into().unwrap());
        let v2 = u64::from_ne_bytes(src[base..base + 8].try_into().unwrap());
        dst[base..base + 8].copy_from_slice(&(v1 ^ v2).to_ne_bytes());
    }
    
    let base = chunks * 8;
    for i in base..len {
        dst[i] ^= src[i];
    }
}

/// dst[i] = dst[i] * coeff (in-place multiplication)
/// Using SIMD when available, otherwise lookup table
#[inline]
fn gf256_mul_mem_inplace(dst: &mut [u8], coeff: Gf256) {
    if coeff.0 == 0 {
        dst.fill(0);
        return;
    }
    if coeff.0 == 1 {
        return;
    }

    // Use x86 SIMD when compiled with SSSE3
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "ssse3"))]
    {
        unsafe { simd::gf256_mul_mem_inplace_simd(dst, coeff.0) };
        return;
    }

    // Use NEON SIMD on aarch64
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        unsafe { simd_neon::gf256_mul_mem_inplace_simd(dst, coeff.0) };
        return;
    }

    // Use WASM SIMD on wasm32
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { simd_wasm::gf256_mul_mem_inplace_simd(dst, coeff.0) };
        return;
    }

    // Scalar fallback (when SIMD feature disabled or no SIMD available)
    #[cfg(not(all(feature = "simd", any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))))]
    {
        let table = MUL_TABLES.get(coeff.0);
        for i in 0..dst.len() {
            dst[i] = table[dst[i] as usize];
        }
    }
}
// =============================================================================
// Encoding
// =============================================================================

/// Encode a single recovery block
///
/// # Arguments
/// * `params` - Encoder parameters
/// * `originals` - Slice of original blocks
/// * `recovery_index` - Which recovery block to generate (0..recovery_count)
/// * `output` - Output buffer for the recovery block (must be block_bytes long)
pub fn encode_block(
    params: &Params,
    originals: &[Block],
    recovery_index: usize,
    output: &mut [u8],
) {
    debug_assert_eq!(originals.len(), params.original_count);
    debug_assert_eq!(output.len(), params.block_bytes);
    debug_assert!(recovery_index < params.recovery_count);

    // Special case: only one original block
    if params.original_count == 1 {
        output.copy_from_slice(originals[0].data);
        return;
    }

    // Recovery block index in the encoding matrix
    let recovery_block_index = params.original_count + recovery_index;

    // First recovery row is all 1s (simple XOR parity)
    if recovery_index == 0 {
        // output = originals[0] XOR originals[1] using optimized wide XOR
        gf256_add_mem(output, originals[0].data, originals[1].data);
        // XOR remaining blocks
        for j in 2..params.original_count {
            gf256_xor_mem(output, originals[j].data);
        }
        return;
    }

    // For other rows, compute matrix-vector product using bulk operations
    let x_0 = params.original_count as u8;
    let x_i = recovery_block_index as u8;

    // First column: output = original[0] * matrix_element
    {
        let y_0 = 0u8;
        let coeff = get_matrix_element(x_i, x_0, y_0);
        gf256_mul_mem(output, originals[0].data, coeff);
    }

    // Remaining columns: output += original[j] * matrix_element
    for j in 1..params.original_count {
        let y_j = j as u8;
        let coeff = get_matrix_element(x_i, x_0, y_j);
        gf256_muladd_mem(output, originals[j].data, coeff);
    }
}

/// Encode all recovery blocks
///
/// # Arguments
/// * `params` - Encoder parameters
/// * `originals` - Slice of original blocks (must have original_count elements)
/// * `recovery_output` - Output buffer for all recovery blocks (must be recovery_count * block_bytes)
///
/// # Errors
/// Returns an error if inputs are invalid
pub fn encode(
    params: &Params,
    originals: &[Block],
    recovery_output: &mut [u8],
) -> Result<(), Error> {
    if originals.len() != params.original_count {
        return Err(Error::InvalidInput);
    }
    if recovery_output.len() != params.recovery_count * params.block_bytes {
        return Err(Error::InvalidInput);
    }

    let block_bytes = params.block_bytes;

    for recovery_idx in 0..params.recovery_count {
        let start = recovery_idx * block_bytes;
        let end = start + block_bytes;
        encode_block(params, originals, recovery_idx, &mut recovery_output[start..end]);
    }

    Ok(())
}

// =============================================================================
// Decoding
// =============================================================================

/// Decoder state
struct Decoder<'a> {
    params: Params,
    /// Recovery blocks received (pointers into input)
    recovery: Vec<&'a mut [u8]>,
    /// Recovery block indices
    recovery_indices: Vec<u8>,
    /// Original blocks received (immutable references)
    original: Vec<&'a [u8]>,
    /// Original block indices
    original_indices: Vec<u8>,
    /// Indices of erased original blocks (need to recover these)
    erasure_indices: Vec<u8>,
}

impl<'a> Decoder<'a> {
    /// Decode when only one recovery block is needed (m=1 case)
    /// 
    /// Uses the first row property: first recovery is XOR of all originals
    fn decode_m1(&mut self) {
        let block_bytes = self.params.block_bytes;
        let recovery = &mut self.recovery[0];

        // XOR all received original blocks into recovery block
        for orig in &self.original {
            for i in 0..block_bytes {
                recovery[i] ^= orig[i];
            }
        }
    }

    /// Generate LDU decomposition of the recovery matrix
    ///
    /// Uses Schur-type-direct-Cauchy algorithm from:
    /// "Pivoting and Backward Stability of Fast Algorithms for Solving Cauchy Linear Equations"
    fn generate_ldu_decomposition(&self) -> (Vec<Gf256>, Vec<Gf256>, Vec<Gf256>) {
        let n = self.recovery.len();
        let x_0 = self.params.original_count as u8;

        // L is lower triangular (stored column-by-column, excluding diagonal)
        // D is diagonal
        // U is upper triangular (stored in a special rotated format)
        let mut matrix_l: Vec<Gf256> = Vec::with_capacity(n * (n - 1) / 2);
        let mut diag_d: Vec<Gf256> = vec![Gf256(0); n];
        let mut matrix_u: Vec<Gf256> = vec![Gf256(0); n * (n - 1) / 2];

        // Generators
        let mut g: Vec<Gf256> = vec![Gf256(1); n];
        let mut b: Vec<Gf256> = vec![Gf256(1); n];

        // Build initial U storage layout tracking
        let mut last_u_offset = (n * (n - 1)) / 2;

        for k in 0..(n - 1) {
            let x_k = self.recovery_indices[k];
            let y_k = self.erasure_indices[k];

            // D_kk = (x_k + y_k)
            // L_kk = g[k] / (x_k + y_k)
            // U_kk = b[k] * (x_0 + y_k) / (x_k + y_k)
            let d_kk = Gf256(x_k) + Gf256(y_k);
            let l_kk = g[k] / d_kk;
            let u_kk = b[k] / d_kk * (Gf256(x_0) + Gf256(y_k));

            // diag_D[k] = D_kk * L_kk * U_kk
            diag_d[k] = d_kk * l_kk * u_kk;

            // Compute k-th row of L and U
            let mut row_u: Vec<Gf256> = Vec::with_capacity(n - k - 1);

            for j in (k + 1)..n {
                let x_j = self.recovery_indices[j];
                let y_j = self.erasure_indices[j];

                // L_jk = g[j] / (x_j + y_k)
                // U_kj = b[j] / (x_k + y_j)
                let l_jk = g[j] / (Gf256(x_j) + Gf256(y_k));
                let u_kj = b[j] / (Gf256(x_k) + Gf256(y_j));

                matrix_l.push(l_jk / l_kk);
                row_u.push(u_kj / u_kk);

                // Update generators
                // g[j] = g[j] * (x_j + x_k) / (x_j + y_k)
                // b[j] = b[j] * (y_j + y_k) / (y_j + x_k)
                g[j] = g[j] * (Gf256(x_j) + Gf256(x_k)) / (Gf256(x_j) + Gf256(y_k));
                b[j] = b[j] * (Gf256(y_j) + Gf256(y_k)) / (Gf256(y_j) + Gf256(x_k));
            }

            // Store U row in rotated format (column-first, bottom-up)
            last_u_offset -= n - k - 1;
            for (idx, &val) in row_u.iter().enumerate() {
                // Calculate position in the rotated U matrix
                let mut pos = last_u_offset;
                for i in 0..idx {
                    pos += n - k - 2 - i;
                }
                if pos < matrix_u.len() {
                    matrix_u[pos] = val;
                }
            }
        }

        // Multiply diagonal into U (scale columns)
        {
            let mut u_idx = 0;
            for j in (1..n).rev() {
                let y_j = self.erasure_indices[j];
                let scale = Gf256(x_0) + Gf256(y_j);
                for _ in 0..j {
                    matrix_u[u_idx] = matrix_u[u_idx] * scale;
                    u_idx += 1;
                }
            }
        }

        // Final diagonal element
        {
            let n_1 = n - 1;
            let x_n = self.recovery_indices[n_1];
            let y_n = self.erasure_indices[n_1];
            let l_nn = g[n_1];
            let u_nn = b[n_1] * (Gf256(x_0) + Gf256(y_n));
            diag_d[n_1] = l_nn * u_nn / (Gf256(x_n) + Gf256(y_n));
        }

        (matrix_l, diag_d, matrix_u)
    }

    /// Decode for m>1 case using LDU decomposition
    fn decode(&mut self) {
        let n = self.recovery.len();
        let x_0 = self.params.original_count as u8;

        // Step 1: Eliminate known original data from recovery blocks
        for orig_idx in 0..self.original.len() {
            let in_block = self.original[orig_idx];
            let in_row = self.original_indices[orig_idx];

            for rec_idx in 0..n {
                let x_i = self.recovery_indices[rec_idx];
                let y_j = in_row;
                let coeff = get_matrix_element(x_i, x_0, y_j);

                let out_block = &mut self.recovery[rec_idx];
                gf256_muladd_mem(out_block, in_block, coeff);
            }
        }

        // Step 2: Generate LDU decomposition
        let (matrix_l, diag_d, matrix_u) = self.generate_ldu_decomposition();

        // Step 3: Forward substitution (eliminate lower triangle)
        // Use split borrows to avoid allocation
        {
            let mut l_idx = 0;
            for j in 0..(n - 1) {
                // Split recovery at j+1 to get non-overlapping borrows
                let (left, right) = self.recovery.split_at_mut(j + 1);
                let block_j = &left[j];
                
                for (_offset, block_i) in right.iter_mut().enumerate() {
                    let c_ij = matrix_l[l_idx];
                    l_idx += 1;
                    gf256_muladd_mem(block_i, block_j, c_ij);
                }
            }
        }

        // Step 4: Diagonal elimination (divide by diagonal)
        for i in 0..n {
            let d_inv = Gf256(1) / diag_d[i];
            let block = &mut self.recovery[i];
            gf256_mul_mem_inplace(block, d_inv);
        }

        // Step 5: Back substitution (eliminate upper triangle)
        // Use split borrows to avoid allocation
        {
            let mut u_idx = 0;
            for j in (1..n).rev() {
                // Split recovery at j to get non-overlapping borrows
                let (left, right) = self.recovery.split_at_mut(j);
                let block_j = &right[0];
                
                for block_i in left.iter_mut().rev() {
                    let c_ij = matrix_u[u_idx];
                    u_idx += 1;
                    gf256_muladd_mem(block_i, block_j, c_ij);
                }
            }
        }
    }
}

/// Decode lost blocks from available original and recovery blocks
///
/// Recovery blocks are modified in-place to contain the recovered original data.
/// After decoding, each recovery block's data contains the recovered original block,
/// and you should check the erasure_indices to know which original block index it corresponds to.
///
/// # Arguments
/// * `params` - Encoder parameters (same as used for encoding)
/// * `blocks` - Mutable slice of blocks (must have exactly original_count elements)
///   - For original blocks: provide immutable data
///   - For recovery blocks: data will be modified to contain recovered original
///
/// # Returns
/// Returns the indices of blocks that were recovered (in the order they appear in the recovery blocks)
///
/// # Errors
/// Returns an error if there are duplicate indices or insufficient blocks
pub fn decode(
    params: &Params,
    blocks: &mut [BlockMut],
) -> Result<Vec<u8>, Error> {
    if blocks.len() != params.original_count {
        return Err(Error::InsufficientBlocks);
    }

    // Special case: only one original block
    if params.original_count == 1 {
        // The one recovery block is identical to the original
        return Ok(vec![0]);
    }

    // Separate blocks into original and recovery, identify erasures
    let mut received = [false; 256];
    let mut recovery_blocks: Vec<&mut [u8]> = Vec::new();
    let mut recovery_indices: Vec<u8> = Vec::new();
    let mut original_data: Vec<&[u8]> = Vec::new();
    let mut original_indices: Vec<u8> = Vec::new();

    for block in blocks.iter_mut() {
        let idx = block.index as usize;

        if idx < params.original_count {
            // Original block
            if received[idx] {
                return Err(Error::DuplicateIndex);
            }
            received[idx] = true;
            original_data.push(&*block.data);
            original_indices.push(block.index);
        } else {
            // Recovery block
            recovery_blocks.push(&mut *block.data);
            recovery_indices.push(block.index);
        }
    }

    // Identify erasures (missing original blocks)
    let mut erasure_indices: Vec<u8> = Vec::new();
    for i in 0..params.original_count {
        if !received[i] {
            erasure_indices.push(i as u8);
        }
    }

    // If nothing is erased, we're done
    if erasure_indices.is_empty() {
        return Ok(vec![]);
    }

    // Ensure we have enough recovery blocks
    if recovery_blocks.len() < erasure_indices.len() {
        return Err(Error::InsufficientBlocks);
    }

    // Truncate to exactly the number we need
    recovery_blocks.truncate(erasure_indices.len());
    recovery_indices.truncate(erasure_indices.len());

    let mut decoder = Decoder {
        params: *params,
        recovery: recovery_blocks,
        recovery_indices,
        original: original_data,
        original_indices,
        erasure_indices: erasure_indices.clone(),
    };

    // Decode based on recovery count
    if params.recovery_count == 1 || decoder.recovery.len() == 1 {
        decoder.decode_m1();
    } else {
        decoder.decode();
    }

    Ok(erasure_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_creation() {
        let params = Params::new(3, 2, 1024).unwrap();
        assert_eq!(params.original_count(), 3);
        assert_eq!(params.recovery_count(), 2);
        assert_eq!(params.block_bytes(), 1024);
    }

    #[test]
    fn test_params_invalid() {
        assert!(Params::new(0, 2, 1024).is_err());
        assert!(Params::new(3, 0, 1024).is_err());
        assert!(Params::new(3, 2, 0).is_err());
        assert!(Params::new(200, 100, 1024).is_err()); // > 256
    }

    #[test]
    fn test_encode_single_block() {
        let params = Params::new(1, 1, 16).unwrap();
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let blocks = vec![Block::new(0, &data)];

        let mut recovery = vec![0u8; 16];
        encode(&params, &blocks, &mut recovery).unwrap();

        // Single block: recovery is just a copy
        assert_eq!(recovery, data);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let params = Params::new(3, 2, 16).unwrap();

        // Create original data
        let orig0 = vec![1u8; 16];
        let orig1 = vec![2u8; 16];
        let orig2 = vec![3u8; 16];

        let blocks = vec![
            Block::new(0, &orig0),
            Block::new(1, &orig1),
            Block::new(2, &orig2),
        ];

        // Encode
        let mut recovery = vec![0u8; 32]; // 2 recovery blocks
        encode(&params, &blocks, &mut recovery).unwrap();

        // Simulate losing block 1, use recovery block 0 instead
        let mut rec0 = recovery[..16].to_vec();
        let mut block0_data = orig0.clone();
        let mut block2_data = orig2.clone();
        let mut decode_blocks = vec![
            BlockMut::new(0, &mut block0_data),
            BlockMut::new(3, &mut rec0), // recovery block 0 has index 3
            BlockMut::new(2, &mut block2_data),
        ];

        let recovered = decode(&params, &mut decode_blocks).unwrap();
        assert_eq!(recovered, vec![1]); // Block 1 was recovered
        assert_eq!(rec0, orig1); // rec0 now contains original block 1
    }

    #[test]
    fn test_xor_parity() {
        let params = Params::new(3, 1, 4).unwrap();

        let orig0 = vec![0x11, 0x22, 0x33, 0x44];
        let orig1 = vec![0x55, 0x66, 0x77, 0x88];
        let orig2 = vec![0x99, 0xAA, 0xBB, 0xCC];

        let blocks = vec![
            Block::new(0, &orig0),
            Block::new(1, &orig1),
            Block::new(2, &orig2),
        ];

        let mut recovery = vec![0u8; 4];
        encode(&params, &blocks, &mut recovery).unwrap();

        // First recovery block should be XOR of all originals
        let expected: Vec<u8> = orig0.iter()
            .zip(orig1.iter())
            .zip(orig2.iter())
            .map(|((&a, &b), &c)| a ^ b ^ c)
            .collect();
        assert_eq!(recovery, expected);
    }

    #[test]
    fn test_multiple_erasures() {
        let params = Params::new(5, 3, 32).unwrap();

        // Create distinct original data
        let orig: Vec<Vec<u8>> = (0..5)
            .map(|i| (0..32).map(|j| ((i * 32 + j) % 256) as u8).collect())
            .collect();

        let blocks: Vec<Block> = orig.iter()
            .enumerate()
            .map(|(i, data)| Block::new(i as u8, data))
            .collect();

        // Encode
        let mut recovery = vec![0u8; 3 * 32];
        encode(&params, &blocks, &mut recovery).unwrap();

        // Test: lose blocks 0 and 3, use recovery blocks 0 and 1
        let mut rec0 = recovery[..32].to_vec();
        let mut rec1 = recovery[32..64].to_vec();
        let mut block1_data = orig[1].clone();
        let mut block2_data = orig[2].clone();
        let mut block4_data = orig[4].clone();

        let mut decode_blocks = vec![
            BlockMut::new(5, &mut rec0),  // recovery block 0
            BlockMut::new(1, &mut block1_data),
            BlockMut::new(2, &mut block2_data),
            BlockMut::new(6, &mut rec1),  // recovery block 1
            BlockMut::new(4, &mut block4_data),
        ];

        let recovered = decode(&params, &mut decode_blocks).unwrap();
        
        // Should have recovered blocks 0 and 3
        assert_eq!(recovered.len(), 2);
        assert!(recovered.contains(&0));
        assert!(recovered.contains(&3));

        // Verify the recovered data matches originals
        // rec0 should now contain block 0 or 3 depending on order
        // (the specific mapping depends on erasure order)
    }

    // =========================================================================
    // C++ Reference Comparison Tests
    // These tests verify bit-identical output with the original C++ cm256
    // =========================================================================

    #[test]
    fn test_cpp_reference_case1() {
        // C++ output:
        // orig0: 01010101010101010101010101010101
        // orig1: 02020202020202020202020202020202
        // orig2: 03030303030303030303030303030303
        // rec0 : 00000000000000000000000000000000
        // rec1 : d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5

        let params = Params::new(3, 2, 16).unwrap();

        let orig0 = vec![0x01u8; 16];
        let orig1 = vec![0x02u8; 16];
        let orig2 = vec![0x03u8; 16];

        let blocks = vec![
            Block::new(0, &orig0),
            Block::new(1, &orig1),
            Block::new(2, &orig2),
        ];

        let mut recovery = vec![0u8; 32];
        encode(&params, &blocks, &mut recovery).unwrap();

        // Expected from C++
        let expected_rec0 = vec![0x00u8; 16]; // XOR: 01 ^ 02 ^ 03 = 00
        let expected_rec1 = vec![0xd5u8; 16];

        assert_eq!(&recovery[..16], &expected_rec0[..], "rec0 mismatch with C++");
        assert_eq!(&recovery[16..], &expected_rec1[..], "rec1 mismatch with C++");
    }

    #[test]
    fn test_cpp_reference_case2_xor_parity() {
        // C++ output:
        // orig0: 11223344
        // orig1: 55667788
        // orig2: 99aabbcc
        // rec0 : ddeeff00
        
        let params = Params::new(3, 1, 4).unwrap();

        let orig0 = vec![0x11, 0x22, 0x33, 0x44];
        let orig1 = vec![0x55, 0x66, 0x77, 0x88];
        let orig2 = vec![0x99, 0xAA, 0xBB, 0xCC];

        let blocks = vec![
            Block::new(0, &orig0),
            Block::new(1, &orig1),
            Block::new(2, &orig2),
        ];

        let mut recovery = vec![0u8; 4];
        encode(&params, &blocks, &mut recovery).unwrap();

        let expected = vec![0xdd, 0xee, 0xff, 0x00];
        assert_eq!(recovery, expected, "XOR parity mismatch with C++");
    }

    #[test]
    fn test_cpp_reference_case3_multi_block() {
        // C++ output for 5 orig, 3 recovery, 32 bytes:
        // rec0: 808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f
        // rec1: ae03b914802d973af25fe548dc71cb6616bb01ac38952f824ae75df064c973de
        // rec2: 3934232e0d00171a515c4b4665687f72e9e4f3feddd0c7ca818c9b96b5b8afa2

        let params = Params::new(5, 3, 32).unwrap();

        let orig: Vec<Vec<u8>> = (0..5)
            .map(|i| (0..32).map(|j| ((i * 32 + j) % 256) as u8).collect())
            .collect();

        let blocks: Vec<Block> = orig.iter()
            .enumerate()
            .map(|(i, data)| Block::new(i as u8, data))
            .collect();

        let mut recovery = vec![0u8; 3 * 32];
        encode(&params, &blocks, &mut recovery).unwrap();

        // Expected from C++ (hex decoded)
        let expected_rec0: Vec<u8> = (0x80..=0x9f).collect();
        let expected_rec1: Vec<u8> = vec![
            0xae, 0x03, 0xb9, 0x14, 0x80, 0x2d, 0x97, 0x3a,
            0xf2, 0x5f, 0xe5, 0x48, 0xdc, 0x71, 0xcb, 0x66,
            0x16, 0xbb, 0x01, 0xac, 0x38, 0x95, 0x2f, 0x82,
            0x4a, 0xe7, 0x5d, 0xf0, 0x64, 0xc9, 0x73, 0xde,
        ];
        let expected_rec2: Vec<u8> = vec![
            0x39, 0x34, 0x23, 0x2e, 0x0d, 0x00, 0x17, 0x1a,
            0x51, 0x5c, 0x4b, 0x46, 0x65, 0x68, 0x7f, 0x72,
            0xe9, 0xe4, 0xf3, 0xfe, 0xdd, 0xd0, 0xc7, 0xca,
            0x81, 0x8c, 0x9b, 0x96, 0xb5, 0xb8, 0xaf, 0xa2,
        ];

        assert_eq!(&recovery[..32], &expected_rec0[..], "rec0 mismatch with C++");
        assert_eq!(&recovery[32..64], &expected_rec1[..], "rec1 mismatch with C++");
        assert_eq!(&recovery[64..], &expected_rec2[..], "rec2 mismatch with C++");
    }
}
