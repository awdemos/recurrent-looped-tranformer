//! Byte-level tokenizer with BOS/EOS specials. Keeps demos self-contained.

use crate::{Result, RltError};

/// Beginning-of-sequence token id (byte-level: 256).
pub const BOS_ID: u32 = 256;
/// End-of-sequence token id (byte-level: 257).
pub const EOS_ID: u32 = 257;
/// Minimum vocab size for this tokenizer (all bytes + BOS + EOS).
pub const MIN_VOCAB: usize = 258;

/// Byte-level tokenizer: token id = byte value; specials live above 255.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    vocab_size: usize,
}

impl ByteTokenizer {
    /// Create a tokenizer; `vocab_size` must be >= [`MIN_VOCAB`].
    pub fn new(vocab_size: usize) -> Result<Self> {
        if vocab_size < MIN_VOCAB {
            return Err(RltError::Token(format!(
                "vocab_size {vocab_size} < {MIN_VOCAB} (need all byte values + BOS + EOS)"
            )));
        }
        Ok(Self { vocab_size })
    }

    /// Encode text to token ids, optionally adding BOS/EOS.
    pub fn encode(&self, text: &str, with_bos: bool, with_eos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        if with_bos {
            ids.push(BOS_ID);
        }
        ids.extend(text.as_bytes().iter().map(|b| u32::from(*b)));
        if with_eos {
            ids.push(EOS_ID);
        }
        ids
    }

    /// Decode token ids back to text; special tokens (>= 256) are skipped and
    /// invalid ids are replaced with the replacement character.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| u8::try_from(t).ok())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size this tokenizer was built for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
