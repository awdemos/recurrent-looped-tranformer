use rlt_core::{ByteTokenizer, BOS_ID, EOS_ID};

#[test]
fn roundtrip_with_specials() {
    let tok = ByteTokenizer::new(258).unwrap();
    let text = "Hello, 世界 → RLT\n";
    let ids = tok.encode(text, true, true);
    assert_eq!(ids[0], BOS_ID);
    assert_eq!(*ids.last().unwrap(), EOS_ID);
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn decode_skips_specials() {
    let tok = ByteTokenizer::new(258).unwrap();
    assert_eq!(
        tok.decode(&[BOS_ID, b'a' as u32, b'b' as u32, EOS_ID]),
        "ab"
    );
}

#[test]
fn rejects_tiny_vocab() {
    assert!(ByteTokenizer::new(100).is_err());
}

#[test]
fn rejects_boundary_vocab() {
    assert!(ByteTokenizer::new(257).is_err()); // one below MIN_VOCAB
    assert!(ByteTokenizer::new(258).is_ok());
}

#[test]
fn all_bytes_roundtrip() {
    let tok = ByteTokenizer::new(258).unwrap();
    // Bytes 0x00-0x7F are valid UTF-8 on their own: exact encode/decode roundtrip.
    let ascii: String = (0u8..=127).map(char::from).collect();
    assert_eq!(tok.decode(&tok.encode(&ascii, false, false)), ascii);
    // Every byte value decodes to the same string as lossy UTF-8 decoding;
    // 0x80-0xFF alone are not valid UTF-8, so they surface as U+FFFD.
    let all: Vec<u8> = (0u8..=255).collect();
    let want = String::from_utf8_lossy(&all).into_owned();
    let ids: Vec<u32> = (0u32..=255).collect();
    assert_eq!(tok.decode(&ids), want);
}
