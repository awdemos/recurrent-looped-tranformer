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
    assert_eq!(tok.decode(&[BOS_ID, b'a' as u32, b'b' as u32, EOS_ID]), "ab");
}

#[test]
fn rejects_tiny_vocab() {
    assert!(ByteTokenizer::new(100).is_err());
}
