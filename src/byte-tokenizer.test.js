// byte-tokenizer.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { ByteTokenizer, entropyBasedPatching } from './byte-tokenizer.js';

describe('Byte Tokenizer', () => {
  test('encode produces correct length', () => {
    const bt = new ByteTokenizer();
    const ids = bt.encode('abc', true);
    // BOS + 3 bytes + EOS = 5
    assert.equal(ids.length, 5);
  });

  test('decode recovers original text', () => {
    const bt = new ByteTokenizer();
    const text = 'Hello, world!';
    const ids = bt.encode(text, false);
    const decoded = bt.decode(ids);
    assert.equal(decoded, text);
  });

  test('roundtrip with special tokens', () => {
    const bt = new ByteTokenizer();
    const text = 'test';
    const ids = bt.encode(text, true);
    const decoded = bt.decode(ids, true); // Skip special
    assert.equal(decoded, text);
  });

  test('handles Unicode (multi-byte characters)', () => {
    const bt = new ByteTokenizer();
    const text = '你好世界'; // Chinese: 12 UTF-8 bytes
    const ids = bt.encode(text, false);
    assert.equal(ids.length, 12); // 4 chars × 3 bytes each
    assert.equal(bt.decode(ids), text);
  });

  test('handles emoji', () => {
    const bt = new ByteTokenizer();
    const text = '🚀';
    const ids = bt.encode(text, false);
    assert.equal(ids.length, 4); // 4 UTF-8 bytes
    assert.equal(bt.decode(ids), text);
  });

  test('vocabSize is special + 256', () => {
    const bt = new ByteTokenizer(['<pad>', '<bos>', '<eos>']);
    assert.equal(bt.vocabSize, 259);
  });

  test('byteToId and idToByte are inverse', () => {
    const bt = new ByteTokenizer();
    for (const byte of [0, 65, 127, 255]) {
      const id = bt.byteToId(byte);
      assert.equal(bt.idToByte(id), byte);
    }
  });

  test('entropy patching groups similar bytes', () => {
    const bytes = [65, 66, 67, 200, 65, 66]; // ABC then jump to 200 then back
    const patches = entropyBasedPatching(bytes, 0.3);
    assert.ok(patches.length >= 2, `Should create multiple patches: ${patches.length}`);
  });
});
