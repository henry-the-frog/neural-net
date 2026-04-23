// bpe-tokenizer.test.js — BPE tokenizer tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { BPETokenizer } from './bpe-tokenizer.js';

describe('BPETokenizer', () => {
  test('basic training on repeated text', () => {
    const tok = new BPETokenizer({ vocabSize: 300 });
    const result = tok.train('aaabdaaabac');
    assert.ok(result.numMerges > 0);
    assert.ok(result.vocabSize > result.baseVocabSize);
    // Most frequent pair is 'aa' → should be first merge
    assert.equal(tok.merges[0].left, 'a');
    assert.equal(tok.merges[0].right, 'a');
    assert.equal(tok.merges[0].merged, 'aa');
  });

  test('encode/decode roundtrip', () => {
    const tok = new BPETokenizer({ vocabSize: 300 });
    tok.train('the quick brown fox jumps over the lazy dog the quick brown fox');
    const text = 'the quick brown fox';
    const ids = tok.encode(text);
    const decoded = tok.decode(ids);
    assert.equal(decoded, text);
  });

  test('encode produces fewer tokens than chars', () => {
    const text = 'abcabcabcabcabcabc';
    const tok = new BPETokenizer({ vocabSize: 300 });
    tok.train(text);
    const ids = tok.encode(text);
    // After merging 'abc' → 6 tokens instead of 18 chars
    assert.ok(ids.length < text.length, `Expected fewer tokens (${ids.length}) than chars (${text.length})`);
  });

  test('handles empty text', () => {
    const tok = new BPETokenizer();
    tok.train('hello world');
    const ids = tok.encode('');
    assert.deepEqual(ids, []);
    assert.equal(tok.decode([]), '');
  });

  test('handles unseen characters gracefully', () => {
    const tok = new BPETokenizer();
    tok.train('hello world');
    // 'z' might not be in vocab — should still encode without crash
    const ids = tok.encode('hello z');
    const decoded = tok.decode(ids);
    assert.ok(decoded.includes('hello'));
  });

  test('vocabulary grows with merges', () => {
    const tok = new BPETokenizer({ vocabSize: 300 });
    const result = tok.train('ababababababababab', 3);
    assert.equal(result.numMerges, 3);
    assert.equal(result.vocabSize, result.baseVocabSize + 3);
  });

  test('getVocab returns sorted entries', () => {
    const tok = new BPETokenizer();
    tok.train('abc abc abc');
    const vocab = tok.getVocab();
    assert.ok(vocab.length > 0);
    // Should be sorted by ID
    for (let i = 1; i < vocab.length; i++) {
      assert.ok(vocab[i][1] >= vocab[i-1][1]);
    }
  });

  test('special tokens', () => {
    const tok = new BPETokenizer();
    tok.addSpecialToken('<|endoftext|>', 0);
    tok.train('hello world hello world');
    assert.equal(tok.vocab.get('<|endoftext|>'), 0);
    assert.equal(tok.inverseVocab.get(0), '<|endoftext|>');
  });

  test('serialization roundtrip', () => {
    const tok = new BPETokenizer({ vocabSize: 300 });
    tok.train('the cat sat on the mat the cat sat on the mat');
    const text = 'the cat sat';
    const ids1 = tok.encode(text);
    
    const json = tok.toJSON();
    const tok2 = BPETokenizer.fromJSON(json);
    const ids2 = tok2.encode(text);
    assert.deepEqual(ids1, ids2);
    assert.equal(tok2.decode(ids2), text);
  });

  test('trains on longer text (Shakespeare)', () => {
    const text = `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream.`;
    
    const tok = new BPETokenizer({ vocabSize: 300 });
    const result = tok.train(text);
    assert.ok(result.numMerges > 10, `Expected many merges, got ${result.numMerges}`);
    
    // Encode should compress significantly
    const ids = tok.encode(text);
    assert.ok(ids.length < text.length, `BPE should compress: ${ids.length} tokens < ${text.length} chars`);
    
    // Should decode back perfectly
    assert.equal(tok.decode(ids), text);
  });

  test('compression ratio improves with more training data', () => {
    const base = 'hello world ';
    const shortText = base.repeat(5);
    const longText = base.repeat(50);
    
    const tok1 = new BPETokenizer({ vocabSize: 300 });
    tok1.train(shortText);
    const ratio1 = tok1.encode(shortText).length / shortText.length;
    
    const tok2 = new BPETokenizer({ vocabSize: 300 });
    tok2.train(longText);
    const ratio2 = tok2.encode(longText).length / longText.length;
    
    // More training data → better compression
    assert.ok(ratio2 <= ratio1 + 0.1, `Long text ratio (${ratio2}) should be <= short (${ratio1})`);
  });

  test('controlled number of merges', () => {
    const tok = new BPETokenizer();
    const result = tok.train('aababab', 2);
    assert.equal(tok.merges.length, 2);
  });
});
