// bpe.test.js — Tests for BPE tokenizer
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BPETokenizer } from './bpe.js';

describe('BPE Tokenizer', () => {
  describe('training', () => {
    it('trains on simple corpus', () => {
      const tok = new BPETokenizer();
      tok.train('aaabdaaabac', 20);
      assert.ok(tok.size > 0);
      assert.ok(tok.merges.length > 0);
    });

    it('first merge is most frequent pair', () => {
      const tok = new BPETokenizer();
      tok.train('aaaa', 10); // 'aa' is the most frequent pair
      assert.equal(tok.merges[0].pair[0], 'a');
      assert.equal(tok.merges[0].pair[1], 'a');
      assert.equal(tok.merges[0].merged, 'aa');
    });

    it('vocabulary grows with merges', () => {
      const tok = new BPETokenizer();
      tok.train('abababab', 15);
      const baseChars = new Set('abababab').size; // a, b → 2 chars
      const specials = 2; // <|endoftext|>, <|pad|>
      assert.ok(tok.size > baseChars + specials, `Vocab should grow: ${tok.size}`);
    });

    it('respects vocab size limit', () => {
      const tok = new BPETokenizer();
      tok.train('the cat sat on the mat', 15);
      assert.ok(tok.size <= 15, `Vocab should be <= 15, got ${tok.size}`);
    });
  });

  describe('encode/decode roundtrip', () => {
    it('decodes back to original text', () => {
      const tok = new BPETokenizer();
      tok.train('hello world hello world', 30);
      const text = 'hello world';
      const ids = tok.encode(text);
      const decoded = tok.decode(ids);
      assert.equal(decoded, text);
    });

    it('handles unseen characters gracefully', () => {
      const tok = new BPETokenizer();
      tok.train('abc', 10);
      // 'z' not in training data → fallback to ID 0
      const ids = tok.encode('abcz');
      assert.ok(ids.length > 0);
    });

    it('encodes empty string', () => {
      const tok = new BPETokenizer();
      tok.train('hello', 10);
      const ids = tok.encode('');
      assert.equal(ids.length, 0);
    });

    it('encodes single character', () => {
      const tok = new BPETokenizer();
      tok.train('abc', 10);
      const ids = tok.encode('a');
      assert.equal(ids.length, 1);
      assert.equal(tok.decode(ids), 'a');
    });
  });

  describe('merge behavior', () => {
    it('merges reduce token count', () => {
      const tok = new BPETokenizer();
      tok.train('abababab', 20);
      
      // Without merges: 8 tokens (a,b,a,b,a,b,a,b)
      // With merges: should be fewer
      const ids = tok.encode('abababab');
      assert.ok(ids.length < 8, `Should compress: ${ids.length} tokens from 8 chars`);
    });

    it('repeated patterns get merged efficiently', () => {
      const tok = new BPETokenizer();
      tok.train('aaa bbb aaa bbb aaa bbb', 30);
      
      const ids1 = tok.encode('aaa');
      const ids2 = tok.encode('bbb');
      // Common patterns should be single tokens
      assert.ok(ids1.length <= 2, `"aaa" should be 1-2 tokens, got ${ids1.length}`);
      assert.ok(ids2.length <= 2, `"bbb" should be 1-2 tokens, got ${ids2.length}`);
    });
  });

  describe('export/import', () => {
    it('roundtrips through export/import', () => {
      const tok = new BPETokenizer();
      tok.train('hello world', 20);
      
      const exported = tok.export();
      const tok2 = BPETokenizer.import(exported);
      
      const text = 'hello world';
      assert.deepEqual(tok2.encode(text), tok.encode(text));
      assert.equal(tok2.decode(tok2.encode(text)), text);
    });

    it('preserves special tokens', () => {
      const tok = new BPETokenizer();
      tok.train('test', 10, ['<s>', '</s>']);
      
      const exported = tok.export();
      const tok2 = BPETokenizer.import(exported);
      
      assert.ok(tok2.specialTokens.has('<s>'));
      assert.ok(tok2.specialTokens.has('</s>'));
    });
  });

  describe('real-world-ish corpus', () => {
    it('tokenizes English text efficiently', () => {
      const corpus = `The quick brown fox jumps over the lazy dog. 
        The quick brown fox jumps over the lazy dog again.
        Dogs and foxes are friends sometimes.`;
      
      const tok = new BPETokenizer();
      tok.train(corpus, 50);
      
      const encoded = tok.encode('The quick brown fox');
      const decoded = tok.decode(encoded);
      assert.equal(decoded, 'The quick brown fox');
      
      // Should compress due to repeated patterns
      assert.ok(encoded.length < 19, `Should compress 19 chars to fewer tokens: ${encoded.length}`);
    });
  });
});
