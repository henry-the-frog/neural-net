import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { compressionRatio, tokenFrequencies } from './tokenizer-stats.js';

describe('Tokenizer Stats', () => {
  test('compression ratio > 1 for character tokenization', () => {
    const ratio = compressionRatio('hello world', [1,2,3,4,5,6,7,8,9,10,11]);
    assert.equal(ratio, 1); // 11 bytes / 11 tokens = 1
  });

  test('compression ratio > 1 for subword', () => {
    const ratio = compressionRatio('hello world', [1, 2]); // 2 tokens for 11 bytes
    assert.ok(ratio > 5);
  });

  test('token frequencies sorted by count', () => {
    const freqs = tokenFrequencies([1, 2, 1, 3, 1, 2]);
    assert.equal(freqs[0][0], 1); // Most frequent
    assert.equal(freqs[0][1], 3);
  });
});
