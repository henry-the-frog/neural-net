// tokenizer-utils.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { applyRepetitionPenalty, applyFrequencyPenalty, minPFilter } from './tokenizer-utils.js';

describe('Tokenizer Utils', () => {
  test('repetition penalty reduces repeated token logits', () => {
    const logits = new Float64Array([5, 3, 1, 2]);
    const modified = applyRepetitionPenalty(logits, [0, 0, 1], 1.5);
    assert.ok(modified[0] < logits[0], 'Token 0 should be penalized');
    assert.ok(modified[1] < logits[1], 'Token 1 should be penalized');
    assert.equal(modified[2], logits[2], 'Token 2 should be unchanged');
  });

  test('repetition penalty=1 is no-op', () => {
    const logits = new Float64Array([5, 3, 1]);
    const modified = applyRepetitionPenalty(logits, [0, 1], 1.0);
    for (let i = 0; i < 3; i++) {
      assert.equal(modified[i], logits[i]);
    }
  });

  test('frequency penalty scales with count', () => {
    const logits = new Float64Array([5, 5, 5]);
    const modified = applyFrequencyPenalty(logits, [0, 0, 0, 1], 0.5, 0);
    // Token 0 appears 3 times → penalized by 1.5
    // Token 1 appears 1 time → penalized by 0.5
    assert.ok(modified[0] < modified[1], 'More frequent = more penalized');
    assert.equal(modified[2], 5, 'Unseen token unchanged');
  });

  test('minP filter removes low-probability tokens', () => {
    const logits = new Float64Array([10, 0, 0, 0, -5]); // Token 0 dominates
    const filtered = minPFilter(logits, 0.1, 1.0);
    // Token 0 has high prob, others should be filtered
    assert.ok(filtered[0] > -Infinity, 'High prob token should survive');
    assert.equal(filtered[4], -Infinity, 'Very low prob token should be filtered');
  });

  test('minP=0 keeps all tokens', () => {
    const logits = new Float64Array([10, 0, -5]);
    const filtered = minPFilter(logits, 0, 1.0);
    for (let i = 0; i < 3; i++) {
      assert.ok(filtered[i] > -Infinity);
    }
  });
});
