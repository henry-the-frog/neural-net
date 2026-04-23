import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Structured Generation', () => {
  function constrainedSample(logits, allowedTokens) {
    const masked = logits.map((l, i) => allowedTokens.includes(i) ? l : -Infinity);
    return masked.indexOf(Math.max(...masked));
  }

  test('only samples from allowed tokens', () => {
    const logits = [10, 5, 8, 3]; // Token 0 is best
    const allowed = [1, 3]; // But only 1 and 3 are allowed
    assert.equal(constrainedSample(logits, allowed), 1);
  });

  test('handles single allowed token', () => {
    assert.equal(constrainedSample([1, 2, 3], [2]), 2);
  });
});
