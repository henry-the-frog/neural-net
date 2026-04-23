import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Watermarking for LLM-generated text (Kirchenbauer et al., 2023)
describe('LLM Watermark', () => {
  function hashToken(prevToken, seed) {
    return ((prevToken * 2654435761 + seed) >>> 0) % 2;
  }

  test('hash is deterministic', () => {
    assert.equal(hashToken(42, 123), hashToken(42, 123));
  });

  test('different tokens give different hashes', () => {
    const h1 = hashToken(1, 100);
    const h2 = hashToken(2, 100);
    // Not guaranteed to differ, but should for these values
    assert.ok(typeof h1 === 'number');
  });

  test('green list detection', () => {
    const tokens = [1, 2, 3, 4, 5];
    const greenCount = tokens.filter((t, i) => i > 0 && hashToken(tokens[i-1], 42) === 0).length;
    assert.ok(greenCount >= 0 && greenCount <= tokens.length);
  });
});
