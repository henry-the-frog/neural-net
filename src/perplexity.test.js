import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { perplexity, bitsPerByte } from './perplexity.js';

describe('Perplexity', () => {
  test('perfect prediction → PPL = 1', () => {
    // log P = 0 for each token (P=1)
    assert.ok(Math.abs(perplexity([0, 0, 0]) - 1) < 0.001);
  });

  test('uniform random → PPL = vocab size', () => {
    // P = 1/100 → log P = -log(100) ≈ -4.605
    const lp = -Math.log(100);
    assert.ok(Math.abs(perplexity([lp, lp, lp]) - 100) < 1);
  });

  test('bits per byte is positive', () => {
    const bpb = bitsPerByte([-2, -3, -1], 10);
    assert.ok(bpb > 0);
  });
});
