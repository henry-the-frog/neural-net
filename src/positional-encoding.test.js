// positional-encoding.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { sinusoidalEncoding, timestepEmbedding, alibiSlopes, alibiBiasMatrix, learnedPositionalEmbedding } from './positional-encoding.js';

describe('Positional Encoding', () => {
  test('sinusoidal has correct shape', () => {
    const pe = sinusoidalEncoding(100, 64);
    assert.equal(pe.rows, 100);
    assert.equal(pe.cols, 64);
  });

  test('sinusoidal values are in [-1, 1]', () => {
    const pe = sinusoidalEncoding(50, 32);
    for (let i = 0; i < pe.data.length; i++) {
      assert.ok(Math.abs(pe.data[i]) <= 1.0001, `Value ${pe.data[i]} out of range`);
    }
  });

  test('sinusoidal: different positions have different encodings', () => {
    const pe = sinusoidalEncoding(10, 16);
    let diff = 0;
    for (let d = 0; d < 16; d++) diff += Math.abs(pe.get(0, d) - pe.get(5, d));
    assert.ok(diff > 0.1, 'Different positions should differ');
  });

  test('sinusoidal: first position has sin(0)=0, cos(0)=1', () => {
    const pe = sinusoidalEncoding(5, 4);
    assert.ok(Math.abs(pe.get(0, 0) - 0) < 0.001); // sin(0) = 0
    assert.ok(Math.abs(pe.get(0, 1) - 1) < 0.001); // cos(0) = 1
  });

  test('timestep embedding has correct dimension', () => {
    const emb = timestepEmbedding(50, 32);
    assert.equal(emb.length, 32);
  });

  test('different timesteps produce different embeddings', () => {
    const e1 = timestepEmbedding(0, 16);
    const e2 = timestepEmbedding(500, 16);
    let diff = 0;
    for (let i = 0; i < 16; i++) diff += Math.abs(e1[i] - e2[i]);
    assert.ok(diff > 0.1);
  });

  test('ALiBi slopes are geometric sequence', () => {
    const slopes = alibiSlopes(8);
    assert.equal(slopes.length, 8);
    // Each slope should be a constant ratio of the previous
    const ratio = slopes[1] / slopes[0];
    for (let i = 2; i < 8; i++) {
      assert.ok(Math.abs(slopes[i] / slopes[i-1] - ratio) < 1e-10);
    }
  });

  test('ALiBi bias matrix is causal', () => {
    const bias = alibiBiasMatrix(4, 0.5);
    // Future tokens (j > i) should be -inf-like
    assert.ok(bias.get(0, 1) < -1e8);
    assert.ok(bias.get(0, 2) < -1e8);
    // Current token (j=i) should be 0
    assert.ok(Math.abs(bias.get(0, 0)) < 1e-10);
    assert.ok(Math.abs(bias.get(1, 1)) < 1e-10);
  });

  test('ALiBi bias increases linearly with distance', () => {
    const slope = 0.25;
    const bias = alibiBiasMatrix(5, slope);
    // bias(3, 0) should be -slope * 3
    assert.ok(Math.abs(bias.get(3, 0) - (-slope * 3)) < 1e-10);
    assert.ok(Math.abs(bias.get(3, 1) - (-slope * 2)) < 1e-10);
    assert.ok(Math.abs(bias.get(3, 2) - (-slope * 1)) < 1e-10);
  });
});
