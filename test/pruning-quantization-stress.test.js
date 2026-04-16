// pruning-quantization-stress.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { magnitudePrune, randomPrune, structuredPrune, sparsity, gradualPrune, StructuredPruner } from '../src/pruning.js';
import { quantize, dequantize, quantizeWeights, bitsRequired } from '../src/quantization.js';
import { Matrix } from '../src/matrix.js';

describe('Pruning Stress', () => {
  it('magnitude pruning zeroes small weights', () => {
    const w = new Matrix(1, 10, new Float64Array([0.01, 0.5, -0.02, 0.8, 0.03, -0.9, 0.04, 0.7, -0.05, 0.6]));
    const pruned = magnitudePrune(w, 0.5); // prune 50%
    let zeros = 0;
    for (let i = 0; i < 10; i++) if (pruned.data[i] === 0) zeros++;
    assert.ok(zeros >= 4, `Should zero ~50% of weights: ${zeros}/10`);
  });

  it('pruning preserves large weights', () => {
    const w = new Matrix(1, 4, new Float64Array([0.01, 10, -0.01, -10]));
    const pruned = magnitudePrune(w, 0.5);
    assert.ok(Math.abs(pruned.data[1] - 10) < 0.01, 'Large positive weight preserved');
    assert.ok(Math.abs(pruned.data[3] - (-10)) < 0.01, 'Large negative weight preserved');
  });

  it('sparsity calculation', () => {
    const w = new Matrix(1, 10, new Float64Array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]));
    const s = sparsity(w);
    assert.ok(Math.abs(s - 0.5) < 0.01, `Sparsity should be 0.5: ${s}`);
  });

  it('100% pruning zeroes everything', () => {
    const w = Matrix.random(3, 4);
    const pruned = magnitudePrune(w, 1.0);
    for (let i = 0; i < pruned.data.length; i++) {
      assert.equal(pruned.data[i], 0, 'All weights should be zero');
    }
  });

  it('0% pruning preserves everything', () => {
    const w = Matrix.random(3, 4);
    const pruned = magnitudePrune(w, 0.0);
    for (let i = 0; i < w.data.length; i++) {
      assert.equal(pruned.data[i], w.data[i], 'All weights should be preserved');
    }
  });
});

describe('Quantization Stress', () => {
  it('quantize-dequantize roundtrip', () => {
    const value = 0.5;
    const q = quantize(value, 8); // 8-bit
    const dq = dequantize(q, 8);
    assert.ok(Math.abs(dq - value) < 0.01, `Roundtrip should be close: ${value} → ${q} → ${dq}`);
  });

  it('quantize preserves sign', () => {
    const pos = quantize(0.7, 8);
    const neg = quantize(-0.7, 8);
    const dqPos = dequantize(pos, 8);
    const dqNeg = dequantize(neg, 8);
    assert.ok(dqPos > 0, 'Positive should stay positive');
    assert.ok(dqNeg < 0, 'Negative should stay negative');
  });

  it('quantize weights matrix', () => {
    const w = new Matrix(3, 4);
    for (let i = 0; i < 12; i++) w.data[i] = (Math.random() - 0.5) * 2;
    const { quantized, scale, zeroPoint } = quantizeWeights(w, 8);
    assert.ok(quantized, 'Quantized matrix should exist');
    assert.ok(isFinite(scale), 'Scale should be finite');
  });

  it('8-bit quantization resolution', () => {
    // 8-bit should have 256 levels
    const bits = bitsRequired(256);
    assert.equal(bits, 8, '256 levels needs 8 bits');
  });

  it('lower bits = more quantization error', () => {
    const value = 0.333;
    const q4 = Math.abs(dequantize(quantize(value, 4), 4) - value);
    const q8 = Math.abs(dequantize(quantize(value, 8), 8) - value);
    assert.ok(q4 >= q8, `4-bit error (${q4}) should be >= 8-bit error (${q8})`);
  });

  it('zero quantizes to zero', () => {
    const q = quantize(0, 8);
    const dq = dequantize(q, 8);
    assert.ok(Math.abs(dq) < 0.01, `Zero should roundtrip: ${dq}`);
  });
});
