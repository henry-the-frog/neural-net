// regularization.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { gradientPenalty, spectralNormalization, l1Penalty, l2Penalty, elasticNetPenalty } from './regularization.js';
import { Matrix } from './matrix.js';

describe('Regularization', () => {
  test('gradient penalty for linear function has grad norm ≈ slope', () => {
    // D(x) = 2*x[0] → gradient norm = 2
    const D = (x) => 2 * x[0];
    const real = new Float64Array([1]);
    const fake = new Float64Array([-1]);
    const { gradNorm } = gradientPenalty(D, real, fake);
    assert.ok(Math.abs(gradNorm - 2) < 0.1, `Grad norm should be ~2, got ${gradNorm}`);
  });

  test('gradient penalty is 0 when grad norm = 1', () => {
    const D = (x) => x[0]; // Gradient = 1
    const { penalty } = gradientPenalty(D, new Float64Array([1]), new Float64Array([-1]));
    assert.ok(penalty < 0.1, `Penalty should be ~0, got ${penalty}`);
  });

  test('spectral normalization reduces largest singular value to ~1', () => {
    const W = Matrix.random(4, 4).map(v => v * 5); // Large weights
    const { normalized, sigma } = spectralNormalization(W, 10);
    assert.ok(sigma > 1, `Sigma should be > 1 for large weights: ${sigma}`);
    
    // Normalized matrix should have spectral norm ~1
    const { sigma: newSigma } = spectralNormalization(normalized, 10);
    assert.ok(Math.abs(newSigma - 1) < 0.2, `Normalized sigma should be ~1, got ${newSigma}`);
  });

  test('L1 penalty is sum of absolute values', () => {
    const w = new Float64Array([1, -2, 3]);
    assert.ok(Math.abs(l1Penalty(w, 1) - 6) < 1e-10);
  });

  test('L2 penalty is half sum of squares', () => {
    const w = new Float64Array([1, -2, 3]);
    assert.ok(Math.abs(l2Penalty(w, 1) - 7) < 1e-10); // 0.5 * (1 + 4 + 9) = 7
  });

  test('elastic net interpolates between L1 and L2', () => {
    const w = new Float64Array([1, -2, 3]);
    const l1 = l1Penalty(w, 1);
    const l2 = l2Penalty(w, 1);
    const elastic = elasticNetPenalty(w, 1, 0.5);
    assert.ok(Math.abs(elastic - 0.5 * l1 - 0.5 * l2) < 1e-10);
  });

  test('L1 penalty: zero weights have zero cost', () => {
    const zeros = new Float64Array([0, 0, 0]);
    assert.equal(l1Penalty(zeros), 0);
    const nonzero = new Float64Array([1, 2, 3]);
    assert.ok(l1Penalty(nonzero) > 0);
  });
});
