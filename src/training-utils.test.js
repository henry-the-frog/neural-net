// training-utils.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { clipGradNorm, clipGradValue, linearWarmup, warmupCosineDecay, warmupInvSqrt, warmupPolynomial } from './training-utils.js';

describe('Training Utils', () => {
  test('clipGradNorm reduces large gradients', () => {
    const grads = [new Float64Array([10, 0, 0])]; // Norm = 10
    const { clipped, gradNorm, wasClipped } = clipGradNorm(grads, 1.0);
    assert.ok(wasClipped);
    assert.ok(Math.abs(gradNorm - 10) < 0.01);
    // Clipped norm should be ~1
    const clippedNorm = Math.sqrt(clipped[0].reduce((s, v) => s + v * v, 0));
    assert.ok(Math.abs(clippedNorm - 1) < 0.01);
  });

  test('clipGradNorm does not modify small gradients', () => {
    const grads = [new Float64Array([0.1, 0.2])];
    const { clipped, wasClipped } = clipGradNorm(grads, 1.0);
    assert.ok(!wasClipped);
    assert.ok(Math.abs(clipped[0][0] - 0.1) < 1e-10);
  });

  test('clipGradValue clips element-wise', () => {
    const grads = [new Float64Array([5, -3, 0.5])];
    const clipped = clipGradValue(grads, 1.0);
    assert.equal(clipped[0][0], 1.0);
    assert.equal(clipped[0][1], -1.0);
    assert.equal(clipped[0][2], 0.5);
  });

  test('linear warmup: 0 at start, baseLR at warmupSteps', () => {
    assert.equal(linearWarmup(0, 0.001, 100), 0);
    assert.ok(Math.abs(linearWarmup(100, 0.001, 100) - 0.001) < 1e-8);
    assert.ok(Math.abs(linearWarmup(50, 0.001, 100) - 0.0005) < 1e-8);
  });

  test('warmup cosine: peaks at warmupSteps, decays to minLR', () => {
    const peak = warmupCosineDecay(100, 0.001, 100, 1000, 0);
    assert.ok(Math.abs(peak - 0.001) < 1e-6);
    
    const end = warmupCosineDecay(1000, 0.001, 100, 1000, 0);
    assert.ok(end < 0.0001, `Should decay near 0: ${end}`);
  });

  test('warmup inv sqrt: peaks near warmup boundary', () => {
    const before = warmupInvSqrt(100, 512, 4000);
    const after = warmupInvSqrt(10000, 512, 4000);
    // After warmup, LR should decay
    assert.ok(after < warmupInvSqrt(4000, 512, 4000));
  });

  test('warmup polynomial: linear decay with power=1', () => {
    const mid = warmupPolynomial(550, 0.001, 100, 1000, 1.0, 0);
    assert.ok(mid > 0 && mid < 0.001, `Mid should be between 0 and baseLR: ${mid}`);
  });
});
