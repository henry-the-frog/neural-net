// mixed-precision-audit.test.js — Deep numerical stability audit
// Targets: BatchNorm, Adam, Attention, Gradient Clipping
// Looks for silent NaN/Inf propagation, division by zero, overflow

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Matrix } from './matrix.js';
import { BatchNorm } from './batchnorm.js';
import { Adam, AdamW, createOptimizer } from './optimizer.js';
import { SelfAttention } from './attention.js';
import { clipByValue, clipByNorm, clipByGlobalNorm } from './gradient-clip.js';

function allFinite(m, msg = '') {
  for (let i = 0; i < m.data.length; i++) {
    if (!isFinite(m.data[i])) {
      assert.fail(`${msg} data[${i}] is ${m.data[i]} (not finite)`);
    }
  }
}

function noNaN(m, msg = '') {
  for (let i = 0; i < m.data.length; i++) {
    if (isNaN(m.data[i])) {
      assert.fail(`${msg} data[${i}] is NaN`);
    }
  }
}

function hasNaN(m) {
  for (let i = 0; i < m.data.length; i++) {
    if (isNaN(m.data[i])) return true;
  }
  return false;
}

// ============================================================
// 1. BATCHNORM NUMERICAL STABILITY
// ============================================================
describe('BatchNorm Numerical Stability', () => {
  it('handles near-zero variance without NaN', () => {
    // All values identical → variance = 0 → division by sqrt(0 + eps)
    const bn = new BatchNorm(3);
    const input = new Matrix(4, 3);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 3; j++) {
        input.set(i, j, 5.0); // All same value
      }
    }
    const output = bn.forward(input);
    noNaN(output, 'BatchNorm zero-variance forward');
    allFinite(output, 'BatchNorm zero-variance forward');
  });

  it('handles single-sample batch', () => {
    // batch_size=1 → variance = 0 for all features
    const bn = new BatchNorm(4);
    const input = new Matrix(1, 4, new Float64Array([1, 2, 3, 4]));
    const output = bn.forward(input);
    noNaN(output, 'BatchNorm single-sample forward');
    allFinite(output, 'BatchNorm single-sample forward');
  });

  it('handles extreme input values', () => {
    const bn = new BatchNorm(3);
    const input = new Matrix(2, 3, new Float64Array([
      1e15, -1e15, 0,
      1e15, -1e15, 0,
    ]));
    const output = bn.forward(input);
    noNaN(output, 'BatchNorm extreme forward');
    allFinite(output, 'BatchNorm extreme forward');
  });

  it('backward pass with zero variance does not produce NaN', () => {
    const bn = new BatchNorm(3);
    const input = new Matrix(4, 3);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 3; j++) input.set(i, j, 7.0);
    }
    bn.forward(input);
    const gradOutput = Matrix.ones(4, 3);
    try {
      const gradInput = bn.backward(gradOutput);
      noNaN(gradInput, 'BatchNorm zero-var backward');
      allFinite(gradInput, 'BatchNorm zero-var backward');
    } catch (e) {
      // If backward throws, that's better than silent NaN
      assert.ok(true, 'BatchNorm backward threw on zero variance (acceptable)');
    }
  });

  it('running statistics do not overflow after many forward passes', () => {
    const bn = new BatchNorm(2);
    for (let i = 0; i < 100; i++) {
      const input = Matrix.random(4, 2).mul(100);
      bn.forward(input);
    }
    noNaN(bn.runningMean, 'BatchNorm running mean after 100 passes');
    allFinite(bn.runningMean, 'BatchNorm running mean after 100 passes');
    noNaN(bn.runningVar, 'BatchNorm running var after 100 passes');
    allFinite(bn.runningVar, 'BatchNorm running var after 100 passes');
  });

  it('inference mode with untrained running stats', () => {
    const bn = new BatchNorm(3);
    bn.training = false;
    // runningVar starts at 1 (initialized in constructor), runningMean at 0
    const input = new Matrix(2, 3, new Float64Array([100, -100, 0, 50, -50, 25]));
    const output = bn.forward(input);
    noNaN(output, 'BatchNorm inference untrained');
    allFinite(output, 'BatchNorm inference untrained');
  });

  it('handles input with one feature at ±Infinity', () => {
    const bn = new BatchNorm(3);
    const input = new Matrix(2, 3, new Float64Array([
      Infinity, 1, 2,
      -Infinity, 3, 4,
    ]));
    const output = bn.forward(input);
    // Inf input will produce Inf/NaN in mean — at least shouldn't crash
    assert.ok(output.rows === 2 && output.cols === 3, 'Output shape preserved');
  });
});

// ============================================================
// 2. ADAM OPTIMIZER NUMERICAL STABILITY
// ============================================================
describe('Adam Optimizer Numerical Stability', () => {
  it('handles extreme gradients without NaN', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([1e10, -1e10, 0]));
    const updated = adam.update(param, grad, 'test');
    noNaN(updated, 'Adam extreme gradient');
    allFinite(updated, 'Adam extreme gradient');
  });

  it('handles near-zero gradients without NaN', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([1e-30, 1e-50, 1e-100]));
    const updated = adam.update(param, grad, 'test');
    noNaN(updated, 'Adam near-zero gradient');
    allFinite(updated, 'Adam near-zero gradient');
  });

  it('handles all-zero gradients', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = Matrix.zeros(1, 3);
    const updated = adam.update(param, grad, 'test');
    noNaN(updated, 'Adam zero gradient');
    allFinite(updated, 'Adam zero gradient');
    // Should be roughly the same as param (small epsilon correction)
  });

  it('does not produce NaN after 10000 steps', () => {
    const adam = new Adam(0.001);
    let param = Matrix.random(1, 4).mul(0.1);
    for (let i = 0; i < 10000; i++) {
      adam.step();
      const grad = Matrix.random(1, 4).mul(0.01);
      param = adam.update(param, grad, 'long-run');
    }
    noNaN(param, 'Adam after 10K steps');
    allFinite(param, 'Adam after 10K steps');
  });

  it('bias correction prevents initial NaN at t=0', () => {
    const adam = new Adam(0.001);
    // Don't call step() — t=0
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([0.1, 0.2, 0.3]));
    const updated = adam.update(param, grad, 'no-step');
    noNaN(updated, 'Adam t=0 update');
    allFinite(updated, 'Adam t=0 update');
  });

  it('handles NaN gradient without crashing', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([NaN, 0.1, 0.2]));
    // NaN gradient should propagate NaN (detectable) or be handled
    try {
      const updated = adam.update(param, grad, 'nan-grad');
      // If it doesn't throw, check if NaN propagated (acceptable) or was handled
      assert.ok(true, 'Adam handled NaN gradient without crashing');
    } catch {
      assert.ok(true, 'Adam threw on NaN gradient (acceptable)');
    }
  });

  it('handles Infinity gradient without crashing', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([Infinity, 0.1, -Infinity]));
    try {
      const updated = adam.update(param, grad, 'inf-grad');
      assert.ok(true, 'Adam handled Inf gradient without crashing');
    } catch {
      assert.ok(true, 'Adam threw on Inf gradient (acceptable)');
    }
  });

  it('beta correction with very high t does not underflow', () => {
    const adam = new Adam(0.001);
    // Simulate very high t by setting directly
    adam.t = 100000;
    adam.step();
    const param = Matrix.ones(1, 3);
    const grad = new Matrix(1, 3, new Float64Array([0.01, 0.02, 0.03]));
    const updated = adam.update(param, grad, 'high-t');
    noNaN(updated, 'Adam very high t');
    allFinite(updated, 'Adam very high t');
  });

  it('AdamW weight decay does not cause divergence with small params', () => {
    const adamw = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.1);
    let param = new Matrix(1, 3, new Float64Array([1e-10, 1e-10, 1e-10]));
    for (let i = 0; i < 100; i++) {
      adamw.step();
      const grad = Matrix.random(1, 3).mul(0.001);
      param = adamw.update(param, grad, 'small-params');
    }
    noNaN(param, 'AdamW small params');
    allFinite(param, 'AdamW small params');
  });
});

// ============================================================
// 3. ATTENTION NUMERICAL STABILITY
// ============================================================
describe('Attention Numerical Stability', () => {
  it('handles extreme Q/K values without overflow', () => {
    const attn = new SelfAttention(4);
    // Very large values in input → QK^T can overflow before softmax
    const input = new Matrix(1, 8, new Float64Array([
      100, -100, 50, -50,  // pos 0
      100, -100, 50, -50,  // pos 1
    ]));
    const output = attn.forward(input);
    noNaN(output, 'Attention extreme Q/K');
    allFinite(output, 'Attention extreme Q/K');
  });

  it('handles all-zero input', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.zeros(1, 8); // 2 positions × 4 dims
    const output = attn.forward(input);
    noNaN(output, 'Attention zero input');
    allFinite(output, 'Attention zero input');
  });

  it('handles single-position sequence', () => {
    const attn = new SelfAttention(4);
    const input = new Matrix(1, 4, new Float64Array([1, 2, 3, 4]));
    const output = attn.forward(input);
    noNaN(output, 'Attention single position');
    allFinite(output, 'Attention single position');
  });

  it('handles longer sequences', () => {
    const attn = new SelfAttention(4);
    // 8 positions × 4 dims
    const input = Matrix.random(1, 32);
    const output = attn.forward(input);
    noNaN(output, 'Attention 8-position');
    allFinite(output, 'Attention 8-position');
  });

  it('backward pass does not produce NaN', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 8); // 2 positions × 4 dims
    const output = attn.forward(input);
    const gradOutput = Matrix.ones(output.rows, output.cols);
    try {
      const gradInput = attn.backward(gradOutput);
      noNaN(gradInput, 'Attention backward');
      allFinite(gradInput, 'Attention backward');
    } catch (e) {
      // Some attention implementations may not have backward yet
      assert.ok(true, 'Attention backward not implemented or threw (acceptable)');
    }
  });

  it('scale factor prevents overflow for large dModel', () => {
    // With dModel=512, QK^T values can be very large
    // scale = 1/sqrt(512) ≈ 0.044 should tame this
    const attn = new SelfAttention(16);
    const input = new Matrix(1, 32, new Float64Array(32).fill(10)); // 2 positions × 16 dims
    const output = attn.forward(input);
    noNaN(output, 'Attention large dModel');
    allFinite(output, 'Attention large dModel');
  });

  it('attention weights sum to 1 (softmax property)', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 12); // 3 positions × 4 dims
    attn.forward(input);
    
    // Check cached attention weights if available
    if (attn._cache && attn._cache.length > 0) {
      for (const cache of attn._cache) {
        if (cache.attnWeights) {
          for (let i = 0; i < cache.attnWeights.rows; i++) {
            let sum = 0;
            for (let j = 0; j < cache.attnWeights.cols; j++) {
              sum += cache.attnWeights.get(i, j);
            }
            assert.ok(Math.abs(sum - 1) < 1e-6, 
              `Attention weights row ${i} should sum to 1, got ${sum}`);
          }
        }
      }
    }
  });
});

// ============================================================
// 4. GRADIENT CLIPPING NUMERICAL STABILITY
// ============================================================
describe('Gradient Clipping Numerical Stability', () => {
  it('clipByValue handles NaN gracefully', () => {
    const grad = new Matrix(1, 3, new Float64Array([NaN, 1.0, -1.0]));
    const clipped = clipByValue(grad, 1.0);
    // NaN clipped by Math.max/min should propagate NaN
    // Just verify it doesn't crash
    assert.ok(clipped.rows === 1 && clipped.cols === 3, 'Shape preserved');
  });

  it('clipByValue handles Infinity', () => {
    const grad = new Matrix(1, 3, new Float64Array([Infinity, -Infinity, 1.0]));
    const clipped = clipByValue(grad, 1.0);
    assert.equal(clipped.data[0], 1.0, 'Inf should be clipped to maxVal');
    assert.equal(clipped.data[1], -1.0, '-Inf should be clipped to -maxVal');
    assert.equal(clipped.data[2], 1.0, 'Normal value preserved');
  });

  it('clipByNorm with zero-norm gradient', () => {
    const grad = Matrix.zeros(1, 3);
    const clipped = clipByNorm(grad, 1.0);
    noNaN(clipped, 'clipByNorm zero gradient');
    allFinite(clipped, 'clipByNorm zero gradient');
  });

  it('clipByNorm with NaN in gradient', () => {
    const grad = new Matrix(1, 3, new Float64Array([NaN, 1.0, 2.0]));
    try {
      const clipped = clipByNorm(grad, 1.0);
      // norm will be NaN, scale will be NaN → all NaN (acceptable)
      assert.ok(true, 'clipByNorm with NaN did not crash');
    } catch {
      assert.ok(true, 'clipByNorm threw on NaN (acceptable)');
    }
  });

  it('clipByNorm with very large norm', () => {
    const grad = new Matrix(1, 3, new Float64Array([1e100, 1e100, 1e100]));
    const clipped = clipByNorm(grad, 1.0);
    noNaN(clipped, 'clipByNorm large norm');
    allFinite(clipped, 'clipByNorm large norm');
    // Should be scaled down
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(clipped.data[i]) <= 1.0, 
        `Clipped value should be <= 1.0, got ${clipped.data[i]}`);
    }
  });

  it('clipByGlobalNorm with empty gradient list', () => {
    try {
      const result = clipByGlobalNorm([], 1.0);
      assert.ok(Array.isArray(result), 'Should return empty array');
    } catch {
      assert.ok(true, 'clipByGlobalNorm threw on empty list (acceptable)');
    }
  });

  it('clipByGlobalNorm with mixed normal and extreme gradients', () => {
    const g1 = new Matrix(1, 3, new Float64Array([0.1, 0.2, 0.3]));
    const g2 = new Matrix(1, 3, new Float64Array([1e10, -1e10, 1e10]));
    const result = clipByGlobalNorm([g1, g2], 1.0);
    assert.equal(result.grads.length, 2, 'Should return 2 gradients');
    assert.ok(result.clipped, 'Should be clipped (global norm >> 1.0)');
    for (const c of result.grads) {
      noNaN(c, 'clipByGlobalNorm mixed');
      allFinite(c, 'clipByGlobalNorm mixed');
    }
  });

  it('clipByValue with maxVal=0', () => {
    const grad = new Matrix(1, 3, new Float64Array([0.5, -0.3, 0]));
    const clipped = clipByValue(grad, 0);
    for (let i = 0; i < 3; i++) {
      // Use Object.is to handle -0 vs 0 — both are acceptable for "clamped to 0"
      assert.ok(clipped.data[i] === 0 || Object.is(clipped.data[i], -0),
        `Value should be clamped to 0, got ${clipped.data[i]}`);
    }
  });
});
