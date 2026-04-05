import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { clipByValue, clipByNorm, clipByGlobalNorm, l2Norm, gradientStats } from '../src/gradient-clip.js';

function mat(rows, cols, values) {
  const m = new Matrix(rows, cols);
  for (let i = 0; i < values.length; i++) m.data[i] = values[i];
  return m;
}

describe('Gradient Clipping', () => {
  describe('clipByValue', () => {
    it('clips large values', () => {
      const grad = mat(1, 4, [10, -20, 5, -1]);
      const clipped = clipByValue(grad, 5);
      assert.equal(clipped.data[0], 5);
      assert.equal(clipped.data[1], -5);
      assert.equal(clipped.data[2], 5);
      assert.equal(clipped.data[3], -1);
    });
    it('passes small values through', () => {
      const grad = mat(1, 3, [1, -1, 0.5]);
      const clipped = clipByValue(grad, 5);
      assert.equal(clipped.data[0], 1);
      assert.equal(clipped.data[1], -1);
    });
  });

  describe('clipByNorm', () => {
    it('scales down large gradients', () => {
      const grad = mat(1, 3, [6, 8, 0]); // norm = 10
      const clipped = clipByNorm(grad, 5);
      const norm = l2Norm(clipped);
      assert.ok(Math.abs(norm - 5) < 0.01, `Norm should be 5, got ${norm}`);
    });
    it('passes small gradients through', () => {
      const grad = mat(1, 3, [1, 1, 1]); // norm ≈ 1.73
      const clipped = clipByNorm(grad, 5);
      assert.equal(clipped.data[0], 1);
    });
    it('preserves direction', () => {
      const grad = mat(1, 2, [6, 8]); // ratio 3:4
      const clipped = clipByNorm(grad, 5);
      const ratio = clipped.data[0] / clipped.data[1];
      assert.ok(Math.abs(ratio - 0.75) < 0.01);
    });
  });

  describe('clipByGlobalNorm', () => {
    it('clips multiple gradients together', () => {
      const grads = [
        mat(1, 2, [3, 4]),  // norm = 5
        mat(1, 2, [6, 8]),  // norm = 10
      ]; // global norm ≈ sqrt(25 + 100) ≈ 11.18
      
      const result = clipByGlobalNorm(grads, 5);
      assert.ok(result.clipped);
      
      // Recompute global norm of clipped
      let normSq = 0;
      for (const g of result.grads) {
        for (let i = 0; i < g.data.length; i++) normSq += g.data[i] ** 2;
      }
      assert.ok(Math.abs(Math.sqrt(normSq) - 5) < 0.01);
    });
    it('returns original if norm is small', () => {
      const grads = [mat(1, 2, [0.1, 0.1])];
      const result = clipByGlobalNorm(grads, 5);
      assert.ok(!result.clipped);
      assert.equal(result.grads[0].data[0], 0.1);
    });
  });

  describe('l2Norm', () => {
    it('computes L2 norm', () => {
      const m = mat(1, 3, [3, 4, 0]);
      assert.equal(l2Norm(m), 5);
    });
    it('handles zeros', () => {
      const m = mat(1, 3, [0, 0, 0]);
      assert.equal(l2Norm(m), 0);
    });
  });

  describe('gradientStats', () => {
    it('computes basic statistics', () => {
      const grad = mat(1, 4, [1, 2, 3, 4]);
      const stats = gradientStats(grad);
      assert.equal(stats.min, 1);
      assert.equal(stats.max, 4);
      assert.equal(stats.mean, 2.5);
      assert.ok(stats.std > 0);
      assert.ok(stats.norm > 0);
    });
    it('handles negative values', () => {
      const grad = mat(1, 3, [-5, 0, 5]);
      const stats = gradientStats(grad);
      assert.equal(stats.min, -5);
      assert.equal(stats.max, 5);
      assert.ok(Math.abs(stats.mean) < 0.01);
    });
  });
});
