// flash-attention.test.js — Tests for Flash Attention implementation
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { standardAttention, flashAttention } from './flash-attention.js';
import { Matrix } from './matrix.js';

describe('Flash Attention', () => {
  describe('correctness: matches standard attention', () => {
    it('non-causal, small sequence', () => {
      const N = 4, d = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const std = standardAttention(Q, K, V, false);
      const flash = flashAttention(Q, K, V, false, 2);

      assertMatricesClose(std.output, flash.output, 1e-6, 'Non-causal');
    });

    it('causal, small sequence', () => {
      const N = 4, d = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const std = standardAttention(Q, K, V, true);
      const flash = flashAttention(Q, K, V, true, 2);

      assertMatricesClose(std.output, flash.output, 1e-6, 'Causal');
    });

    it('large sequence with various tile sizes', () => {
      const N = 16, d = 8;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const std = standardAttention(Q, K, V, true);

      for (const tileSize of [1, 2, 4, 8, 16]) {
        const flash = flashAttention(Q, K, V, true, tileSize);
        assertMatricesClose(std.output, flash.output, 1e-5, `TileSize=${tileSize}`);
      }
    });

    it('single token', () => {
      const Q = Matrix.random(1, 4);
      const K = Matrix.random(1, 4);
      const V = Matrix.random(1, 4);

      const std = standardAttention(Q, K, V, true);
      const flash = flashAttention(Q, K, V, true, 1);

      // Single token: output should equal V (attention to self = 1)
      assertMatricesClose(flash.output, V, 1e-6, 'Single token');
    });

    it('identity attention (Q=K, uniform V)', () => {
      const N = 4, d = 4;
      const Q = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const std = standardAttention(Q, Q, V, false);
      const flash = flashAttention(Q, Q, V, false, 2);

      assertMatricesClose(std.output, flash.output, 1e-6, 'Q=K');
    });
  });

  describe('memory efficiency', () => {
    it('flash uses less peak memory than standard', () => {
      const N = 32, d = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const std = standardAttention(Q, K, V, false);
      const flash = flashAttention(Q, K, V, false, 4);

      console.log(`  Standard: ${std.stats.peakMemory} elements (${N}×${N})`);
      console.log(`  Flash: ${flash.stats.peakMemory} elements (${N}×${flash.stats.tileSize})`);

      assert.ok(flash.stats.peakMemory < std.stats.peakMemory,
        `Flash should use less memory: ${flash.stats.peakMemory} < ${std.stats.peakMemory}`);
    });

    it('memory scales linearly with tile size, not quadratically', () => {
      const N = 64, d = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const flash4 = flashAttention(Q, K, V, false, 4);
      const flash8 = flashAttention(Q, K, V, false, 8);

      // Doubling tile size should roughly double peak memory
      const ratio = flash8.stats.peakMemory / flash4.stats.peakMemory;
      assert.ok(ratio < 3, `Memory should scale linearly: ratio ${ratio}`);
    });
  });

  describe('numerical stability', () => {
    it('handles large logits without overflow', () => {
      const N = 4, d = 2;
      const Q = new Matrix(N, d);
      const K = new Matrix(N, d);
      const V = Matrix.random(N, d);

      // Large values that would overflow naive exp()
      for (let i = 0; i < N; i++) {
        Q.set(i, 0, 100 * (i + 1));
        Q.set(i, 1, 0);
        K.set(i, 0, 100 * (i + 1));
        K.set(i, 1, 0);
      }

      const std = standardAttention(Q, K, V, false);
      const flash = flashAttention(Q, K, V, false, 2);

      // Both should produce finite values
      for (let i = 0; i < N; i++)
        for (let j = 0; j < d; j++) {
          assert.ok(isFinite(flash.output.get(i, j)), `Flash NaN at (${i},${j})`);
          assert.ok(isFinite(std.output.get(i, j)), `Standard NaN at (${i},${j})`);
        }

      assertMatricesClose(std.output, flash.output, 1e-3, 'Large logits');
    });
  });
});

function assertMatricesClose(a, b, tolerance, label) {
  assert.equal(a.rows, b.rows, `${label}: rows mismatch`);
  assert.equal(a.cols, b.cols, `${label}: cols mismatch`);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      const diff = Math.abs(a.get(i, j) - b.get(i, j));
      assert.ok(diff < tolerance,
        `${label}: mismatch at (${i},${j}): ${a.get(i, j)} vs ${b.get(i, j)} (diff=${diff})`);
    }
  }
}
