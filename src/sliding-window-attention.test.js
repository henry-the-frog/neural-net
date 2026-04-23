// sliding-window-attention.test.js — Sliding Window Attention tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { slidingWindowAttention, effectiveReceptiveField } from './sliding-window-attention.js';
import { standardAttention } from './flash-attention.js';
import { Matrix } from './matrix.js';

describe('Sliding Window Attention', () => {
  test('output has correct shape', () => {
    const Q = Matrix.random(8, 4);
    const K = Matrix.random(8, 4);
    const V = Matrix.random(8, 4);
    const out = slidingWindowAttention(Q, K, V, 3);
    assert.equal(out.rows, 8);
    assert.equal(out.cols, 4);
  });

  test('window=N matches causal attention', () => {
    const N = 6, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    const swa = slidingWindowAttention(Q, K, V, N); // Full window = causal
    const causal = standardAttention(Q, K, V, true);
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < d; j++) {
        assert.ok(Math.abs(swa.get(i, j) - causal.get(i, j)) < 0.01,
          `SWA should match causal at (${i},${j}): ${swa.get(i,j)} vs ${causal.get(i,j)}`);
      }
    }
  });

  test('window=1 only attends to self', () => {
    const N = 4, d = 2;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = new Matrix(N, d);
    for (let i = 0; i < N; i++) V.set(i, 0, i); // V[i] = [i, 0]
    
    const out = slidingWindowAttention(Q, K, V, 1);
    // Each position only attends to itself → output = V[i]
    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(out.get(i, 0) - i) < 0.01, 
        `Window=1: position ${i} should output ${i}, got ${out.get(i,0)}`);
    }
  });

  test('position 0 always only attends to itself', () => {
    const N = 8, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = new Matrix(N, d);
    V.set(0, 0, 42); // V[0] = [42, 0, 0, 0]
    
    const out = slidingWindowAttention(Q, K, V, 3);
    // Position 0 always has window [0,0] regardless of window size
    assert.ok(Math.abs(out.get(0, 0) - 42) < 0.01);
  });

  test('later positions attend to window of positions', () => {
    const N = 6, d = 2;
    const Q = Matrix.ones(N, d); // Equal queries
    const K = Matrix.ones(N, d); // Equal keys → uniform attention
    const V = new Matrix(N, d);
    for (let i = 0; i < N; i++) V.set(i, 0, i);
    
    const out = slidingWindowAttention(Q, K, V, 3);
    // Position 5, window [3,4,5], uniform attention → mean(3,4,5) = 4
    assert.ok(Math.abs(out.get(5, 0) - 4) < 0.01,
      `Position 5 with window 3 should be ~4, got ${out.get(5,0)}`);
  });

  test('effectiveReceptiveField calculation', () => {
    const result = effectiveReceptiveField(4096, 32);
    assert.equal(result.receptiveField, 131072);
    assert.ok(result.description.includes('131072'));
  });

  test('smaller window = less compute (conceptual)', () => {
    // Window=2 only computes 2 attention scores per position
    // Window=N computes N scores per position
    const N = 100, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    // Both should produce valid outputs
    const small = slidingWindowAttention(Q, K, V, 4);
    const large = slidingWindowAttention(Q, K, V, N);
    assert.equal(small.rows, N);
    assert.equal(large.rows, N);
  });
});
