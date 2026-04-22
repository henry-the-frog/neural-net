// sliding-window.test.js — Tests for Sliding Window Attention
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { slidingWindowAttention, SlidingWindowKVCache } from './sliding-window.js';
import { standardAttention } from './flash-attention.js';
import { Matrix } from './matrix.js';

describe('Sliding Window Attention', () => {
  describe('correctness', () => {
    it('window=N matches full causal attention', () => {
      const N = 6, d = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const full = standardAttention(Q, K, V, true);
      const swa = slidingWindowAttention(Q, K, V, N, true);

      assertClose(full, swa.output, 1e-6, 'Full window should match full attention');
    });

    it('window=1: each token only attends to itself', () => {
      const N = 4, d = 2;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const swa = slidingWindowAttention(Q, K, V, 1, true);

      // Each output row should equal the corresponding V row
      for (let i = 0; i < N; i++)
        for (let dd = 0; dd < d; dd++)
          assert.ok(Math.abs(swa.output.get(i, dd) - V.get(i, dd)) < 1e-6,
            `Window=1: output should equal V at (${i},${dd})`);
    });

    it('small window restricts attention span', () => {
      const N = 8, d = 4, W = 3;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const full = standardAttention(Q, K, V, true);
      const swa = slidingWindowAttention(Q, K, V, W, true);

      // First W positions should match (they see the same context)
      for (let i = 0; i < W; i++)
        for (let dd = 0; dd < d; dd++)
          assert.ok(Math.abs(full.get(i, dd) - swa.output.get(i, dd)) < 1e-6,
            `First W positions should match at (${i},${dd})`);

      // Later positions should differ (SWA can't see distant tokens)
      let diff = 0;
      for (let i = W; i < N; i++)
        for (let dd = 0; dd < d; dd++)
          diff += Math.abs(full.get(i, dd) - swa.output.get(i, dd));
      assert.ok(diff > 0.01, 'Later positions should differ from full attention');
    });

    it('produces finite values', () => {
      const N = 10, d = 4, W = 4;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const swa = slidingWindowAttention(Q, K, V, W, true);
      for (let i = 0; i < N; i++)
        for (let dd = 0; dd < d; dd++)
          assert.ok(isFinite(swa.output.get(i, dd)));
    });
  });

  describe('memory efficiency', () => {
    it('SWA uses O(N*W) vs O(N²) memory', () => {
      const N = 64, d = 4, W = 8;
      const Q = Matrix.random(N, d);
      const K = Matrix.random(N, d);
      const V = Matrix.random(N, d);

      const swa = slidingWindowAttention(Q, K, V, W, true);

      const fullPeakMemory = N * N; // Full attention materializes N×N matrix
      console.log(`  Full: ${fullPeakMemory} elements`);
      console.log(`  SWA: ${swa.stats.peakMemory} elements (${(swa.stats.peakMemory/fullPeakMemory*100).toFixed(0)}%)`);

      assert.ok(swa.stats.peakMemory < fullPeakMemory);
    });
  });
});

describe('SlidingWindowKVCache', () => {
  it('stores up to window size', () => {
    const cache = new SlidingWindowKVCache(4, 2);
    for (let i = 0; i < 3; i++) cache.append([i, i], [i, i]);
    assert.equal(cache.size, 3);
  });

  it('evicts oldest when over window', () => {
    const cache = new SlidingWindowKVCache(3, 2);
    for (let i = 0; i < 5; i++) cache.append([i, i], [i, i]);
    
    assert.equal(cache.size, 3);
    const stats = cache.stats();
    assert.equal(stats.totalTokensSeen, 5);
    assert.equal(stats.evicted, 2);

    // Oldest keys should be evicted
    const keys = cache.getKeys();
    assert.equal(keys.get(0, 0), 2); // first remaining is token 2
  });

  it('getKeys/getValues return correct matrices', () => {
    const cache = new SlidingWindowKVCache(10, 3);
    cache.append([1, 2, 3], [4, 5, 6]);
    cache.append([7, 8, 9], [10, 11, 12]);

    const K = cache.getKeys();
    const V = cache.getValues();
    assert.equal(K.rows, 2);
    assert.equal(K.cols, 3);
    assert.equal(K.get(0, 0), 1);
    assert.equal(V.get(1, 2), 12);
  });

  it('clear resets everything', () => {
    const cache = new SlidingWindowKVCache(5, 2);
    cache.append([1, 2], [3, 4]);
    cache.clear();
    assert.equal(cache.size, 0);
    assert.equal(cache.stats().totalTokensSeen, 0);
  });

  it('memory is bounded', () => {
    const cache = new SlidingWindowKVCache(4, 8);
    for (let i = 0; i < 1000; i++) {
      cache.append(new Array(8).fill(i), new Array(8).fill(i));
    }
    assert.equal(cache.size, 4, 'Cache should never exceed window size');
    const stats = cache.stats();
    assert.equal(stats.memoryElements, 4 * 8 * 2, 'Memory should be bounded');
    console.log(`  1000 tokens, window=4: cache=${stats.cached}, evicted=${stats.evicted}`);
  });
});

function assertClose(a, b, tolerance, label) {
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++) {
      const diff = Math.abs(a.get(i, j) - b.get(i, j));
      assert.ok(diff < tolerance, `${label}: (${i},${j}): ${a.get(i, j)} vs ${b.get(i, j)}`);
    }
}
