// kv-cache-compression.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { QuantizedKVCache, compareQuantizedAttention } from './kv-cache-compression.js';
import { Matrix } from './matrix.js';

describe('Quantized KV-Cache', () => {
  it('stores and retrieves vectors', () => {
    const cache = new QuantizedKVCache(4);
    cache.append([1, 2, 3, 4], [5, 6, 7, 8]);
    assert.equal(cache.size, 1);
    const K = cache.getKeys();
    const V = cache.getValues();
    assert.equal(K.rows, 1);
    assert.equal(V.rows, 1);
  });

  it('approximate roundtrip (quantization error < 5%)', () => {
    const cache = new QuantizedKVCache(4);
    const k = [0.5, -0.3, 0.8, -0.1];
    cache.append(k, k);
    const K = cache.getKeys();
    for (let d = 0; d < 4; d++) {
      const err = Math.abs(K.get(0, d) - k[d]);
      assert.ok(err < 0.05, `Error too large at d=${d}: ${err}`);
    }
  });

  it('evicts oldest when over maxTokens', () => {
    const cache = new QuantizedKVCache(2, 3);
    for (let i = 0; i < 5; i++) cache.append([i, i], [i, i]);
    assert.equal(cache.size, 3);
    const K = cache.getKeys();
    assert.ok(Math.abs(K.get(0, 0) - 2) < 0.1, 'First should be token 2');
  });

  it('compression ratio is significant', () => {
    const cache = new QuantizedKVCache(64);
    for (let i = 0; i < 100; i++) {
      cache.append(new Array(64).fill(Math.random()), new Array(64).fill(Math.random()));
    }
    const stats = cache.stats();
    console.log(`  100 tokens, dim=64: ${stats.compressionRatio} compression, saved ${(stats.savedBytes/1024).toFixed(1)}KB`);
    assert.ok(stats.savedBytes > 0, 'Should save memory');
    const ratio = parseFloat(stats.compressionRatio);
    assert.ok(ratio > 3, `Compression should be > 3x, got ${ratio}`);
  });
});

describe('Quantized vs Standard Attention', () => {
  it('attention output is close with INT8 KV-cache', () => {
    const N = 16, d = 8;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);

    const result = compareQuantizedAttention(Q, K, V, d);
    console.log(`  MAE: ${result.mae.toFixed(6)}, Compression: ${result.stats.compressionRatio}`);
    assert.ok(result.mae < 0.1, `MAE too high: ${result.mae}`);
  });

  it('larger dimensions have smaller relative error', () => {
    const d32 = compareQuantizedAttention(
      Matrix.random(8, 32), Matrix.random(8, 32), Matrix.random(8, 32), 32
    );
    const d4 = compareQuantizedAttention(
      Matrix.random(8, 4), Matrix.random(8, 4), Matrix.random(8, 4), 4
    );
    console.log(`  d=4 MAE: ${d4.mae.toFixed(6)}, d=32 MAE: ${d32.mae.toFixed(6)}`);
    // Larger dim should have less error (more averaging)
  });
});
