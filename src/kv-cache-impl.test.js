// kv-cache-impl.test.js — KV Cache tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { KVCache } from './kv-cache-impl.js';

describe('KV Cache', () => {
  test('initial seqLen is 0', () => {
    const cache = new KVCache(128, 2, 4, 64);
    assert.equal(cache.seqLen, 0);
  });

  test('append and retrieve keys', () => {
    const cache = new KVCache(128, 1, 1, 4);
    const k1 = new Float64Array([1, 2, 3, 4]);
    const v1 = new Float64Array([5, 6, 7, 8]);
    
    cache.append(0, k1, v1);
    cache.incrementSeqLen();
    
    const keys = cache.getKeys(0);
    assert.equal(keys.rows, 1);
    assert.equal(keys.cols, 4);
    assert.ok(Math.abs(keys.get(0, 0) - 1) < 0.001);
    assert.ok(Math.abs(keys.get(0, 3) - 4) < 0.001);
  });

  test('append multiple tokens', () => {
    const cache = new KVCache(128, 1, 1, 2);
    
    cache.append(0, new Float64Array([1, 2]), new Float64Array([3, 4]));
    cache.incrementSeqLen();
    cache.append(0, new Float64Array([5, 6]), new Float64Array([7, 8]));
    cache.incrementSeqLen();
    
    const keys = cache.getKeys(0);
    assert.equal(keys.rows, 2);
    assert.ok(Math.abs(keys.get(0, 0) - 1) < 0.001);
    assert.ok(Math.abs(keys.get(1, 0) - 5) < 0.001);
    
    const vals = cache.getValues(0);
    assert.ok(Math.abs(vals.get(0, 0) - 3) < 0.001);
    assert.ok(Math.abs(vals.get(1, 0) - 7) < 0.001);
  });

  test('multiple layers are independent', () => {
    const cache = new KVCache(128, 2, 1, 2);
    
    cache.append(0, new Float64Array([1, 0]), new Float64Array([0, 0]));
    cache.append(1, new Float64Array([0, 2]), new Float64Array([0, 0]));
    cache.incrementSeqLen();
    
    const k0 = cache.getKeys(0);
    const k1 = cache.getKeys(1);
    assert.ok(Math.abs(k0.get(0, 0) - 1) < 0.001);
    assert.ok(Math.abs(k1.get(0, 1) - 2) < 0.001);
  });

  test('reset clears cache', () => {
    const cache = new KVCache(128, 1, 1, 2);
    cache.append(0, new Float64Array([1, 2]), new Float64Array([3, 4]));
    cache.incrementSeqLen();
    assert.equal(cache.seqLen, 1);
    
    cache.reset();
    assert.equal(cache.seqLen, 0);
  });

  test('throws on overflow', () => {
    const cache = new KVCache(2, 1, 1, 2);
    cache.append(0, new Float64Array([1, 2]), new Float64Array([3, 4]));
    cache.incrementSeqLen();
    cache.append(0, new Float64Array([5, 6]), new Float64Array([7, 8]));
    cache.incrementSeqLen();
    
    assert.throws(() => cache.append(0, new Float64Array([9, 10]), new Float64Array([11, 12])), /full/);
  });

  test('memoryBytes calculation', () => {
    const cache = new KVCache(1024, 32, 8, 64);
    // 32 layers * 2 (K+V) * 1024 * 8*64 * 8 bytes
    const expected = 32 * 2 * 1024 * 512 * 8;
    assert.equal(cache.memoryBytes(), expected);
  });

  test('utilization tracking', () => {
    const cache = new KVCache(100, 1, 1, 2);
    assert.equal(cache.utilization(), '0.0%');
    
    for (let i = 0; i < 50; i++) {
      cache.append(0, new Float64Array([i, 0]), new Float64Array([0, i]));
      cache.incrementSeqLen();
    }
    assert.equal(cache.utilization(), '50.0%');
  });
});
