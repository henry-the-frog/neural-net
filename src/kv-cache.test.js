// kv-cache.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { KVCache, ModelKVCache } from './kv-cache.js';
import { Matrix } from './matrix.js';

describe('KVCache', () => {
  it('starts empty', () => {
    const cache = new KVCache(2, 4, 64);
    assert.equal(cache.seqLen, 0);
    assert.equal(cache.size(), 0);
  });
  
  it('append single token', () => {
    const cache = new KVCache(2, 4);
    const keys = [Matrix.random(1, 4), Matrix.random(1, 4)];
    const vals = [Matrix.random(1, 4), Matrix.random(1, 4)];
    
    cache.append(keys, vals);
    assert.equal(cache.seqLen, 1);
    
    // Verify stored values
    const k0 = cache.getKeys(0);
    assert.equal(k0.rows, 1);
    assert.equal(k0.cols, 4);
    for (let d = 0; d < 4; d++) {
      assert.ok(Math.abs(k0.get(0, d) - keys[0].get(0, d)) < 1e-10);
    }
  });
  
  it('append multiple tokens', () => {
    const cache = new KVCache(1, 4);
    const keys = [Matrix.random(3, 4)];
    const vals = [Matrix.random(3, 4)];
    
    cache.appendMultiple(keys, vals);
    assert.equal(cache.seqLen, 3);
    
    const k = cache.getKeys(0);
    assert.equal(k.rows, 3);
  });
  
  it('incremental append matches bulk', () => {
    const cache1 = new KVCache(1, 4);
    const cache2 = new KVCache(1, 4);
    
    const allKeys = Matrix.random(5, 4);
    const allVals = Matrix.random(5, 4);
    
    // Bulk append
    cache1.appendMultiple([allKeys], [allVals]);
    
    // Incremental append
    for (let t = 0; t < 5; t++) {
      const k = new Matrix(1, 4);
      const v = new Matrix(1, 4);
      for (let d = 0; d < 4; d++) {
        k.set(0, d, allKeys.get(t, d));
        v.set(0, d, allVals.get(t, d));
      }
      cache2.append([k], [v]);
    }
    
    // Should be identical
    const k1 = cache1.getKeys(0);
    const k2 = cache2.getKeys(0);
    for (let t = 0; t < 5; t++)
      for (let d = 0; d < 4; d++)
        assert.ok(Math.abs(k1.get(t, d) - k2.get(t, d)) < 1e-10);
  });
  
  it('size calculation', () => {
    const cache = new KVCache(4, 8); // 4 heads, 8 dim
    cache.appendMultiple(
      [Matrix.random(10, 8), Matrix.random(10, 8), Matrix.random(10, 8), Matrix.random(10, 8)],
      [Matrix.random(10, 8), Matrix.random(10, 8), Matrix.random(10, 8), Matrix.random(10, 8)]
    );
    // size = 2 * seqLen * numKVHeads * headDim = 2 * 10 * 4 * 8 = 640
    assert.equal(cache.size(), 640);
  });
  
  it('reset clears cache', () => {
    const cache = new KVCache(1, 4);
    cache.append([Matrix.random(1, 4)], [Matrix.random(1, 4)]);
    assert.equal(cache.seqLen, 1);
    cache.reset();
    assert.equal(cache.seqLen, 0);
    assert.equal(cache.size(), 0);
  });
  
  it('clone creates independent copy', () => {
    const cache = new KVCache(1, 4);
    cache.append([Matrix.random(1, 4)], [Matrix.random(1, 4)]);
    
    const clone = cache.clone();
    assert.equal(clone.seqLen, 1);
    
    // Modify original
    cache.append([Matrix.random(1, 4)], [Matrix.random(1, 4)]);
    assert.equal(cache.seqLen, 2);
    assert.equal(clone.seqLen, 1); // Clone unchanged
  });
  
  it('throws on overflow', () => {
    const cache = new KVCache(1, 4, 3);
    cache.appendMultiple([Matrix.random(3, 4)], [Matrix.random(3, 4)]);
    assert.throws(() => cache.append([Matrix.random(1, 4)], [Matrix.random(1, 4)]), /full/);
  });
});

describe('ModelKVCache', () => {
  it('creates multi-layer cache', () => {
    const cache = new ModelKVCache(6, 4, 8, 1024); // 6 layers, 4 heads, 8 dim
    assert.equal(cache.layers.length, 6);
    assert.equal(cache.seqLen(), 0);
  });
  
  it('totalSize sums all layers', () => {
    const cache = new ModelKVCache(3, 2, 4);
    for (let l = 0; l < 3; l++) {
      cache.getLayer(l).append(
        [Matrix.random(1, 4), Matrix.random(1, 4)],
        [Matrix.random(1, 4), Matrix.random(1, 4)]
      );
    }
    // 3 layers * 2 * 1 * 2 * 4 = 48
    assert.equal(cache.totalSize(), 48);
  });
  
  it('reset clears all layers', () => {
    const cache = new ModelKVCache(3, 1, 4);
    cache.getLayer(0).append([Matrix.random(1, 4)], [Matrix.random(1, 4)]);
    cache.reset();
    assert.equal(cache.seqLen(), 0);
  });
});
