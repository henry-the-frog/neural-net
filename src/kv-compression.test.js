import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('KV Cache Compression', () => {
  function compressKVCache(keys, ratio = 0.5) {
    // Simple eviction: keep every nth key
    const keepEvery = Math.ceil(1 / (1 - ratio));
    return keys.filter((_, i) => i % keepEvery === 0);
  }

  test('compresses by ratio', () => {
    const keys = [0,1,2,3,4,5,6,7,8,9];
    const compressed = compressKVCache(keys, 0.5);
    assert.ok(compressed.length < keys.length);
  });

  test('preserves some keys', () => {
    const keys = [0,1,2,3];
    const compressed = compressKVCache(keys, 0.5);
    assert.ok(compressed.length > 0);
    assert.ok(compressed.includes(0));
  });
});
