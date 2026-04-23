import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { RotaryKVCache } from './rotary-cache.js';

describe('RotaryKVCache', () => {
  test('append grows cache', () => {
    const cache = new RotaryKVCache(10, 4);
    cache.append([1,2,3,4], [5,6,7,8]);
    assert.equal(cache.length, 1);
    cache.append([1,2,3,4], [5,6,7,8]);
    assert.equal(cache.length, 2);
  });

  test('evicts old entries at maxLen', () => {
    const cache = new RotaryKVCache(2, 4);
    cache.append([1], [1]);
    cache.append([2], [2]);
    cache.append([3], [3]);
    assert.equal(cache.length, 2);
    assert.deepEqual(cache.getKeys()[0], [2]); // First entry evicted
  });

  test('clear empties cache', () => {
    const cache = new RotaryKVCache(10, 4);
    cache.append([1], [1]);
    cache.clear();
    assert.equal(cache.length, 0);
  });
});
