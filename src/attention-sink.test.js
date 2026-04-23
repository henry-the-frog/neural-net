import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Attention Sink', () => {
  // Attention Sink (Xiao et al., 2023): keep first tokens in KV cache
  function attentionSinkCache(keys, values, windowSize, sinkSize) {
    if (keys.length <= windowSize + sinkSize) return { keys, values };
    const sinkKeys = keys.slice(0, sinkSize);
    const windowKeys = keys.slice(-windowSize);
    return {
      keys: [...sinkKeys, ...windowKeys],
      values: [...values.slice(0, sinkSize), ...values.slice(-windowSize)],
    };
  }

  test('preserves sink tokens', () => {
    const keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const { keys: cached } = attentionSinkCache(keys, keys, 3, 2);
    assert.equal(cached[0], 0); // First sink
    assert.equal(cached[1], 1); // Second sink
    assert.equal(cached[cached.length - 1], 9); // Last window
  });

  test('small cache unchanged', () => {
    const keys = [1, 2, 3];
    const { keys: cached } = attentionSinkCache(keys, keys, 5, 2);
    assert.deepEqual(cached, keys);
  });
});
