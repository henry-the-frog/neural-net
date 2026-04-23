import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Semantic Deduplication', () => {
  function deduplicate(items, similarityFn, threshold = 0.9) {
    const kept = [];
    for (const item of items) {
      const isDup = kept.some(k => similarityFn(k, item) > threshold);
      if (!isDup) kept.push(item);
    }
    return kept;
  }

  test('removes duplicates', () => {
    const items = ['hello', 'hello', 'world'];
    const result = deduplicate(items, (a, b) => a === b ? 1 : 0, 0.9);
    assert.equal(result.length, 2);
  });

  test('keeps unique items', () => {
    const items = ['a', 'b', 'c'];
    const result = deduplicate(items, () => 0);
    assert.equal(result.length, 3);
  });
});
