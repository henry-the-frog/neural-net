import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { magnitudePrune, computeSparsity, topKSparsify } from './structured-pruning.js';

describe('Pruning', () => {
  test('magnitude prune zeros small weights', () => {
    const pruned = magnitudePrune([0.1, 5, 0.01, 3, 0.001], 0.5);
    assert.ok(pruned[0] === 0 || pruned[2] === 0);
  });

  test('computeSparsity counts zeros', () => {
    assert.equal(computeSparsity([0, 1, 0, 1, 0]), 0.6);
  });

  test('topK keeps only K largest', () => {
    const sparse = topKSparsify([1, 5, 2, 4, 3], 2);
    const nonzero = sparse.filter(v => v !== 0);
    assert.equal(nonzero.length, 2);
    assert.ok(sparse[1] === 5); // Largest
    assert.ok(sparse[3] === 4); // Second largest
  });
});
