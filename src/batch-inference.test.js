import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Batch Inference', () => {
  function continuousBatching(requests, maxBatch) {
    const batches = [];
    let current = [];
    for (const req of requests) {
      current.push(req);
      if (current.length >= maxBatch) {
        batches.push([...current]);
        current = [];
      }
    }
    if (current.length) batches.push(current);
    return batches;
  }

  test('splits into max batch size', () => {
    const reqs = [1,2,3,4,5];
    const batches = continuousBatching(reqs, 2);
    assert.equal(batches.length, 3);
    assert.equal(batches[0].length, 2);
  });

  test('handles partial last batch', () => {
    const batches = continuousBatching([1,2,3], 2);
    assert.equal(batches[batches.length - 1].length, 1);
  });
});
