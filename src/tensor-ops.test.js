import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { dotProduct, outerProduct, hadamard, l2Norm, normalize } from './tensor-ops.js';

describe('Tensor Ops', () => {
  test('dotProduct', () => assert.equal(dotProduct([1,2,3], [4,5,6]), 32));
  test('outerProduct shape', () => {
    const r = outerProduct([1,2], [3,4,5]);
    assert.equal(r.length, 2);
    assert.equal(r[0].length, 3);
  });
  test('hadamard', () => assert.deepEqual(hadamard([2,3], [4,5]), [8, 15]));
  test('l2Norm', () => assert.ok(Math.abs(l2Norm([3,4]) - 5) < 0.01));
  test('normalize gives unit vector', () => {
    const n = normalize([3, 4]);
    assert.ok(Math.abs(l2Norm(n) - 1) < 0.01);
  });
});
