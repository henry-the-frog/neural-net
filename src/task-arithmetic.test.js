import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Task Arithmetic', () => {
  // Task vectors: τ = θ_finetuned - θ_base (Ilharco et al., 2023)
  function taskVector(base, finetuned) {
    return base.map((v, i) => finetuned[i] - v);
  }

  function applyTaskVector(base, vector, scale = 1.0) {
    return base.map((v, i) => v + scale * vector[i]);
  }

  function negateTask(base, vector) {
    return applyTaskVector(base, vector, -1.0);
  }

  test('task vector captures difference', () => {
    const tv = taskVector([1,2,3], [2,4,6]);
    assert.deepEqual(tv, [1,2,3]);
  });

  test('apply task vector modifies base', () => {
    const base = [1, 1, 1];
    const tv = [0.5, 0.5, 0.5];
    const result = applyTaskVector(base, tv, 1.0);
    assert.deepEqual(result, [1.5, 1.5, 1.5]);
  });

  test('negate undoes the task', () => {
    const base = [1, 2, 3];
    const ft = [2, 4, 6];
    const tv = taskVector(base, ft);
    const negated = negateTask(ft, tv);
    assert.deepEqual(negated, base);
  });
});
