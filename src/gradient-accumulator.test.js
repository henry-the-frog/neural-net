import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { GradientAccumulator } from './gradient-accumulator.js';

describe('Gradient Accumulator', () => {
  test('accumulates over N steps', () => {
    const acc = new GradientAccumulator(3, 2);
    const r1 = acc.accumulate(new Float64Array([1, 2, 3]));
    assert.ok(!r1.ready);
    const r2 = acc.accumulate(new Float64Array([3, 2, 1]));
    assert.ok(r2.ready);
    assert.ok(Math.abs(r2.gradients[0] - 2) < 1e-10); // (1+3)/2
  });

  test('resets after flush', () => {
    const acc = new GradientAccumulator(2, 2);
    acc.accumulate(new Float64Array([1, 1]));
    acc.accumulate(new Float64Array([1, 1]));
    const r = acc.accumulate(new Float64Array([10, 10]));
    assert.ok(!r.ready); // Reset, this is step 1 of new cycle
  });
});
