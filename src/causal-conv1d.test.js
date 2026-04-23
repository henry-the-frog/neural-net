import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { causalConv1d, depthwiseCausalConv } from './causal-conv1d.js';

describe('Causal Conv1D', () => {
  test('output same length as input', () => {
    const out = causalConv1d([1,2,3,4,5], [1, 0.5]);
    assert.equal(out.length, 5);
  });

  test('causal: output at t depends only on past', () => {
    const out = causalConv1d([0,0,0,1,0], [1, 0.5, 0.25]);
    // Impulse at t=3 should not affect t=0,1,2
    assert.equal(out[0], 0);
    assert.equal(out[1], 0);
    assert.equal(out[2], 0);
    assert.ok(out[3] > 0);
  });

  test('identity kernel preserves input', () => {
    const input = [1, 2, 3, 4];
    const out = causalConv1d(input, [1]);
    for (let i = 0; i < 4; i++) assert.equal(out[i], input[i]);
  });

  test('depthwise applies per channel', () => {
    const channels = [[1,2,3], [4,5,6]];
    const out = depthwiseCausalConv(channels, [1]);
    assert.equal(out.length, 2);
    assert.equal(out[0].length, 3);
  });
});
