import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Model Merging', () => {
  function lerp(w1, w2, alpha = 0.5) {
    return w1.map((v, i) => v * (1 - alpha) + w2[i] * alpha);
  }

  function slerp(w1, w2, t = 0.5) {
    const dot = w1.reduce((s, v, i) => s + v * w2[i], 0);
    const n1 = Math.sqrt(w1.reduce((s, v) => s + v * v, 0));
    const n2 = Math.sqrt(w2.reduce((s, v) => s + v * v, 0));
    const theta = Math.acos(Math.min(1, dot / (n1 * n2 + 1e-8)));
    if (theta < 1e-6) return lerp(w1, w2, t);
    const s = Math.sin(theta);
    return w1.map((v, i) => (Math.sin((1-t)*theta)/s) * v + (Math.sin(t*theta)/s) * w2[i]);
  }

  test('lerp at 0 returns first', () => {
    const result = lerp([1,2,3], [4,5,6], 0);
    assert.deepEqual(result, [1,2,3]);
  });

  test('lerp at 0.5 averages', () => {
    const result = lerp([0,0], [2,2], 0.5);
    assert.deepEqual(result, [1,1]);
  });

  test('slerp is smooth interpolation', () => {
    const result = slerp([1,0], [0,1], 0.5);
    assert.ok(result[0] > 0 && result[1] > 0);
  });
});
