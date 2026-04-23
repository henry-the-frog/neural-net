import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { linearInterpolation, dynamicNTK, gradualScaling } from './position-interpolation.js';

describe('Position Interpolation', () => {
  test('linear: halves position with scale 2', () => {
    assert.equal(linearInterpolation(100, 2), 50);
  });

  test('dynamic NTK: no scaling within train length', () => {
    assert.equal(dynamicNTK(50, 4096, 2048), 50);
  });

  test('dynamic NTK: scales beyond train length', () => {
    const scaled = dynamicNTK(100, 4096, 8192);
    assert.ok(scaled < 100);
  });

  test('gradual: no scaling in safe zone', () => {
    assert.equal(gradualScaling(100, 4096, 0.8), 100);
  });
});
