import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { layerNorm, rmsNorm, compareNorms } from './normalization.js';
import { Matrix } from './matrix.js';

describe('Normalization', () => {
  test('layerNorm: zero mean per row', () => {
    const x = new Matrix(2, 4);
    x.set(0, 0, 1); x.set(0, 1, 2); x.set(0, 2, 3); x.set(0, 3, 4);
    const normed = layerNorm(x);
    let mean = 0;
    for (let j = 0; j < 4; j++) mean += normed.get(0, j);
    assert.ok(Math.abs(mean / 4) < 0.001);
  });

  test('rmsNorm: unit RMS per row', () => {
    const x = new Matrix(1, 4);
    x.set(0, 0, 2); x.set(0, 1, 4); x.set(0, 2, 6); x.set(0, 3, 8);
    const normed = rmsNorm(x);
    let sumSq = 0;
    for (let j = 0; j < 4; j++) sumSq += normed.get(0, j) ** 2;
    assert.ok(Math.abs(Math.sqrt(sumSq / 4) - 1) < 0.01);
  });

  test('compareNorms returns all types', () => {
    const x = Matrix.random(2, 4);
    const result = compareNorms(x);
    assert.ok(result.layerNorm);
    assert.ok(result.rmsNorm);
    assert.ok(result.instanceNorm);
  });
});
