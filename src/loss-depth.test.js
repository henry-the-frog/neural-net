// loss-depth.test.js — Loss function depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { mse, crossEntropy, getLoss } from './loss.js';
import { Matrix } from './matrix.js';

describe('MSE Loss', () => {
  it('zero loss when prediction equals target', () => {
    const pred = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    const target = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    const loss = mse.compute(pred, target);
    assert.ok(Math.abs(loss) < 1e-10, `Loss should be 0, got ${loss}`);
  });

  it('positive loss when prediction differs from target', () => {
    const pred = new Matrix(1, 2, new Float64Array([1, 1]));
    const target = new Matrix(1, 2, new Float64Array([0, 0]));
    const loss = mse.compute(pred, target);
    assert.ok(loss > 0, 'Loss should be positive');
  });

  it('MSE gradient has correct shape', () => {
    const pred = new Matrix(3, 4, new Float64Array(12).fill(0.5));
    const target = new Matrix(3, 4, new Float64Array(12).fill(0));
    const grad = mse.gradient(pred, target);
    assert.equal(grad.rows, 3);
    assert.equal(grad.cols, 4);
  });

  it('MSE gradient is zero when prediction equals target', () => {
    const pred = new Matrix(1, 3, new Float64Array([1, 2, 3]));
    const target = new Matrix(1, 3, new Float64Array([1, 2, 3]));
    const grad = mse.gradient(pred, target);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(grad.get(0, i)) < 1e-10);
    }
  });
});

describe('Cross-Entropy Loss', () => {
  it('zero loss for perfect prediction', () => {
    const pred = new Matrix(1, 3, new Float64Array([0, 0, 1]));
    const target = new Matrix(1, 3, new Float64Array([0, 0, 1]));
    const loss = crossEntropy.compute(pred, target);
    assert.ok(loss < 0.001, `Loss should be near 0, got ${loss}`);
  });

  it('high loss for wrong prediction', () => {
    const pred = new Matrix(1, 3, new Float64Array([0.9, 0.05, 0.05]));
    const target = new Matrix(1, 3, new Float64Array([0, 0, 1]));
    const loss = crossEntropy.compute(pred, target);
    assert.ok(loss > 1, `Loss should be high, got ${loss}`);
  });

  it('cross-entropy gradient has correct shape', () => {
    const pred = new Matrix(2, 3, new Float64Array([0.3, 0.3, 0.4, 0.1, 0.2, 0.7]));
    const target = new Matrix(2, 3, new Float64Array([0, 0, 1, 0, 0, 1]));
    const grad = crossEntropy.gradient(pred, target);
    assert.equal(grad.rows, 2);
    assert.equal(grad.cols, 3);
  });
});

describe('getLoss', () => {
  it('returns mse by default', () => {
    const loss = getLoss('mse');
    assert.equal(loss.name, 'mse');
  });

  it('returns cross_entropy', () => {
    const loss = getLoss('cross_entropy');
    assert.equal(loss.name, 'cross_entropy');
  });

  it('returns mse for unknown', () => {
    const loss = getLoss('unknown');
    assert.equal(loss.name, 'mse');
  });
});
