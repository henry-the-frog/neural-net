// pool-flatten-depth.test.js — MaxPool2D + Flatten depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { MaxPool2D, Flatten } from './conv.js';
import { Matrix } from './matrix.js';

describe('MaxPool2D Output Shape', () => {
  it('2×2 pool on 4×4 input', () => {
    const pool = new MaxPool2D(4, 4, 1, 2); // H=4, W=4, C=1, poolSize=2
    assert.equal(pool.outputH, 2);
    assert.equal(pool.outputW, 2);
    assert.equal(pool.outputSize, 2 * 2 * 1);
  });

  it('2×2 pool on 6×6 input with 3 channels', () => {
    const pool = new MaxPool2D(6, 6, 3, 2);
    assert.equal(pool.outputH, 3);
    assert.equal(pool.outputW, 3);
    assert.equal(pool.outputSize, 3 * 3 * 3);
  });
});

describe('MaxPool2D Forward', () => {
  it('forward pass produces correct shape', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    const input = Matrix.random(1, 4 * 4 * 1);
    const output = pool.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, pool.outputSize);
  });

  it('batch forward', () => {
    const pool = new MaxPool2D(4, 4, 2, 2);
    const input = Matrix.random(8, 4 * 4 * 2);
    const output = pool.forward(input);
    assert.equal(output.rows, 8);
    assert.equal(output.cols, pool.outputSize);
  });

  it('max pool selects maximum values', () => {
    const pool = new MaxPool2D(2, 2, 1, 2);
    // 2×2 input, pool 2×2 → 1×1 output
    const input = new Matrix(1, 4, new Float64Array([1, 5, 3, 9]));
    const output = pool.forward(input);
    assert.equal(output.get(0, 0), 9); // Max of [1,5,3,9]
  });
});

describe('MaxPool2D Backward', () => {
  it('backward returns correct gradient shape', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    const input = Matrix.random(2, 16);
    pool.forward(input);
    const dOutput = Matrix.random(2, pool.outputSize);
    const dInput = pool.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 16);
  });
});

describe('Flatten Shape', () => {
  it('flatten preserves total elements', () => {
    const flat = new Flatten(8);
    const input = Matrix.random(4, 8);
    const output = flat.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 8);
  });

  it('backward returns same shape', () => {
    const flat = new Flatten(16);
    const input = Matrix.random(2, 16);
    flat.forward(input);
    const dOutput = Matrix.random(2, 16);
    const dInput = flat.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 16);
  });
});
