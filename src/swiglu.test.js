// swiglu.test.js — SwiGLU activation tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { SwiGLU, swish, swishDerivative } from './swiglu.js';
import { Matrix } from './matrix.js';

describe('SwiGLU', () => {
  test('swish(0) = 0', () => {
    assert.ok(Math.abs(swish(0)) < 0.001);
  });

  test('swish is approximately identity for large positive x', () => {
    assert.ok(Math.abs(swish(10) - 10) < 0.1);
  });

  test('swish derivative at 0 = 0.5', () => {
    // swish'(0) = sigmoid(0) + 0 = 0.5
    assert.ok(Math.abs(swishDerivative(0) - 0.5) < 0.001);
  });

  test('forward produces correct shape', () => {
    const sg = new SwiGLU(8, 16);
    const x = Matrix.random(3, 8);
    const out = sg.forward(x);
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 8); // Same as input dim
  });

  test('backward produces correct shape', () => {
    const sg = new SwiGLU(4, 8);
    const x = Matrix.random(2, 4);
    sg.forward(x);
    const dOut = Matrix.random(2, 4);
    const dInput = sg.backward(dOut);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 4);
  });

  test('gradient shapes are correct', () => {
    const sg = new SwiGLU(4, 8);
    const x = Matrix.random(3, 4);
    sg.forward(x);
    sg.backward(Matrix.ones(3, 4));
    
    assert.equal(sg.dW1.rows, 4);
    assert.equal(sg.dW1.cols, 8);
    assert.equal(sg.dWgate.rows, 4);
    assert.equal(sg.dWgate.cols, 8);
    assert.equal(sg.dW2.rows, 8);
    assert.equal(sg.dW2.cols, 4);
  });

  test('update changes weights', () => {
    const sg = new SwiGLU(4, 8);
    const x = Matrix.random(3, 4);
    sg.forward(x);
    sg.backward(Matrix.ones(3, 4));
    
    const origW = sg.W1.data[0];
    sg.update(0.01);
    assert.notEqual(sg.W1.data[0], origW);
  });

  test('paramCount is correct', () => {
    const sg = new SwiGLU(8, 16);
    // W1: 8*16=128, b1: 16, Wgate: 8*16=128, bgate: 16, W2: 16*8=128, b2: 8
    // Total: 128+16+128+16+128+8 = 424
    assert.equal(sg.paramCount(), 424);
  });
});
