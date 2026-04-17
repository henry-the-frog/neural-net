// conv2d-depth.test.js — Conv2D layer depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Conv2D } from './conv.js';
import { Matrix } from './matrix.js';

describe('Conv2D Output Shape', () => {
  it('basic 3×3 filter on 5×5 input', () => {
    const layer = new Conv2D(5, 5, 1, 4, 3, 'relu');
    assert.equal(layer.outputH, 3);
    assert.equal(layer.outputW, 3);
    assert.equal(layer.outputSize, 3 * 3 * 4);
  });

  it('with padding=1', () => {
    const layer = new Conv2D(5, 5, 1, 4, 3, 'relu', { padding: 1 });
    // (5 + 2*1 - 3) / 1 + 1 = 5
    assert.equal(layer.outputH, 5);
    assert.equal(layer.outputW, 5);
  });

  it('with stride=2', () => {
    const layer = new Conv2D(6, 6, 1, 8, 3, 'relu', { stride: 2 });
    // (6 - 3) / 2 + 1 = 2.5 → floor → 2
    assert.equal(layer.outputH, 2);
    assert.equal(layer.outputW, 2);
  });

  it('multi-channel input', () => {
    const layer = new Conv2D(8, 8, 3, 16, 3, 'relu');
    assert.equal(layer.inputC, 3);
    assert.equal(layer.numFilters, 16);
    assert.equal(layer.filters.cols, 3 * 3 * 3); // filterSize^2 * channels
  });

  it('1×1 convolution', () => {
    const layer = new Conv2D(4, 4, 8, 16, 1, 'relu');
    assert.equal(layer.outputH, 4);
    assert.equal(layer.outputW, 4);
  });
});

describe('Conv2D Forward Pass', () => {
  it('forward produces correct output dimensions', () => {
    const layer = new Conv2D(4, 4, 1, 2, 3, 'relu');
    layer.training = false;
    const input = Matrix.random(1, 4 * 4 * 1);
    const output = layer.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, layer.outputSize);
  });

  it('batch forward pass', () => {
    const layer = new Conv2D(6, 6, 1, 4, 3, 'relu');
    layer.training = false;
    const input = Matrix.random(8, 6 * 6 * 1);
    const output = layer.forward(input);
    assert.equal(output.rows, 8);
    assert.equal(output.cols, layer.outputSize);
  });

  it('forward with padding', () => {
    const layer = new Conv2D(4, 4, 1, 2, 3, 'relu', { padding: 1 });
    layer.training = false;
    const input = Matrix.random(1, 4 * 4 * 1);
    const output = layer.forward(input);
    assert.equal(output.cols, 4 * 4 * 2); // Same spatial dims with padding=1
  });
});

describe('Conv2D Backward Pass', () => {
  it('backward returns correct gradient shape', () => {
    const layer = new Conv2D(4, 4, 1, 2, 3, 'relu');
    layer.training = false;
    const input = Matrix.random(1, 4 * 4 * 1);
    layer.forward(input);
    const dOutput = Matrix.random(1, layer.outputSize);
    const dInput = layer.backward(dOutput);
    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 4 * 4 * 1);
  });

  it('filter gradients have correct shape', () => {
    const layer = new Conv2D(4, 4, 1, 3, 3, 'relu');
    layer.training = false;
    const input = Matrix.random(2, 4 * 4 * 1);
    layer.forward(input);
    layer.backward(Matrix.random(2, layer.outputSize));
    assert.equal(layer.dFilters.rows, 3); // numFilters
    assert.equal(layer.dFilters.cols, 3 * 3 * 1); // filterSize^2 * channels
  });
});

describe('Conv2D Output Values', () => {
  it('all-zeros input produces bias-only output', () => {
    const layer = new Conv2D(3, 3, 1, 1, 3, 'relu');
    layer.training = false;
    // Set biases to known value
    layer.biases = new Matrix(1, 1, new Float64Array([5.0]));
    
    const input = Matrix.zeros(1, 3 * 3 * 1);
    const output = layer.forward(input);
    // With ReLU, output should be max(0, 5) = 5 for each spatial position
    assert.ok(output.data[0] >= 0, 'ReLU output should be non-negative');
  });
});
