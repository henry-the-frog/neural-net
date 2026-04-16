// conv-stress.test.js — Stress tests for Conv2D backward correctness
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Conv2D, MaxPool2D } from '../src/conv.js';
import { Matrix } from '../src/matrix.js';

// Numerical gradient for a single weight element
function numGradWeight(layer, input, dOutput, weightMatrix, i, j, eps = 1e-5) {
  const orig = weightMatrix.get(i, j);
  
  weightMatrix.set(i, j, orig + eps);
  const outPlus = layer.forward(input);
  let lossPlus = 0;
  for (let r = 0; r < outPlus.rows; r++)
    for (let c = 0; c < outPlus.cols; c++)
      lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
  
  weightMatrix.set(i, j, orig - eps);
  const outMinus = layer.forward(input);
  let lossMinus = 0;
  for (let r = 0; r < outMinus.rows; r++)
    for (let c = 0; c < outMinus.cols; c++)
      lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
  
  weightMatrix.set(i, j, orig);
  return (lossPlus - lossMinus) / (2 * eps);
}

function numGradInput(layer, input, dOutput, b, idx, eps = 1e-5) {
  const orig = input.get(b, idx);
  
  input.set(b, idx, orig + eps);
  const outPlus = layer.forward(input);
  let lossPlus = 0;
  for (let r = 0; r < outPlus.rows; r++)
    for (let c = 0; c < outPlus.cols; c++)
      lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
  
  input.set(b, idx, orig - eps);
  const outMinus = layer.forward(input);
  let lossMinus = 0;
  for (let r = 0; r < outMinus.rows; r++)
    for (let c = 0; c < outMinus.cols; c++)
      lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
  
  input.set(b, idx, orig);
  return (lossPlus - lossMinus) / (2 * eps);
}

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-8);
}

describe('Conv2D Stress', () => {
  it('output shape is correct', () => {
    const conv = new Conv2D(4, 4, 1, 2, 3, 'relu');
    const input = Matrix.random(1, 16); // 4x4x1
    const output = conv.forward(input);
    assert.equal(output.rows, 1);
    // (4-3)/1 + 1 = 2, so 2x2x2 = 8
    assert.equal(output.cols, 8);
  });

  it('output shape with padding', () => {
    const conv = new Conv2D(4, 4, 1, 3, 3, 'relu', { padding: 1 });
    const input = Matrix.random(1, 16);
    const output = conv.forward(input);
    // (4+2-3)/1 + 1 = 4, so 4x4x3 = 48
    assert.equal(output.cols, 48);
  });

  it('filter gradient numerical check (identity activation)', () => {
    // Use identity activation to avoid non-linearity complications
    const conv = new Conv2D(3, 3, 1, 1, 2, 'linear');
    const input = Matrix.random(1, 9);
    conv.forward(input);
    const dOutput = Matrix.random(1, conv.outputSize);
    conv.backward(dOutput);
    
    // Check a few filter weights numerically
    let maxErr = 0;
    for (let i = 0; i < conv.filters.rows; i++) {
      for (let j = 0; j < Math.min(conv.filters.cols, 4); j++) {
        const ng = numGradWeight(conv, input, dOutput, conv.filters, i, j);
        const ag = conv.dFilters.get(i, j);
        const err = relErr(ag, ng);
        maxErr = Math.max(maxErr, err);
      }
    }
    assert.ok(maxErr < 0.01, `Filter gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('input gradient numerical check', () => {
    const conv = new Conv2D(3, 3, 1, 1, 2, 'linear');
    const input = Matrix.random(1, 9);
    conv.forward(input);
    const dOutput = Matrix.random(1, conv.outputSize);
    const dInput = conv.backward(dOutput);
    
    let maxErr = 0;
    for (let idx = 0; idx < 9; idx++) {
      const ng = numGradInput(conv, input, dOutput, 0, idx);
      const ag = dInput.get(0, idx);
      const err = relErr(ag, ng);
      maxErr = Math.max(maxErr, err);
    }
    assert.ok(maxErr < 0.01, `Input gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('multi-channel input gradient check', () => {
    const conv = new Conv2D(3, 3, 2, 1, 2, 'linear'); // 2 input channels
    const input = Matrix.random(1, 18); // 3x3x2
    conv.forward(input);
    const dOutput = Matrix.random(1, conv.outputSize);
    const dInput = conv.backward(dOutput);
    
    let maxErr = 0;
    for (let idx = 0; idx < 18; idx++) {
      const ng = numGradInput(conv, input, dOutput, 0, idx);
      const ag = dInput.get(0, idx);
      const err = relErr(ag, ng);
      maxErr = Math.max(maxErr, err);
    }
    assert.ok(maxErr < 0.01, `Multi-channel input gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('training loop decreases loss', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const conv = new Conv2D(4, 4, 1, 2, 3, 'relu');
      const input = Matrix.random(4, 16);
      const target = Matrix.random(4, conv.outputSize);
      
      let firstLoss = null;
      for (let step = 0; step < 50; step++) {
        const output = conv.forward(input);
        let loss = 0;
        const dOutput = new Matrix(4, conv.outputSize);
        for (let r = 0; r < 4; r++) {
          for (let c = 0; c < conv.outputSize; c++) {
            const diff = output.get(r, c) - target.get(r, c);
            loss += diff * diff;
            dOutput.set(r, c, 2 * diff / (4 * conv.outputSize));
          }
        }
        loss /= 4 * conv.outputSize;
        if (firstLoss === null) firstLoss = loss;
        conv.backward(dOutput);
        conv.update(0.01);
      }
      
      const finalOutput = conv.forward(input);
      let finalLoss = 0;
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < conv.outputSize; c++)
          finalLoss += (finalOutput.get(r, c) - target.get(r, c)) ** 2;
      finalLoss /= 4 * conv.outputSize;
      
      if (finalLoss < firstLoss) passed = true;
    }
    assert.ok(passed, 'Conv2D training should decrease loss');
  });
});

describe('MaxPool2D Stress', () => {
  it('output shape is correct', () => {
    const pool = new MaxPool2D(4, 4, 2, 2); // 4x4, 2 channels, pool=2
    const input = Matrix.random(1, 32); // 4x4x2
    const output = pool.forward(input);
    // 2x2x2 = 8
    assert.equal(output.cols, 8);
  });

  it('backward routes gradient to max indices only', () => {
    const pool = new MaxPool2D(2, 2, 1, 2);
    // Input: 2x2, 1 channel = [a, b, c, d]
    // Put max at index 2 (value 5.0)
    const input = new Matrix(1, 4, new Float64Array([1.0, 2.0, 5.0, 3.0]));
    const output = pool.forward(input);
    
    assert.equal(output.get(0, 0), 5.0); // Max of the pool
    
    const dOutput = new Matrix(1, 1, new Float64Array([1.0]));
    const dInput = pool.backward(dOutput);
    
    // Gradient should only go to index 2 (the max)
    assert.equal(dInput.get(0, 0), 0);
    assert.equal(dInput.get(0, 1), 0);
    assert.equal(dInput.get(0, 2), 1); // max element
    assert.equal(dInput.get(0, 3), 0);
  });

  it('handles tied max values', () => {
    const pool = new MaxPool2D(2, 2, 1, 2);
    const input = new Matrix(1, 4, new Float64Array([3.0, 3.0, 3.0, 3.0]));
    const output = pool.forward(input);
    assert.equal(output.get(0, 0), 3.0);
    
    const dOutput = new Matrix(1, 1, new Float64Array([1.0]));
    const dInput = pool.backward(dOutput);
    
    // Gradient goes to first max (implementation-dependent)
    let gradCount = 0;
    for (let i = 0; i < 4; i++) {
      if (dInput.get(0, i) > 0) gradCount++;
    }
    assert.equal(gradCount, 1, 'Should route gradient to exactly one max element');
  });

  it('batch processing', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    const input = Matrix.random(3, 16); // batch=3
    const output = pool.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 4); // 2x2x1
    
    const dOutput = Matrix.random(3, 4);
    const dInput = pool.backward(dOutput);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 16);
  });
});
