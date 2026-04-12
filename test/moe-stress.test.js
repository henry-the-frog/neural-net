// moe-stress.test.js — Deep stress tests for Mixture of Experts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MixtureOfExperts } from '../src/moe.js';
import { Matrix } from '../src/matrix.js';

describe('Gating Probabilities', () => {
  it('gating weights sum to 1 per sample', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2); // 4in, 8hidden, 3out, top2 of numExperts
    const input = Matrix.random(5, 4);
    const output = moe.forward(input);
    
    // Gating probs should sum to 1 per sample
    const probs = moe.gate.probs;
    for (let i = 0; i < probs.rows; i++) {
      let sum = 0;
      for (let j = 0; j < probs.cols; j++) sum += probs.get(i, j);
      assert.ok(Math.abs(sum - 1) < 1e-5, `Sample ${i} gating sum: ${sum.toFixed(6)}`);
    }
  });

  it('gating weights are non-negative', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    moe.forward(input);
    
    const probs = moe.gate.probs;
    for (let i = 0; i < probs.data.length; i++) {
      assert.ok(probs.data[i] >= -1e-10, `Gating weight should be >= 0: ${probs.data[i]}`);
    }
  });
});

describe('Forward Pass', () => {
  it('output has correct shape', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    const output = moe.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 3);
  });

  it('output is finite', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    const output = moe.forward(input);
    assert.ok(output.data.every(Number.isFinite), 'Output should be finite');
  });

  it('different inputs produce different outputs', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const in1 = Matrix.random(1, 4);
    const in2 = Matrix.random(1, 4);
    const out1 = moe.forward(in1);
    const out2 = moe.forward(in2);
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01, 'Different inputs should produce different outputs');
  });

  it('handles single sample', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(1, 4);
    const output = moe.forward(input);
    assert.equal(output.rows, 1);
    assert.ok(output.data.every(Number.isFinite));
  });

  it('handles large batch', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(100, 4);
    const output = moe.forward(input);
    assert.equal(output.rows, 100);
    assert.ok(output.data.every(Number.isFinite));
  });
});

describe('Backward Pass', () => {
  it('backward produces finite gradients', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    moe.forward(input);
    const dOutput = Matrix.random(5, 3);
    const dInput = moe.backward(dOutput);
    assert.ok(dInput.data.every(Number.isFinite), 'Gradients should be finite');
    assert.equal(dInput.rows, 5);
    assert.equal(dInput.cols, 4);
  });

  it('update changes weights', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    
    const w0 = moe.gate.weights.data[0];
    moe.forward(input);
    moe.backward(Matrix.random(5, 3));
    moe.update(0.01);
    
    assert.notEqual(moe.gate.weights.data[0], w0, 'Weights should change after update');
  });
});

describe('Training', () => {
  it('loss decreases during training', () => {
    const moe = new MixtureOfExperts(2, 4, 8, 1, 2);
    const inputs = Matrix.random(20, 2);
    const targets = Matrix.random(20, 1);
    
    // First loss
    const out1 = moe.forward(inputs);
    let loss1 = 0;
    for (let i = 0; i < 20; i++) loss1 += (out1.get(i, 0) - targets.get(i, 0)) ** 2;
    
    // Train
    for (let epoch = 0; epoch < 50; epoch++) {
      const output = moe.forward(inputs);
      const dOutput = new Matrix(20, 1);
      for (let i = 0; i < 20; i++) {
        dOutput.set(i, 0, 2 * (output.get(i, 0) - targets.get(i, 0)) / 20);
      }
      moe.backward(dOutput);
      moe.update(0.01);
    }
    
    const out2 = moe.forward(inputs);
    let loss2 = 0;
    for (let i = 0; i < 20; i++) loss2 += (out2.get(i, 0) - targets.get(i, 0)) ** 2;
    
    assert.ok(loss2 < loss1, `Loss should decrease: ${loss1.toFixed(4)} → ${loss2.toFixed(4)}`);
  });
});
