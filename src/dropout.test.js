// dropout.test.js — Dropout layer tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { Dropout } from './dropout.js';
import { Matrix } from './matrix.js';

describe('Dropout', () => {
  test('constructor sets defaults', () => {
    const d = new Dropout(0.3);
    assert.equal(d.rate, 0.3);
    assert.equal(d.training, true);
  });

  test('forward in training mode drops elements', () => {
    const d = new Dropout(0.5);
    const input = Matrix.ones(10, 20);
    const output = d.forward(input);
    assert.equal(output.rows, 10);
    assert.equal(output.cols, 20);
    
    // Some elements should be 0 (dropped)
    let zeros = 0;
    for (let i = 0; i < output.data.length; i++) {
      if (output.data[i] === 0) zeros++;
    }
    // With rate=0.5, roughly half should be dropped
    assert.ok(zeros > 30, `Expected significant dropout, got ${zeros} zeros out of 200`);
    assert.ok(zeros < 170, `Too many dropped: ${zeros} zeros out of 200`);
  });

  test('forward in eval mode is identity', () => {
    const d = new Dropout(0.5);
    d.training = false;
    const input = Matrix.ones(5, 10);
    const output = d.forward(input);
    
    // All elements should be exactly 1
    for (let i = 0; i < output.data.length; i++) {
      assert.equal(output.data[i], 1);
    }
  });

  test('inverted dropout: non-zero values are scaled by 1/(1-rate)', () => {
    const d = new Dropout(0.5);
    const input = Matrix.ones(10, 10);
    const output = d.forward(input);
    
    for (let i = 0; i < output.data.length; i++) {
      // Values are either 0 (dropped) or 2.0 (scaled: 1/(1-0.5) = 2)
      assert.ok(output.data[i] === 0 || Math.abs(output.data[i] - 2.0) < 0.001,
        `Expected 0 or 2.0, got ${output.data[i]}`);
    }
  });

  test('expected value preserved: mean ≈ input mean', () => {
    const d = new Dropout(0.3);
    const input = Matrix.ones(100, 100);
    const output = d.forward(input);
    
    let sum = 0;
    for (let i = 0; i < output.data.length; i++) sum += output.data[i];
    const mean = sum / output.data.length;
    // Mean should be close to 1.0 (input value) due to inverted dropout scaling
    assert.ok(Math.abs(mean - 1.0) < 0.15, `Mean ${mean} should be close to 1.0`);
  });

  test('backward propagates through mask', () => {
    const d = new Dropout(0.5);
    const input = Matrix.ones(5, 5);
    d.forward(input); // sets mask
    
    const dOutput = Matrix.ones(5, 5);
    const dInput = d.backward(dOutput);
    assert.equal(dInput.rows, 5);
    assert.equal(dInput.cols, 5);
    
    // dInput should match the mask pattern
    for (let i = 0; i < dInput.data.length; i++) {
      assert.ok(dInput.data[i] === 0 || dInput.data[i] > 0);
    }
  });

  test('rate=0 means no dropout', () => {
    const d = new Dropout(0);
    const input = Matrix.ones(5, 5);
    const output = d.forward(input);
    for (let i = 0; i < output.data.length; i++) {
      assert.equal(output.data[i], 1);
    }
  });

  test('paramCount is 0', () => {
    assert.equal(new Dropout().paramCount(), 0);
  });

  test('update is no-op', () => {
    const d = new Dropout();
    d.update(0.01); // Should not throw
  });
});
