// dropout-depth.test.js — Dropout layer depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Dropout } from './dropout.js';
import { Matrix } from './matrix.js';

describe('Dropout Shape', () => {
  it('preserves shape during training', () => {
    const d = new Dropout(0.5);
    d.training = true;
    const input = Matrix.random(4, 8);
    const output = d.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 8);
  });

  it('preserves shape during eval', () => {
    const d = new Dropout(0.5);
    d.training = false;
    const input = Matrix.random(4, 8);
    const output = d.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 8);
  });
});

describe('Dropout Training Behavior', () => {
  it('zeroes some elements during training', () => {
    const d = new Dropout(0.5);
    d.training = true;
    const input = Matrix.ones(1, 100);
    const output = d.forward(input);
    
    let zeroCount = 0;
    for (let i = 0; i < output.cols; i++) {
      if (output.get(0, i) === 0) zeroCount++;
    }
    // With 50% dropout, expect roughly 40-60 zeros
    assert.ok(zeroCount > 20 && zeroCount < 80,
      `Expected ~50 zeros, got ${zeroCount}`);
  });

  it('eval mode passes through unchanged', () => {
    const d = new Dropout(0.5);
    d.training = false;
    const input = Matrix.ones(1, 10);
    const output = d.forward(input);
    
    for (let i = 0; i < 10; i++) {
      assert.equal(output.get(0, i), 1, 'Eval should pass through');
    }
  });

  it('dropout rate 0 passes everything through', () => {
    const d = new Dropout(0);
    d.training = true;
    const input = Matrix.ones(1, 100);
    const output = d.forward(input);
    
    for (let i = 0; i < 100; i++) {
      assert.ok(output.get(0, i) !== 0, 'Rate 0 should not drop anything');
    }
  });
});

describe('Dropout Backward', () => {
  it('backward returns correct shape', () => {
    const d = new Dropout(0.3);
    d.training = true;
    const input = Matrix.random(3, 5);
    d.forward(input);
    const dOutput = Matrix.random(3, 5);
    const dInput = d.backward(dOutput);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 5);
  });

  it('backward zeroes same positions as forward', () => {
    const d = new Dropout(0.5);
    d.training = true;
    const input = Matrix.ones(1, 50);
    const output = d.forward(input);
    const dOutput = Matrix.ones(1, 50);
    const dInput = d.backward(dOutput);
    
    // Positions zeroed in forward should be zeroed in backward
    for (let i = 0; i < 50; i++) {
      if (output.get(0, i) === 0) {
        assert.equal(dInput.get(0, i), 0, 'Dropped position should have zero gradient');
      }
    }
  });
});
