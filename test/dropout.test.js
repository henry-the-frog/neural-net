import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { Dropout } from '../src/dropout.js';

function mat(rows, cols, values) {
  const m = new Matrix(rows, cols);
  for (let i = 0; i < values.length; i++) m.data[i] = values[i];
  return m;
}

describe('Dropout', () => {
  it('drops approximately rate fraction of neurons', () => {
    const dropout = new Dropout(0.5);
    dropout.training = true;
    
    // Large input to get statistical significance
    const input = new Matrix(1, 1000).fill(1);
    const output = dropout.forward(input);
    
    let zeros = 0;
    for (let i = 0; i < output.data.length; i++) {
      if (output.data[i] === 0) zeros++;
    }
    
    const dropRate = zeros / 1000;
    assert.ok(dropRate > 0.3 && dropRate < 0.7, 
      `Drop rate should be ~0.5, got ${dropRate}`);
  });

  it('scales kept neurons by 1/(1-rate)', () => {
    const dropout = new Dropout(0.5);
    dropout.training = true;
    
    const input = new Matrix(1, 1000).fill(1);
    const output = dropout.forward(input);
    
    // Non-zero values should be 2.0 (= 1/(1-0.5))
    for (let i = 0; i < output.data.length; i++) {
      if (output.data[i] !== 0) {
        assert.ok(Math.abs(output.data[i] - 2.0) < 0.001, 
          `Scaled value should be 2.0, got ${output.data[i]}`);
      }
    }
  });

  it('preserves expected value', () => {
    const dropout = new Dropout(0.3);
    dropout.training = true;
    
    const input = new Matrix(1, 10000).fill(5.0);
    const output = dropout.forward(input);
    
    let sum = 0;
    for (let i = 0; i < output.data.length; i++) sum += output.data[i];
    const mean = sum / 10000;
    
    // Expected value should be ~5.0 (inverted dropout preserves scale)
    assert.ok(Math.abs(mean - 5.0) < 0.5, 
      `Expected value should be ~5.0, got ${mean}`);
  });

  it('passes through in eval mode', () => {
    const dropout = new Dropout(0.5);
    dropout.training = false;
    
    const input = mat(1, 3, [1, 2, 3]);
    const output = dropout.forward(input);
    
    assert.equal(output.data[0], 1);
    assert.equal(output.data[1], 2);
    assert.equal(output.data[2], 3);
  });

  it('backward applies same mask', () => {
    const dropout = new Dropout(0.5);
    dropout.training = true;
    
    const input = new Matrix(1, 100).fill(1);
    dropout.forward(input);
    
    const dOutput = new Matrix(1, 100).fill(1);
    const dInput = dropout.backward(dOutput);
    
    // dInput should have same pattern as dropout output
    for (let i = 0; i < 100; i++) {
      if (dropout.mask.data[i] === 0) {
        assert.equal(dInput.data[i], 0, `Dropped neuron should have zero gradient`);
      } else {
        assert.ok(dInput.data[i] > 0, `Kept neuron should have nonzero gradient`);
      }
    }
  });

  it('rate=0 passes everything through', () => {
    const dropout = new Dropout(0);
    dropout.training = true;
    
    const input = mat(1, 3, [1, 2, 3]);
    const output = dropout.forward(input);
    
    assert.equal(output.data[0], 1);
    assert.equal(output.data[1], 2);
    assert.equal(output.data[2], 3);
  });

  it('has zero parameters', () => {
    const dropout = new Dropout(0.5);
    assert.equal(dropout.paramCount(), 0);
  });
});
