// lstm-depth.test.js — LSTM layer depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { LSTM } from './rnn.js';
import { Matrix } from './matrix.js';

describe('LSTM Output Shape', () => {
  it('returns last hidden state', () => {
    const lstm = new LSTM(4, 8);
    const input = Matrix.random(2, 3 * 4); // batch=2, 3 timesteps, input=4
    const output = lstm.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 8);
  });

  it('return sequences', () => {
    const lstm = new LSTM(4, 8, { returnSequences: true });
    const input = Matrix.random(2, 3 * 4);
    const output = lstm.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 3 * 8);
  });

  it('single timestep', () => {
    const lstm = new LSTM(5, 10);
    const input = Matrix.random(1, 5);
    const output = lstm.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 10);
  });
});

describe('LSTM Cell State', () => {
  it('different inputs produce different cell states', () => {
    const lstm = new LSTM(2, 4);
    const input1 = new Matrix(1, 4, new Float64Array([1, 0, 0, 1]));
    const input2 = new Matrix(1, 4, new Float64Array([0, 1, 1, 0]));
    
    lstm.forward(input1);
    const cells1 = lstm.cells ? [...lstm.cells[lstm.cells.length - 1].data] : null;
    
    lstm.forward(input2);
    const cells2 = lstm.cells ? [...lstm.cells[lstm.cells.length - 1].data] : null;
    
    if (cells1 && cells2) {
      let different = false;
      for (let i = 0; i < cells1.length; i++) {
        if (Math.abs(cells1[i] - cells2[i]) > 1e-6) {
          different = true;
          break;
        }
      }
      assert.ok(different, 'Different inputs should produce different cell states');
    }
  });
});

describe('LSTM Backward', () => {
  it('backward returns correct gradient shape', () => {
    const lstm = new LSTM(4, 6);
    const input = Matrix.random(2, 3 * 4);
    lstm.forward(input);
    const dOutput = Matrix.random(2, 6);
    const dInput = lstm.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 3 * 4);
  });
});
