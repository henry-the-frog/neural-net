// rnn-depth.test.js — RNN layer depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { RNN } from './rnn.js';
import { Matrix } from './matrix.js';

describe('RNN Output Shapes', () => {
  it('last hidden state output', () => {
    const rnn = new RNN(4, 8, { returnSequences: false });
    // Input: batch=2, sequence=3 steps of input_size=4 → flattened to 12
    const input = Matrix.random(2, 3 * 4);
    const output = rnn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 8);
  });

  it('return all sequences', () => {
    const rnn = new RNN(4, 8, { returnSequences: true });
    const input = Matrix.random(2, 3 * 4); // 3 timesteps
    const output = rnn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 3 * 8); // seqLen * hiddenSize
  });

  it('single timestep', () => {
    const rnn = new RNN(5, 10);
    const input = Matrix.random(1, 5); // 1 timestep
    const output = rnn.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 10);
  });

  it('batch size 16', () => {
    const rnn = new RNN(3, 6);
    const input = Matrix.random(16, 5 * 3); // 5 timesteps
    const output = rnn.forward(input);
    assert.equal(output.rows, 16);
    assert.equal(output.cols, 6);
  });
});

describe('RNN Hidden State Propagation', () => {
  it('different inputs produce different outputs', () => {
    const rnn = new RNN(2, 4);
    const input1 = new Matrix(1, 4, new Float64Array([1, 0, 0, 1]));
    const input2 = new Matrix(1, 4, new Float64Array([0, 1, 1, 0]));
    const out1 = rnn.forward(input1);
    const out2 = rnn.forward(input2);
    
    let different = false;
    for (let i = 0; i < out1.cols; i++) {
      if (Math.abs(out1.get(0, i) - out2.get(0, i)) > 1e-6) {
        different = true;
        break;
      }
    }
    assert.ok(different, 'Different inputs should produce different outputs');
  });

  it('hidden states build sequentially', () => {
    const rnn = new RNN(2, 3, { returnSequences: true });
    const input = Matrix.random(1, 3 * 2); // 3 timesteps
    rnn.forward(input);
    
    // Should have 4 hidden states (h0 + 3 timesteps)
    assert.equal(rnn.hiddens.length, 4);
    assert.equal(rnn.hiddens[0].rows, 1);
    assert.equal(rnn.hiddens[0].cols, 3);
  });
});

describe('RNN Backward Pass', () => {
  it('backward returns correct gradient shape', () => {
    const rnn = new RNN(4, 6);
    rnn.training = false;
    const input = Matrix.random(2, 3 * 4);
    rnn.forward(input);
    const dOutput = Matrix.random(2, 6);
    const dInput = rnn.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 3 * 4);
  });

  it('weight gradients have correct shapes', () => {
    const rnn = new RNN(3, 5);
    const input = Matrix.random(2, 4 * 3);
    rnn.forward(input);
    rnn.backward(Matrix.random(2, 5));
    
    assert.equal(rnn.dWih.rows, 3);
    assert.equal(rnn.dWih.cols, 5);
    assert.equal(rnn.dWhh.rows, 5);
    assert.equal(rnn.dWhh.cols, 5);
  });
});

describe('RNN Values', () => {
  it('zero input produces tanh of bias', () => {
    const rnn = new RNN(2, 3);
    const input = Matrix.zeros(1, 2); // 1 timestep, all zeros
    const output = rnn.forward(input);
    
    // With zero input and zero h0, output should be tanh(bias)
    // Biases are initialized to zero, so output should be tanh(0) = 0
    for (let i = 0; i < output.cols; i++) {
      assert.ok(Math.abs(output.get(0, i)) < 0.1, 
        `Zero input should give near-zero output, got ${output.get(0, i)}`);
    }
  });
});
