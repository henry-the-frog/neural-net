import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, RNN, LSTM, GRU, Network, Dense } from '../src/index.js';

describe('RNN', () => {
  it('forward produces correct shape (last state)', () => {
    const rnn = new RNN(3, 8); // inputSize=3, hiddenSize=8
    // 2 samples, sequence length 4, inputSize 3 → cols = 4*3 = 12
    const input = Matrix.random(2, 12);
    const output = rnn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 8);
  });

  it('forward produces correct shape (return sequences)', () => {
    const rnn = new RNN(3, 8, { returnSequences: true });
    const input = Matrix.random(2, 12); // seq_len=4
    const output = rnn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 32); // 4 * 8
  });

  it('backward produces correct gradient shape', () => {
    const rnn = new RNN(3, 8);
    const input = Matrix.random(2, 12);
    const output = rnn.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = rnn.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 12);
  });

  it('backward with returnSequences', () => {
    const rnn = new RNN(3, 8, { returnSequences: true });
    const input = Matrix.random(2, 12);
    rnn.forward(input);
    const dOutput = Matrix.random(2, 32);
    const dInput = rnn.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 12);
  });

  it('param count', () => {
    const rnn = new RNN(4, 10);
    // Wih: 4*10=40, Whh: 10*10=100, bh: 10 = 150
    assert.equal(rnn.paramCount(), 150);
  });

  it('hidden state changes over time', () => {
    const rnn = new RNN(2, 4);
    // 1 sample, seq_len=3, inputSize=2
    const input = Matrix.random(1, 6);
    rnn.forward(input);
    // Check that hidden states are different at each timestep
    const h1 = rnn.hiddens[1]; // After step 1
    const h2 = rnn.hiddens[2]; // After step 2
    let diff = 0;
    for (let j = 0; j < 4; j++) diff += Math.abs(h1.get(0, j) - h2.get(0, j));
    assert.ok(diff > 0.001, 'Hidden states should differ across timesteps');
  });

  it('trains on simple sequence pattern', () => {
    // Task: predict if sum of sequence > 0
    const net = new Network();
    net.add(new RNN(1, 8));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const n = 40;
    const seqLen = 5;
    const inputs = new Matrix(n, seqLen);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        const v = Math.random() * 2 - 1;
        inputs.set(i, j, v);
        sum += v;
      }
      targets.set(i, 0, sum > 0 ? 1 : 0);
    }

    const history = net.train({ inputs, targets }, { epochs: 50, learningRate: 0.01, batchSize: 20 });
    assert.ok(history[history.length - 1] < history[0], 'RNN should reduce loss on sequence task');
  });
});

describe('LSTM', () => {
  it('forward produces correct shape (last state)', () => {
    const lstm = new LSTM(3, 8);
    const input = Matrix.random(2, 12); // seq_len=4
    const output = lstm.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 8);
  });

  it('forward produces correct shape (return sequences)', () => {
    const lstm = new LSTM(3, 8, { returnSequences: true });
    const input = Matrix.random(2, 12);
    const output = lstm.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 32); // 4 * 8
  });

  it('backward produces correct gradient shape', () => {
    const lstm = new LSTM(3, 8);
    const input = Matrix.random(2, 12);
    lstm.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = lstm.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 12);
  });

  it('param count', () => {
    const lstm = new LSTM(4, 10);
    // 4 gates × ((4+10)*10 + 10) = 4 * 150 = 600
    assert.equal(lstm.paramCount(), 600);
  });

  it('forget gate bias initialized to 1', () => {
    const lstm = new LSTM(3, 5);
    for (let j = 0; j < 5; j++) {
      assert.equal(lstm.bf.get(0, j), 1.0, 'Forget bias should be 1.0');
    }
  });

  it('trains on sequence pattern', () => {
    const net = new Network();
    net.add(new LSTM(1, 8));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const n = 40;
    const seqLen = 5;
    const inputs = new Matrix(n, seqLen);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        const v = Math.random() * 2 - 1;
        inputs.set(i, j, v);
        sum += v;
      }
      targets.set(i, 0, sum > 0 ? 1 : 0);
    }

    const history = net.train({ inputs, targets }, { epochs: 50, learningRate: 0.01, batchSize: 20 });
    assert.ok(history[history.length - 1] < history[0], 'LSTM should reduce loss');
  });

  it('LSTM has more capacity than RNN', () => {
    // LSTM should have 4x more params for same input/hidden size
    const rnn = new RNN(5, 10);
    const lstm = new LSTM(5, 10);
    assert.ok(lstm.paramCount() > rnn.paramCount() * 3, 
      `LSTM (${lstm.paramCount()}) should have significantly more params than RNN (${rnn.paramCount()})`);
  });
});

describe('GRU', () => {
  it('forward produces correct shape (last state)', () => {
    const gru = new GRU(3, 8);
    const input = Matrix.random(2, 12);
    const output = gru.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 8);
  });

  it('forward produces correct shape (return sequences)', () => {
    const gru = new GRU(3, 8, { returnSequences: true });
    const input = Matrix.random(2, 12);
    const output = gru.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 32); // 4 * 8
  });

  it('backward produces correct gradient shape', () => {
    const gru = new GRU(3, 8);
    const input = Matrix.random(2, 12);
    gru.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = gru.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 12);
  });

  it('param count: 3 gates', () => {
    const gru = new GRU(4, 10);
    // 3 gates × ((4+10)*10 + 10) = 3 * 150 = 450
    assert.equal(gru.paramCount(), 450);
  });

  it('fewer params than LSTM', () => {
    const gru = new GRU(5, 10);
    const lstm = new LSTM(5, 10);
    assert.ok(gru.paramCount() < lstm.paramCount(),
      `GRU (${gru.paramCount()}) should have fewer params than LSTM (${lstm.paramCount()})`);
  });

  it('trains on sequence pattern', () => {
    const net = new Network();
    net.add(new GRU(1, 8));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const n = 40, seqLen = 5;
    const inputs = new Matrix(n, seqLen);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < seqLen; j++) {
        const v = Math.random() * 2 - 1;
        inputs.set(i, j, v);
        sum += v;
      }
      targets.set(i, 0, sum > 0 ? 1 : 0);
    }

    const history = net.train({ inputs, targets }, { epochs: 50, learningRate: 0.01, batchSize: 20 });
    assert.ok(history[history.length - 1] < history[0], 'GRU should reduce loss');
  });

  it('hidden state changes over time', () => {
    const gru = new GRU(2, 4);
    const input = Matrix.random(1, 6);
    gru.forward(input);
    const h1 = gru._cache.hiddens[1];
    const h2 = gru._cache.hiddens[2];
    let diff = 0;
    for (let j = 0; j < 4; j++) diff += Math.abs(h1.get(0, j) - h2.get(0, j));
    assert.ok(diff > 0.001, 'Hidden states should differ');
  });
});
