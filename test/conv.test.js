import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, Conv2D, MaxPool2D, Flatten, Network, Dense } from '../src/index.js';

describe('Conv2D', () => {
  it('computes correct output shape', () => {
    const conv = new Conv2D(5, 5, 1, 4, 3); // 5×5 input, 1 channel, 4 filters, 3×3
    assert.equal(conv.outputH, 3);
    assert.equal(conv.outputW, 3);
    assert.equal(conv.outputSize, 3 * 3 * 4); // 36
  });

  it('forward produces correct shape', () => {
    const conv = new Conv2D(5, 5, 1, 2, 3);
    const input = Matrix.random(4, 25); // batch of 4, 5×5×1 flattened
    const output = conv.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 3 * 3 * 2); // 18
  });

  it('param count', () => {
    const conv = new Conv2D(5, 5, 1, 4, 3);
    assert.equal(conv.paramCount(), 4 * 9 * 1 + 4); // 4 filters × 3×3×1 + 4 biases = 40
  });

  it('backward produces correct gradient shape', () => {
    const conv = new Conv2D(5, 5, 1, 2, 3);
    const input = Matrix.random(2, 25);
    const output = conv.forward(input);
    const dOutput = Matrix.random(2, conv.outputSize);
    const dInput = conv.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 25);
  });

  it('backward produces non-zero gradients for filters', () => {
    const conv = new Conv2D(5, 5, 1, 2, 3);
    const input = Matrix.random(2, 25);
    conv.forward(input);
    const dOutput = Matrix.random(2, conv.outputSize);
    conv.backward(dOutput);
    
    let filterGradSum = 0;
    for (let i = 0; i < conv.dFilters.rows; i++)
      for (let j = 0; j < conv.dFilters.cols; j++)
        filterGradSum += Math.abs(conv.dFilters.get(i, j));
    assert.ok(filterGradSum > 0, 'Filter gradients should be non-zero');
  });

  it('backward produces non-zero input gradients (col2im)', () => {
    const conv = new Conv2D(5, 5, 1, 2, 3);
    const input = Matrix.random(2, 25);
    conv.forward(input);
    const dOutput = Matrix.random(2, conv.outputSize);
    const dInput = conv.backward(dOutput);
    
    let inputGradSum = 0;
    for (let i = 0; i < dInput.rows; i++)
      for (let j = 0; j < dInput.cols; j++)
        inputGradSum += Math.abs(dInput.get(i, j));
    assert.ok(inputGradSum > 0, 'Input gradients should be non-zero via col2im');
  });

  it('with padding', () => {
    const conv = new Conv2D(5, 5, 1, 2, 3, 'relu', { padding: 1 });
    assert.equal(conv.outputH, 5);
    assert.equal(conv.outputW, 5);
    const input = Matrix.random(1, 25);
    const output = conv.forward(input);
    assert.equal(output.cols, 5 * 5 * 2);
  });

  it('with stride 2', () => {
    const conv = new Conv2D(6, 6, 1, 3, 3, 'relu', { stride: 2 });
    assert.equal(conv.outputH, 2);
    assert.equal(conv.outputW, 2);
  });

  it('multi-channel input', () => {
    const conv = new Conv2D(5, 5, 3, 4, 3);
    const input = Matrix.random(2, 75);
    const output = conv.forward(input);
    assert.equal(output.cols, 3 * 3 * 4);
    assert.equal(conv.paramCount(), 4 * 9 * 3 + 4);
  });
});

describe('MaxPool2D', () => {
  it('downsamples by factor of 2', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    assert.equal(pool.outputH, 2);
    assert.equal(pool.outputW, 2);
    assert.equal(pool.outputSize, 4);
  });

  it('forward selects max values', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    const data = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16];
    const input = Matrix.fromArray([data]);
    const output = pool.forward(input);
    assert.equal(output.get(0, 0), 6);
    assert.equal(output.get(0, 1), 8);
    assert.equal(output.get(0, 2), 14);
    assert.equal(output.get(0, 3), 16);
  });

  it('backward routes gradients to max positions', () => {
    const pool = new MaxPool2D(4, 4, 1, 2);
    const data = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16];
    pool.forward(Matrix.fromArray([data]));
    const dOutput = Matrix.fromArray([[1, 1, 1, 1]]);
    const dInput = pool.backward(dOutput);
    assert.equal(dInput.get(0, 5), 1);
    assert.equal(dInput.get(0, 15), 1);
    assert.equal(dInput.get(0, 0), 0);
  });

  it('multi-channel pooling', () => {
    const pool = new MaxPool2D(4, 4, 2, 2);
    assert.equal(pool.outputSize, 2 * 2 * 2);
    const input = Matrix.random(1, 32);
    const output = pool.forward(input);
    assert.equal(output.cols, 8);
  });
});

describe('Flatten', () => {
  it('passes through', () => {
    const flat = new Flatten();
    const input = Matrix.random(3, 16);
    const output = flat.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 16);
  });
});

describe('Network with conv layers', () => {
  it('network.add() works with Conv2D + Dense', () => {
    const net = new Network();
    net.add(new Conv2D(5, 5, 1, 2, 3));
    net.add(new Flatten());
    net.add(new Dense(18, 10, 'softmax'));
    net.loss('crossEntropy');
    
    const input = Matrix.random(4, 25);
    const output = net.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 10);
  });

  it('ConvNet trains and reduces loss', () => {
    const net = new Network();
    net.add(new Conv2D(5, 5, 1, 2, 3));
    net.add(new Flatten());
    net.add(new Dense(18, 2, 'softmax'));
    net.loss('crossEntropy');
    
    const n = 20;
    const inputs = Matrix.random(n, 25);
    const targets = new Matrix(n, 2);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < 25; j++) sum += inputs.get(i, j);
      if (sum > 0) { targets.set(i, 0, 1); } else { targets.set(i, 1, 1); }
    }
    
    const history = net.train({ inputs, targets }, { epochs: 30, learningRate: 0.01, batchSize: 10 });
    assert.ok(history[history.length - 1] < history[0], 'Loss should decrease');
  });

  it('optimizer integration: Adam with Network', () => {
    const net = new Network();
    net.dense(2, 8, 'relu');
    net.dense(8, 1, 'sigmoid');
    net.loss('mse');
    net.optimizer('adam', { lr: 0.01 });
    
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[0]]);
    
    const history = net.train({ inputs, targets }, { epochs: 200, learningRate: 0.01, batchSize: 4 });
    assert.ok(history[history.length - 1] < history[0], 'Adam should reduce loss');
  });

  it('Network.toJSON and fromJSON roundtrip', () => {
    const net = new Network();
    net.dense(2, 4, 'sigmoid');
    net.dense(4, 1, 'sigmoid');
    net.loss('mse');
    
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[0]]);
    net.train({ inputs, targets }, { epochs: 50, learningRate: 1.0, batchSize: 4 });
    
    const json = JSON.stringify(net.toJSON());
    const loaded = Network.fromJSON(json);
    
    const testInput = Matrix.fromArray([[1, 0]]);
    const orig = net.predict(testInput).get(0, 0);
    const loadedPred = loaded.predict(testInput).get(0, 0);
    assert.ok(Math.abs(orig - loadedPred) < 1e-10);
  });

  it('Network.summary() works with mixed layers', () => {
    const net = new Network();
    net.add(new Conv2D(5, 5, 1, 4, 3));
    net.add(new MaxPool2D(3, 3, 4, 1));
    net.add(new Flatten());
    net.add(new Dense(12, 10, 'softmax'));
    
    const summary = net.summary();
    assert.ok(summary.includes('Conv2D'));
    assert.ok(summary.includes('Dense'));
    assert.ok(summary.includes('Total parameters'));
  });
});
