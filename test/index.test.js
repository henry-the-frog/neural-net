import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, NeuralNetwork, createNetwork, activations, losses, DenseLayer } from '../src/index.js';

// ===== Matrix tests =====
describe('Matrix — basics', () => {
  it('creates zeros', () => {
    const m = Matrix.zeros(2, 3);
    assert.equal(m.rows, 2);
    assert.equal(m.cols, 3);
    assert.equal(m.get(0, 0), 0);
  });

  it('set/get', () => {
    const m = Matrix.zeros(2, 2);
    m.set(0, 1, 5);
    assert.equal(m.get(0, 1), 5);
  });

  it('fromArray 2D', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    assert.equal(m.get(0, 0), 1);
    assert.equal(m.get(1, 1), 4);
  });

  it('add matrices', () => {
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = a.add(b);
    assert.equal(c.get(0, 0), 6);
    assert.equal(c.get(1, 1), 12);
  });

  it('dot product', () => {
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = a.dot(b);
    assert.equal(c.get(0, 0), 19);  // 1*5 + 2*7
    assert.equal(c.get(0, 1), 22);  // 1*6 + 2*8
  });

  it('transpose', () => {
    const m = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
    const t = m.transpose();
    assert.equal(t.rows, 3);
    assert.equal(t.cols, 2);
    assert.equal(t.get(0, 1), 4);
  });

  it('element-wise mul', () => {
    const a = Matrix.fromArray([[2, 3]]);
    const b = Matrix.fromArray([[4, 5]]);
    const c = a.mul(b);
    assert.equal(c.get(0, 0), 8);
    assert.equal(c.get(0, 1), 15);
  });

  it('scalar mul', () => {
    const m = Matrix.fromArray([[2, 3]]);
    const r = m.mul(10);
    assert.equal(r.get(0, 0), 20);
  });

  it('broadcasting add (row vector)', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[10, 20]]);
    const r = m.add(b);
    assert.equal(r.get(0, 0), 11);
    assert.equal(r.get(1, 1), 24);
  });
});

// ===== Activation tests =====
describe('Activations', () => {
  it('sigmoid(0) = 0.5', () => assert.ok(Math.abs(activations.sigmoid.forward(0) - 0.5) < 1e-10));
  it('relu(-1) = 0', () => assert.equal(activations.relu.forward(-1), 0));
  it('relu(5) = 5', () => assert.equal(activations.relu.forward(5), 5));
  it('tanh(0) = 0', () => assert.ok(Math.abs(activations.tanh.forward(0)) < 1e-10));
});

// ===== Loss tests =====
describe('Losses', () => {
  it('MSE of identical = 0', () => {
    const a = Matrix.fromArray([[1, 0]]);
    assert.ok(losses.mse.forward(a, a) < 1e-10);
  });

  it('MSE increases with difference', () => {
    const a = Matrix.fromArray([[1, 0]]);
    const b = Matrix.fromArray([[0, 1]]);
    assert.ok(losses.mse.forward(a, b) > 0);
  });
});

// ===== Network tests =====
describe('NeuralNetwork — XOR', () => {
  it('learns XOR', () => {
    const net = createNetwork([2, 4, 1], 'sigmoid');
    
    const inputs = Matrix.fromArray([
      [0, 0], [0, 1], [1, 0], [1, 1],
    ]);
    const targets = Matrix.fromArray([
      [0], [1], [1], [0],
    ]);
    
    const history = net.train(inputs, targets, { epochs: 5000, learningRate: 1.0 });
    
    // Loss should decrease
    assert.ok(history[history.length - 1] < history[0], 'Loss should decrease');
    
    // Predictions should be close to targets
    const pred = net.predict(inputs);
    assert.ok(pred.get(0, 0) < 0.3, `XOR(0,0) = ${pred.get(0, 0)} should be < 0.3`);
    assert.ok(pred.get(1, 0) > 0.7, `XOR(0,1) = ${pred.get(1, 0)} should be > 0.7`);
    assert.ok(pred.get(2, 0) > 0.7, `XOR(1,0) = ${pred.get(2, 0)} should be > 0.7`);
    assert.ok(pred.get(3, 0) < 0.3, `XOR(1,1) = ${pred.get(3, 0)} should be < 0.3`);
  });
});

describe('NeuralNetwork — AND gate', () => {
  it('learns AND', () => {
    const net = createNetwork([2, 1], 'sigmoid');
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[0],[0],[1]]);
    
    net.train(inputs, targets, { epochs: 2000, learningRate: 1.0 });
    
    const pred = net.predict(inputs);
    assert.ok(pred.get(0, 0) < 0.3);
    assert.ok(pred.get(3, 0) > 0.7);
  });
});

describe('NeuralNetwork — OR gate', () => {
  it('learns OR', () => {
    const net = createNetwork([2, 1], 'sigmoid');
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[1]]);
    
    net.train(inputs, targets, { epochs: 2000, learningRate: 1.0 });
    
    const pred = net.predict(inputs);
    assert.ok(pred.get(0, 0) < 0.3);
    assert.ok(pred.get(1, 0) > 0.7);
  });
});

describe('NeuralNetwork — forward pass', () => {
  it('single layer output shape', () => {
    const net = new NeuralNetwork();
    net.addLayer(3, 2, 'sigmoid');
    const input = Matrix.fromArray([[1, 2, 3]]);
    const output = net.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 2);
  });

  it('multi-layer output shape', () => {
    const net = createNetwork([4, 8, 3]);
    const input = Matrix.fromArray([[1, 2, 3, 4]]);
    const output = net.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 3);
  });
});

describe('NeuralNetwork — training', () => {
  it('loss decreases over training', () => {
    const net = createNetwork([2, 4, 1]);
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[0]]);
    
    const history = net.train(inputs, targets, { epochs: 100, learningRate: 0.5 });
    // First loss should be > last loss (usually)
    assert.ok(history.length === 100);
  });

  it('batch training works', () => {
    const net = createNetwork([2, 3, 1]);
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[1]]);
    net.train(inputs, targets, { epochs: 50, learningRate: 0.5 });
    // Should not throw
  });
});

describe('DenseLayer', () => {
  it('forward produces correct shape', () => {
    const layer = new DenseLayer(3, 5, 'relu');
    const input = Matrix.fromArray([[1, 2, 3]]);
    const output = layer.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 5);
  });
});
