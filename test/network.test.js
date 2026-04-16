import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, sigmoid, relu, tanh, softmax, leakyRelu, linear, Network } from '../src/index.js';

describe('Activation functions', () => {
  it('sigmoid outputs (0, 1)', () => {
    const x = Matrix.fromArray([[-10, 0, 10]]);
    const y = sigmoid.forward(x);
    assert.ok(y.get(0, 0) < 0.01);
    assert.ok(Math.abs(y.get(0, 1) - 0.5) < 0.01);
    assert.ok(y.get(0, 2) > 0.99);
  });

  it('relu clips negatives', () => {
    const x = Matrix.fromArray([[-2, 0, 3]]);
    const y = relu.forward(x);
    assert.equal(y.get(0, 0), 0);
    assert.equal(y.get(0, 1), 0);
    assert.equal(y.get(0, 2), 3);
  });

  it('tanh outputs (-1, 1)', () => {
    const x = Matrix.fromArray([[-100, 0, 100]]);
    const y = tanh.forward(x);
    assert.ok(y.get(0, 0) < -0.99);
    assert.ok(Math.abs(y.get(0, 1)) < 0.01);
    assert.ok(y.get(0, 2) > 0.99);
  });

  it('softmax sums to 1', () => {
    const x = Matrix.fromArray([[1, 2, 3]]);
    const y = softmax.forward(x);
    const sum = y.get(0, 0) + y.get(0, 1) + y.get(0, 2);
    assert.ok(Math.abs(sum - 1) < 1e-6);
    assert.ok(y.get(0, 2) > y.get(0, 1)); // Largest input → largest output
  });

  it('softmax is numerically stable', () => {
    const x = Matrix.fromArray([[1000, 1001, 1002]]);
    const y = softmax.forward(x);
    const sum = y.get(0, 0) + y.get(0, 1) + y.get(0, 2);
    assert.ok(Math.abs(sum - 1) < 1e-6);
    assert.ok(!isNaN(y.get(0, 0)));
  });

  it('leaky relu passes negative values', () => {
    const x = Matrix.fromArray([[-10, 5]]);
    const y = leakyRelu.forward(x);
    assert.ok(Math.abs(y.get(0, 0) - (-0.1)) < 1e-6);
    assert.equal(y.get(0, 1), 5);
  });
});

describe('Network — XOR', () => {
  it('learns XOR', () => {
    const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = Matrix.fromArray([[0], [1], [1], [0]]);

    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const net = new Network();
      net.dense(2, 16, 'sigmoid');
      net.dense(16, 1, 'sigmoid');
      net.loss('mse');

      const history = net.train({ inputs, targets }, {
        epochs: 5000,
        learningRate: 1.0,
        batchSize: 4
      });

      if (history[history.length - 1] >= 0.1) continue;

      const pred = net.predict([[0, 0]]);
      const pred2 = net.predict([[1, 0]]);
      const pred3 = net.predict([[0, 1]]);
      const pred4 = net.predict([[1, 1]]);

      if (pred.get(0, 0) < 0.3 && pred2.get(0, 0) > 0.7 &&
          pred3.get(0, 0) > 0.7 && pred4.get(0, 0) < 0.3) {
        passed = true;
      }
    }
    assert.ok(passed, 'XOR should converge in 1 of 3 attempts');
  });

  it('network summary', () => {
    const net = new Network();
    net.dense(2, 8, 'relu');
    net.dense(8, 4, 'relu');
    net.dense(4, 1, 'sigmoid');
    const summary = net.summary();
    assert.ok(summary.includes('Total parameters'));
    assert.ok(summary.includes('relu'));
    assert.ok(summary.includes('sigmoid'));
  });
});
