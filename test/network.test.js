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
    const net = new Network();
    net.dense(2, 8, 'sigmoid');
    net.dense(8, 1, 'sigmoid');
    net.loss('mse');

    const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = Matrix.fromArray([[0], [1], [1], [0]]);

    // Train
    const history = net.train({ inputs, targets }, {
      epochs: 5000,
      learningRate: 1.0,
      batchSize: 4
    });

    // Final loss should be low
    assert.ok(history[history.length - 1] < 0.1, `Loss too high: ${history[history.length - 1]}`);

    // Test predictions
    const pred = net.predict([[0, 0]]);
    assert.ok(pred.get(0, 0) < 0.3, `XOR(0,0) should be ~0, got ${pred.get(0, 0)}`);

    const pred2 = net.predict([[1, 0]]);
    assert.ok(pred2.get(0, 0) > 0.7, `XOR(1,0) should be ~1, got ${pred2.get(0, 0)}`);

    const pred3 = net.predict([[0, 1]]);
    assert.ok(pred3.get(0, 0) > 0.7, `XOR(0,1) should be ~1, got ${pred3.get(0, 0)}`);

    const pred4 = net.predict([[1, 1]]);
    assert.ok(pred4.get(0, 0) < 0.3, `XOR(1,1) should be ~0, got ${pred4.get(0, 0)}`);
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
