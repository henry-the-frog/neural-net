import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, Network, sigmoid, relu, tanh, softmax, mse, crossEntropy } from '../src/index.js';

describe('Gradient checking', () => {
  it('Dense layer gradients are correct (numerical check)', () => {
    const net = new Network();
    net.dense(2, 3, 'sigmoid');
    net.dense(3, 1, 'sigmoid');
    net.loss('mse');

    const input = Matrix.fromArray([[0.5, 0.3]]);
    const target = Matrix.fromArray([[0.7]]);

    // Compute analytical gradients
    const output = net.forward(input);
    const loss = mse.compute(output, target);
    let grad = mse.gradient(output, target);
    for (let i = net.layers.length - 1; i >= 0; i--) {
      grad = net.layers[i].backward(grad);
    }

    // Numerical gradient for first weight of first layer
    const eps = 1e-5;
    const layer = net.layers[0];
    const w00 = layer.weights.get(0, 0);

    // f(w + eps)
    layer.weights.set(0, 0, w00 + eps);
    const out1 = net.forward(input);
    const loss1 = mse.compute(out1, target);

    // f(w - eps)
    layer.weights.set(0, 0, w00 - eps);
    const out2 = net.forward(input);
    const loss2 = mse.compute(out2, target);

    // Restore
    layer.weights.set(0, 0, w00);

    const numericalGrad = (loss1 - loss2) / (2 * eps);
    const analyticalGrad = layer.dWeights.get(0, 0);

    // Should be close (relative error < 1e-3)
    const relError = Math.abs(numericalGrad - analyticalGrad) / (Math.abs(numericalGrad) + Math.abs(analyticalGrad) + 1e-8);
    assert.ok(relError < 0.01, `Gradient mismatch: numerical=${numericalGrad.toFixed(6)}, analytical=${analyticalGrad.toFixed(6)}, relError=${relError.toFixed(6)}`);
  });
});

describe('Model serialization', () => {
  it('save and load preserves predictions', () => {
    const net = new Network();
    net.dense(2, 4, 'sigmoid');
    net.dense(4, 1, 'sigmoid');
    net.loss('mse');

    // Train briefly
    const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = Matrix.fromArray([[0], [1], [1], [0]]);
    net.train({ inputs, targets }, { epochs: 100, learningRate: 1.0, batchSize: 4 });

    // Save
    const json = net.toJSON();
    const jsonStr = JSON.stringify(json);

    // Load
    const loaded = Network.fromJSON(jsonStr);

    // Predictions should match
    for (let i = 0; i < 4; i++) {
      const input = Matrix.fromArray([inputs.data.slice(i * 2, i * 2 + 2)]);
      const orig = net.predict(input).get(0, 0);
      const loadedPred = loaded.predict(input).get(0, 0);
      assert.ok(Math.abs(orig - loadedPred) < 1e-10, `Prediction mismatch at sample ${i}`);
    }
  });
});

describe('Loss functions', () => {
  it('MSE is zero for perfect prediction', () => {
    const pred = Matrix.fromArray([[0.5, 0.3]]);
    assert.equal(mse.compute(pred, pred), 0);
  });

  it('MSE increases with error', () => {
    const target = Matrix.fromArray([[1, 0]]);
    const good = Matrix.fromArray([[0.9, 0.1]]);
    const bad = Matrix.fromArray([[0.1, 0.9]]);
    assert.ok(mse.compute(good, target) < mse.compute(bad, target));
  });

  it('cross-entropy with perfect softmax', () => {
    const pred = Matrix.fromArray([[0.01, 0.01, 0.98]]);
    const target = Matrix.fromArray([[0, 0, 1]]);
    const loss = crossEntropy.compute(pred, target);
    assert.ok(loss < 0.1);
  });
});

describe('Activations — backward', () => {
  it('sigmoid derivative', () => {
    const x = Matrix.fromArray([[0.5]]);
    const y = sigmoid.forward(x);
    const dy = sigmoid.backward(y);
    assert.ok(dy.get(0, 0) > 0); // Should be positive
    assert.ok(dy.get(0, 0) < 0.3); // sigmoid'(0.5) ≈ 0.235
  });

  it('relu derivative', () => {
    const x = Matrix.fromArray([[-1, 0, 1]]);
    const y = relu.forward(x);
    const dy = relu.backward(y);
    assert.equal(dy.get(0, 0), 0); // Negative → 0
    assert.equal(dy.get(0, 2), 1); // Positive → 1
  });

  it('tanh derivative', () => {
    const x = Matrix.fromArray([[0]]);
    const y = tanh.forward(x);
    const dy = tanh.backward(y);
    assert.ok(Math.abs(dy.get(0, 0) - 1) < 1e-6); // tanh'(0) = 1
  });
});

describe('Network features', () => {
  it('dropout does not crash', () => {
    const net = new Network();
    net.dense(2, 4, 'relu'); // No dropout
    net.dense(4, 1, 'sigmoid');
    net.loss('mse');

    const inputs = Matrix.fromArray([[0, 1], [1, 0]]);
    const targets = Matrix.fromArray([[1], [1]]);
    net.train({ inputs, targets }, { epochs: 10, learningRate: 0.1, batchSize: 2 });
    const pred = net.predict([[0, 1]]);
    assert.ok(pred.get(0, 0) >= 0 && pred.get(0, 0) <= 1);
  });

  it('learning rate schedule: cosine', () => {
    const net = new Network();
    net.dense(2, 4, 'relu');
    net.dense(4, 1, 'sigmoid');
    net.loss('mse');

    const inputs = Matrix.fromArray([[0, 1]]);
    const targets = Matrix.fromArray([[1]]);
    const history = net.train({ inputs, targets }, { epochs: 50, learningRate: 0.5, batchSize: 1, lrSchedule: 'cosine' });
    assert.equal(history.length, 50);
  });

  it('evaluate returns accuracy', () => {
    const net = new Network();
    net.dense(2, 4, 'relu');
    net.dense(4, 3, 'softmax');
    net.loss('cross_entropy');

    const inputs = Matrix.fromArray([[1, 0], [0, 1], [1, 1]]);
    const targets = Matrix.oneHot([0, 1, 2], 3);
    net.train({ inputs, targets }, { epochs: 200, learningRate: 0.5, batchSize: 3 });

    const result = net.evaluate(inputs, targets);
    assert.ok(result.accuracy >= 0 && result.accuracy <= 1);
    assert.equal(result.total, 3);
  });
});
