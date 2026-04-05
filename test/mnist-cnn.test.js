import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { createMiniDigits, trainTestSplit, evaluate, confusionMatrix, predict } from '../src/mnist.js';
import { Network } from '../src/network.js';
import { Dense } from '../src/layer.js';
import { Conv2D, MaxPool2D, Flatten } from '../src/conv.js';
import { Matrix } from '../src/matrix.js';

describe('MNIST CNN Training', () => {
  it('should build a CNN for 8×8 digits', () => {
    // Conv2D: 8×8×1 → 6×6×4 (filter 3×3, no padding)
    const conv = new Conv2D(8, 8, 1, 4, 3, 'relu');
    assert.equal(conv.outputH, 6);
    assert.equal(conv.outputW, 6);
    assert.equal(conv.outputSize, 6 * 6 * 4);

    // MaxPool: 6×6×4 → 3×3×4
    const pool = new MaxPool2D(6, 6, 4, 2);
    assert.equal(pool.outputH, 3);
    assert.equal(pool.outputW, 3);
    assert.equal(pool.outputSize, 3 * 3 * 4);
  });

  it('should forward through CNN pipeline', () => {
    const net = new Network();
    net.add(new Conv2D(8, 8, 1, 4, 3, 'relu'));
    net.add(new MaxPool2D(6, 6, 4, 2));
    net.add(new Flatten(3, 3, 4));
    net.add(new Dense(36, 10, 'softmax'));

    // Single sample: 1×64 (8×8 flattened)
    const input = new Matrix(1, 64);
    for (let i = 0; i < 64; i++) input.data[i] = Math.random();
    const out = net.forward(input);
    assert.equal(out.cols, 10);
    // Softmax output should sum to ~1
    const sum = out.data.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 0.01, `Softmax sum should be ~1, got ${sum}`);
  });

  it('should train CNN on mini digits with decent accuracy', () => {
    const data = createMiniDigits({ samplesPerDigit: 30, noise: 0.02 });
    const { train, test } = trainTestSplit(data.images, data.labels, data.rawLabels, 0.2);

    const net = new Network();
    net.add(new Conv2D(8, 8, 1, 4, 3, 'relu'));
    net.add(new MaxPool2D(6, 6, 4, 2));
    net.add(new Flatten(3, 3, 4));
    net.add(new Dense(36, 10, 'softmax'));
    net.loss('crossEntropy');

    // Pack training data
    const n = train.images.length;
    const inputData = new Float64Array(n * 64);
    const targetData = new Float64Array(n * 10);
    for (let i = 0; i < n; i++) {
      inputData.set(train.images[i].data, i * 64);
      targetData.set(train.labels[i].data, i * 10);
    }
    const inputs = new Matrix(n, 64, inputData);
    const targets = new Matrix(n, 10, targetData);

    net.train({ inputs, targets }, {
      epochs: 50,
      learningRate: 0.005,
      batchSize: 16,
    });

    const acc = evaluate(net, test.images, test.rawLabels);
    assert.ok(acc > 0.3, `CNN accuracy should be > 30%, got ${(acc * 100).toFixed(1)}%`);
  });

  it('should produce confusion matrix', () => {
    const data = createMiniDigits({ samplesPerDigit: 10, noise: 0.02 });

    const net = new Network();
    net.add(new Dense(64, 32, 'relu'));
    net.add(new Dense(32, 10, 'softmax'));
    net.loss('crossEntropy');

    // Quick train
    const n = data.images.length;
    const inputData = new Float64Array(n * 64);
    const targetData = new Float64Array(n * 10);
    for (let i = 0; i < n; i++) {
      inputData.set(data.images[i].data, i * 64);
      targetData.set(data.labels[i].data, i * 10);
    }
    net.train({ inputs: new Matrix(n, 64, inputData), targets: new Matrix(n, 10, targetData) }, {
      epochs: 50, learningRate: 0.005, batchSize: 16,
    });

    const preds = data.images.map(img => predict(net, img));
    const cm = confusionMatrix(preds, data.rawLabels);
    assert.equal(cm.length, 10);
    assert.equal(cm[0].length, 10);
    // Total predictions should equal total samples
    const total = cm.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);
    assert.equal(total, n);
  });

  it('should handle serialization of trained model', () => {
    const net = new Network();
    net.add(new Dense(64, 16, 'relu'));
    net.add(new Dense(16, 10, 'softmax'));
    net.loss('crossEntropy');

    const data = createMiniDigits({ samplesPerDigit: 5, noise: 0 });
    const n = data.images.length;
    const inputData = new Float64Array(n * 64);
    const targetData = new Float64Array(n * 10);
    for (let i = 0; i < n; i++) {
      inputData.set(data.images[i].data, i * 64);
      targetData.set(data.labels[i].data, i * 10);
    }
    net.train({ inputs: new Matrix(n, 64, inputData), targets: new Matrix(n, 10, targetData) }, {
      epochs: 20, learningRate: 0.01, batchSize: 16,
    });

    // Serialize and deserialize
    const json = net.toJSON();
    const restored = Network.fromJSON(json);

    // Compare predictions
    for (let i = 0; i < 5; i++) {
      const pred1 = predict(net, data.images[i]);
      const pred2 = predict(restored, data.images[i]);
      assert.equal(pred1, pred2, `Prediction mismatch at sample ${i}`);
    }
  });
});
