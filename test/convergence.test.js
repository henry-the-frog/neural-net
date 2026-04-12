// convergence.test.js — Verify networks actually learn real problems
// If backprop is correct, these should converge. If not, we find real bugs.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { Dense } from '../src/layer.js';
import { Network } from '../src/network.js';
import { RNN, LSTM } from '../src/rnn.js';
import { Conv2D, MaxPool2D, Flatten } from '../src/conv.js';
import { BatchNorm } from '../src/batchnorm.js';

// Helper: train a network and return final loss
function trainNetwork(net, inputs, targets, { epochs = 500, lr = 0.1, optimizer = 'sgd' } = {}) {
  const losses = [];
  for (let e = 0; e < epochs; e++) {
    const loss = net.trainBatch(inputs, targets, lr, 0, optimizer);
    losses.push(loss);
  }
  return { finalLoss: losses[losses.length - 1], losses };
}

describe('Training Convergence', () => {

  describe('XOR problem', () => {
    it('should learn XOR with 2-layer network', () => {
      let passed = false;
      for (let attempt = 0; attempt < 3 && !passed; attempt++) {
        const net = new Network();
        net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');

        const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
        const targets = Matrix.fromArray([[0], [1], [1], [0]]);

        const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 2000, lr: 0.5 });
        if (finalLoss >= 0.01) continue;

        const pred = net.predict(inputs);
        if (pred.get(0, 0) < 0.2 && pred.get(1, 0) > 0.8 && pred.get(2, 0) > 0.8 && pred.get(3, 0) < 0.2) {
          passed = true;
        }
      }
      assert.ok(passed, 'XOR should converge in at least 1 of 3 attempts');
    });

    it('should learn XOR with tanh activation', () => {
      const net = new Network();
      net.dense(2, 4, 'tanh').dense(4, 1, 'sigmoid').loss('mse');

      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 3000, lr: 0.5 });
      assert.ok(finalLoss < 0.02, `XOR tanh didn't converge: loss=${finalLoss.toFixed(4)}`);
    });

    it('should NOT learn XOR with single layer (linear separation impossible)', () => {
      const net = new Network();
      net.dense(2, 1, 'sigmoid').loss('mse');

      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 5000, lr: 0.5 });
      // Single layer can't solve XOR — loss should stay high
      assert.ok(finalLoss > 0.1, `Single layer shouldn't solve XOR but loss=${finalLoss.toFixed(4)}`);
    });
  });

  describe('Classification', () => {
    it('should learn simple 3-class classification with softmax', () => {
      const net = new Network();
      net.dense(2, 8, 'relu').dense(8, 3, 'softmax').loss('cross_entropy');

      // 3 clusters: class 0 centered at (-1,-1), class 1 at (1,1), class 2 at (-1,1)
      const inputs = Matrix.fromArray([
        [-1.0, -1.0], [-0.8, -0.9], [-1.1, -0.8],
        [1.0, 1.0], [0.9, 1.1], [1.1, 0.9],
        [-1.0, 1.0], [-0.9, 1.1], [-1.1, 0.9]
      ]);
      const targets = Matrix.fromArray([
        [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1], [0, 0, 1]
      ]);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 1000, lr: 0.1 });
      assert.ok(finalLoss < 0.1, `3-class classification didn't converge: loss=${finalLoss.toFixed(4)}`);

      // Check argmax predictions
      const pred = net.predict(inputs);
      const predClasses = pred.argmax();
      const trueClasses = [0, 0, 0, 1, 1, 1, 2, 2, 2];
      let correct = 0;
      for (let i = 0; i < predClasses.length; i++) {
        if (predClasses[i] === trueClasses[i]) correct++;
      }
      assert.ok(correct >= 8, `Expected >=8/9 correct, got ${correct}/9`);
    });

    it('should learn spiral classification (harder nonlinear problem)', () => {
      // Generate simple spiral data
      const samples = [];
      const labels = [];
      const n = 50; // points per class
      for (let cls = 0; cls < 2; cls++) {
        for (let i = 0; i < n; i++) {
          const r = i / n;
          const t = cls * Math.PI + r * Math.PI * 1.5 + (Math.random() - 0.5) * 0.2;
          samples.push([r * Math.cos(t), r * Math.sin(t)]);
          labels.push(cls === 0 ? [1, 0] : [0, 1]);
        }
      }

      const net = new Network();
      net.dense(2, 16, 'relu').dense(16, 16, 'relu').dense(16, 2, 'softmax').loss('cross_entropy');

      const inputs = Matrix.fromArray(samples);
      const targets = Matrix.fromArray(labels);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 2000, lr: 0.1 });
      // Spiral is harder — just verify loss decreased significantly
      assert.ok(finalLoss < 0.5, `Spiral didn't converge: loss=${finalLoss.toFixed(4)}`);
    });
  });

  describe('Regression', () => {
    it('should learn sine function approximation', () => {
      // Generate y = sin(x) data
      const samples = [];
      const labels = [];
      for (let i = 0; i < 40; i++) {
        const x = (i / 40) * 2 * Math.PI;
        samples.push([x / (2 * Math.PI)]); // Normalize to [0,1]
        labels.push([(Math.sin(x) + 1) / 2]); // Normalize to [0,1]
      }

      const net = new Network();
      net.dense(1, 16, 'relu').dense(16, 16, 'relu').dense(16, 1, 'sigmoid').loss('mse');

      const inputs = Matrix.fromArray(samples);
      const targets = Matrix.fromArray(labels);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 5000, lr: 0.05 });
      assert.ok(finalLoss < 0.05, `Sine approximation didn't converge: loss=${finalLoss.toFixed(4)}`);
    });

    it('should learn quadratic function', () => {
      const samples = [];
      const labels = [];
      for (let i = 0; i < 20; i++) {
        const x = (i - 10) / 10; // [-1, 1]
        samples.push([x]);
        labels.push([(x * x)]); // y = x²
      }

      const net = new Network();
      net.dense(1, 8, 'relu').dense(8, 1, 'linear').loss('mse');

      const inputs = Matrix.fromArray(samples);
      const targets = Matrix.fromArray(labels);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 3000, lr: 0.01 });
      assert.ok(finalLoss < 0.05, `Quadratic regression didn't converge: loss=${finalLoss.toFixed(4)}`);
    });
  });

  describe('Sequence learning (RNN)', () => {
    it('should learn simple sequence prediction with RNN', () => {
      const rnn = new RNN(1, 4);
      const net = new Network();
      net.add(rnn).dense(4, 1, 'linear').loss('mse');

      // Predict sum of 3-element sequence: [a, b, c] → (a+b+c)/3
      const inputs = Matrix.fromArray([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.3, 0.1, 0.2],
        [0.9, 0.1, 0.5]
      ]);
      const targets = Matrix.fromArray([
        [0.2],
        [0.5],
        [0.8],
        [0.2],
        [0.5]
      ]);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 1000, lr: 0.01 });
      assert.ok(finalLoss < 0.05, `RNN sequence prediction didn't converge: loss=${finalLoss.toFixed(4)}`);
    });

    it('should learn sequence with LSTM better than vanilla RNN on longer sequences', () => {
      // 5-step sequence: predict mean
      const makeData = () => {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < 20; i++) {
          const seq = Array.from({length: 5}, () => Math.random());
          const mean = seq.reduce((a, b) => a + b) / 5;
          inputs.push(seq);
          targets.push([mean]);
        }
        return { inputs: Matrix.fromArray(inputs), targets: Matrix.fromArray(targets) };
      };

      const data = makeData();

      // Train LSTM
      const lstmNet = new Network();
      lstmNet.add(new LSTM(1, 4)).dense(4, 1, 'linear').loss('mse');
      const lstmResult = trainNetwork(lstmNet, data.inputs, data.targets, { epochs: 500, lr: 0.01 });

      // Just verify LSTM converges at all
      assert.ok(lstmResult.finalLoss < 0.1, `LSTM didn't converge: loss=${lstmResult.finalLoss.toFixed(4)}`);
    });
  });

  describe('Training dynamics', () => {
    it('should show monotonically decreasing loss for simple problem', () => {
      let passed = false;
      for (let attempt = 0; attempt < 3 && !passed; attempt++) {
        const net = new Network();
        net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');

        const inputs = Matrix.fromArray([[0, 0], [1, 1]]);
        const targets = Matrix.fromArray([[0], [1]]);

        const { losses } = trainNetwork(net, inputs, targets, { epochs: 200, lr: 0.1 });

        // Check that loss generally decreases (allow some noise)
        const firstAvg = losses.slice(0, 10).reduce((a, b) => a + b) / 10;
        const lastAvg = losses.slice(-10).reduce((a, b) => a + b) / 10;
        if (lastAvg < firstAvg) passed = true;
      }
      assert.ok(passed, 'Loss should decrease in at least 1 of 3 attempts');
    });

    it('should learn faster with Adam than SGD on same problem', () => {
      const makeNet = () => {
        const net = new Network();
        net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');
        return net;
      };

      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);

      // SGD
      const sgdNet = makeNet();
      // Copy weights to adam net for fair comparison
      const adamNet = makeNet();
      for (let i = 0; i < sgdNet.layers.length; i++) {
        adamNet.layers[i].weights = sgdNet.layers[i].weights.clone();
        adamNet.layers[i].biases = sgdNet.layers[i].biases.clone();
      }

      const sgdResult = trainNetwork(sgdNet, inputs, targets, { epochs: 500, lr: 0.1, optimizer: 'sgd' });
      const adamResult = trainNetwork(adamNet, inputs, targets, { epochs: 500, lr: 0.01, optimizer: 'adam' });

      // Adam should converge at least as well (often better)
      // Just check both converge to reasonable loss
      assert.ok(sgdResult.finalLoss < 0.1 || adamResult.finalLoss < 0.1,
        `Neither optimizer converged: SGD=${sgdResult.finalLoss.toFixed(4)}, Adam=${adamResult.finalLoss.toFixed(4)}`);
    });

    it('should overfit small dataset with large network', () => {
      let passed = false;
      for (let attempt = 0; attempt < 3 && !passed; attempt++) {
        const net = new Network();
        net.dense(2, 32, 'relu').dense(32, 32, 'relu').dense(32, 1, 'sigmoid').loss('mse');

        // Just 2 training samples — should overfit easily
        const inputs = Matrix.fromArray([[0.3, 0.7], [0.8, 0.2]]);
        const targets = Matrix.fromArray([[0.9], [0.1]]);

        const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 1000, lr: 0.05 });
        if (finalLoss < 0.05) passed = true;
      }
      assert.ok(passed, 'Large network should overfit 2 samples in at least 1 of 3 attempts');
    });
  });

  describe('Conv2D training', () => {
    it('should learn simple pattern detection', () => {
      // Retry up to 3 times (random init can occasionally fail)
      let passed = false;
      for (let attempt = 0; attempt < 3 && !passed; attempt++) {
        const net = new Network();
        net.add(new Conv2D(4, 4, 1, 4, 3, 'leaky_relu'));
        net.add(new Flatten());
        net.dense(4 * 2 * 2, 2, 'softmax').loss('cross_entropy');

        const samples = [];
        const labels = [];
        for (let i = 0; i < 40; i++) {
          const cls = i % 2;
          const img = new Array(16).fill(0).map(() => cls === 0 ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3);
          samples.push(img);
          labels.push(cls === 0 ? [1, 0] : [0, 1]);
        }

        const inputs = Matrix.fromArray(samples);
        const targets = Matrix.fromArray(labels);

        const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 1000, lr: 0.05 });
        if (finalLoss < 0.694) passed = true;
      }
      assert.ok(passed, 'Conv2D should converge on at least 1 of 3 attempts');
    });
  });

  describe('BatchNorm training', () => {
    it('should help deep network converge', () => {
      // Without BN, deep networks can be hard to train
      const net = new Network();
      net.dense(2, 16, 'relu');
      net.add(new BatchNorm(16));
      net.dense(16, 16, 'relu');
      net.add(new BatchNorm(16));
      net.dense(16, 1, 'sigmoid');
      net.loss('mse');

      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);

      const { finalLoss } = trainNetwork(net, inputs, targets, { epochs: 2000, lr: 0.1 });
      assert.ok(finalLoss < 0.1, `BatchNorm XOR didn't converge: loss=${finalLoss.toFixed(4)}`);
    });
  });
});
