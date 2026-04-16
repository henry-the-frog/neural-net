// gradient-check.test.js — Numerical gradient verification
// The gold standard: compare analytical gradients (backprop) against finite differences

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { Dense } from '../src/layer.js';
import { Conv2D, MaxPool2D } from '../src/conv.js';
import { RNN, LSTM, GRU } from '../src/rnn.js';
import { Network } from '../src/network.js';
import { getLoss } from '../src/loss.js';
import { BatchNorm } from '../src/batchnorm.js';

// Numerical gradient via central differences: (f(x+h) - f(x-h)) / 2h
// More accurate than forward differences (O(h²) vs O(h))
function numericalGradient(fn, params, h = 1e-5) {
  const grads = new Float64Array(params.length);
  for (let i = 0; i < params.length; i++) {
    const orig = params[i];
    params[i] = orig + h;
    const fPlus = fn();
    params[i] = orig - h;
    const fMinus = fn();
    params[i] = orig;
    grads[i] = (fPlus - fMinus) / (2 * h);
  }
  return grads;
}

// Relative error between two gradient arrays
// Uses max(|a|, |b|, 1e-8) denominator to handle near-zero gradients
function relativeError(analytical, numerical) {
  let maxErr = 0;
  for (let i = 0; i < analytical.length; i++) {
    const denom = Math.max(Math.abs(analytical[i]), Math.abs(numerical[i]), 1e-8);
    const err = Math.abs(analytical[i] - numerical[i]) / denom;
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

describe('Numerical Gradient Checking', () => {
  
  describe('Dense layer gradients', () => {
    it('should match numerical gradients for weights (sigmoid)', () => {
      const layer = new Dense(3, 2, 'sigmoid');
      layer.training = true;
      const input = Matrix.fromArray([[0.5, -0.3, 0.8], [0.1, 0.7, -0.2]]);
      const target = Matrix.fromArray([[1, 0], [0, 1]]);
      const loss = getLoss('mse');

      // Forward + backward to get analytical gradients
      const output = layer.forward(input);
      const lossGrad = loss.gradient(output, target);
      layer.backward(lossGrad);

      // Numerical gradient for weights
      const numGrads = numericalGradient(() => {
        const out = layer.forward(input);
        return loss.compute(out, target);
      }, layer.weights.data);

      // Analytical gradient (averaged over batch like numerical)
      const batchSize = input.rows;
      const analyticalGrads = layer.dWeights.mul(1.0 / batchSize).data;

      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-5, `Weight gradient relative error too high: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for biases (sigmoid)', () => {
      const layer = new Dense(3, 2, 'sigmoid');
      const input = Matrix.fromArray([[0.5, -0.3, 0.8], [0.1, 0.7, -0.2]]);
      const target = Matrix.fromArray([[1, 0], [0, 1]]);
      const loss = getLoss('mse');

      const output = layer.forward(input);
      const lossGrad = loss.gradient(output, target);
      layer.backward(lossGrad);

      const numGrads = numericalGradient(() => {
        const out = layer.forward(input);
        return loss.compute(out, target);
      }, layer.biases.data);

      const batchSize = input.rows;
      const analyticalGrads = layer.dBiases.mul(1.0 / batchSize).data;

      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-5, `Bias gradient relative error too high: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for weights (relu)', () => {
      const layer = new Dense(4, 3, 'relu');
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1]]);
      const target = Matrix.fromArray([[1, 0.5, 0]]);
      const loss = getLoss('mse');

      const output = layer.forward(input);
      const lossGrad = loss.gradient(output, target);
      layer.backward(lossGrad);

      const numGrads = numericalGradient(() => {
        const out = layer.forward(input);
        return loss.compute(out, target);
      }, layer.weights.data);

      const analyticalGrads = layer.dWeights.data; // batchSize = 1
      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-5, `ReLU weight gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for weights (tanh)', () => {
      const layer = new Dense(3, 2, 'tanh');
      const input = Matrix.fromArray([[0.5, -0.3, 0.8], [-0.1, 0.4, 0.2]]);
      const target = Matrix.fromArray([[0.8, -0.5], [0.3, 0.7]]);
      const loss = getLoss('mse');

      const output = layer.forward(input);
      const lossGrad = loss.gradient(output, target);
      layer.backward(lossGrad);

      const numGrads = numericalGradient(() => {
        const out = layer.forward(input);
        return loss.compute(out, target);
      }, layer.weights.data);

      const batchSize = input.rows;
      const analyticalGrads = layer.dWeights.mul(1.0 / batchSize).data;
      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-5, `Tanh weight gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for input (backprop to previous layer)', () => {
      const layer = new Dense(3, 2, 'sigmoid');
      const input = Matrix.fromArray([[0.5, -0.3, 0.8]]);
      const target = Matrix.fromArray([[1, 0]]);
      const loss = getLoss('mse');

      const output = layer.forward(input);
      const lossGrad = loss.gradient(output, target);
      const dInput = layer.backward(lossGrad);

      const numGrads = numericalGradient(() => {
        const out = layer.forward(input);
        return loss.compute(out, target);
      }, input.data);

      const err = relativeError(dInput.data, numGrads);
      assert.ok(err < 1e-5, `Input gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('Multi-layer network gradients', () => {
    it('should propagate correct gradients through 2 Dense layers', () => {
      const net = new Network();
      net.dense(3, 4, 'sigmoid').dense(4, 2, 'sigmoid').loss('mse');

      const input = Matrix.fromArray([[0.5, -0.3, 0.8]]);
      const target = Matrix.fromArray([[1, 0]]);

      // Forward + backward
      const output = net.forward(input);
      const lossObj = getLoss('mse');
      let grad = lossObj.gradient(output, target);
      for (let i = net.layers.length - 1; i >= 0; i--) {
        grad = net.layers[i].backward(grad);
      }

      // Check layer 0 (first dense) weight gradients
      const layer0 = net.layers[0];
      const numGrads = numericalGradient(() => {
        const out = net.forward(input);
        return lossObj.compute(out, target);
      }, layer0.weights.data);

      const err = relativeError(layer0.dWeights.data, numGrads);
      assert.ok(err < 1e-3, `Layer 0 weight gradient error: ${err.toExponential(2)}`);
    });

    it('should propagate correct gradients through 3 Dense layers', () => {
      const net = new Network();
      net.dense(4, 5, 'tanh').dense(5, 3, 'relu').dense(3, 2, 'sigmoid').loss('mse');

      const input = Matrix.fromArray([[0.2, -0.5, 0.3, 0.8]]);
      const target = Matrix.fromArray([[0.7, 0.3]]);

      const output = net.forward(input);
      const lossObj = getLoss('mse');
      let grad = lossObj.gradient(output, target);
      for (let i = net.layers.length - 1; i >= 0; i--) {
        grad = net.layers[i].backward(grad);
      }

      // Check FIRST layer (deepest in chain — hardest to get right)
      const layer0 = net.layers[0];
      const numGrads = numericalGradient(() => {
        const out = net.forward(input);
        return lossObj.compute(out, target);
      }, layer0.weights.data);

      const err = relativeError(layer0.dWeights.data, numGrads);
      assert.ok(err < 1e-3, `Deep layer gradient error: ${err.toExponential(2)}`);
    });

    it('should match gradients with softmax + cross-entropy', () => {
      const net = new Network();
      net.dense(3, 4, 'relu').dense(4, 3, 'softmax').loss('cross_entropy');

      const input = Matrix.fromArray([[0.5, -0.3, 0.8]]);
      const target = Matrix.fromArray([[0, 1, 0]]); // One-hot

      const output = net.forward(input);
      const lossObj = getLoss('cross_entropy');
      let grad = lossObj.gradient(output, target);
      for (let i = net.layers.length - 1; i >= 0; i--) {
        grad = net.layers[i].backward(grad);
      }

      // Check first layer weights
      const layer0 = net.layers[0];
      const numGrads = numericalGradient(() => {
        const out = net.forward(input);
        return lossObj.compute(out, target);
      }, layer0.weights.data);

      const err = relativeError(layer0.dWeights.data, numGrads);
      assert.ok(err < 1e-3, `Softmax+CE gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('Conv2D gradients', () => {
    it('should match numerical gradients for filter weights', () => {
      const conv = new Conv2D(4, 4, 1, 2, 3, 'linear'); // 4x4 input, 1 channel, 2 filters, 3x3
      const inputSize = 4 * 4 * 1;
      const input = new Matrix(1, inputSize);
      for (let i = 0; i < inputSize; i++) input.data[i] = Math.random() * 2 - 1;

      const outputSize = conv.outputSize;
      const target = new Matrix(1, outputSize);
      for (let i = 0; i < outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');

      // Forward + backward
      const output = conv.forward(input);
      const lossGrad = loss.gradient(output, target);
      conv.backward(lossGrad);

      // Numerical gradient for filters
      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, conv.filters.data);

      // Conv2D backward already divides by batchSize, so compare directly
      const err = relativeError(conv.dFilters.data, numGrads);
      assert.ok(err < 1e-3, `Conv2D filter gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for bias', () => {
      const conv = new Conv2D(4, 4, 1, 2, 3, 'linear');
      const input = new Matrix(1, 16);
      for (let i = 0; i < 16; i++) input.data[i] = Math.random() * 2 - 1;

      const target = new Matrix(1, conv.outputSize);
      for (let i = 0; i < conv.outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');
      const output = conv.forward(input);
      conv.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, conv.biases.data);

      const err = relativeError(conv.dBiases.data, numGrads);
      assert.ok(err < 1e-3, `Conv2D bias gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for input (dInput)', () => {
      const conv = new Conv2D(4, 4, 1, 2, 3, 'linear');
      const input = new Matrix(1, 16);
      for (let i = 0; i < 16; i++) input.data[i] = Math.random() * 2 - 1;

      const target = new Matrix(1, conv.outputSize);
      for (let i = 0; i < conv.outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');
      const output = conv.forward(input);
      const dInput = conv.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, input.data);

      const err = relativeError(dInput.data, numGrads);
      assert.ok(err < 1e-3, `Conv2D input gradient error: ${err.toExponential(2)}`);
    });

    it('should match gradients with padding and stride', () => {
      const conv = new Conv2D(6, 6, 1, 1, 3, 'linear', { stride: 2, padding: 1 });
      const input = new Matrix(1, 36);
      for (let i = 0; i < 36; i++) input.data[i] = Math.random() * 2 - 1;

      const target = new Matrix(1, conv.outputSize);
      for (let i = 0; i < conv.outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');
      const output = conv.forward(input);
      conv.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, conv.filters.data);

      const err = relativeError(conv.dFilters.data, numGrads);
      assert.ok(err < 1e-3, `Conv2D stride+padding filter gradient error: ${err.toExponential(2)}`);
    });

    it('should match gradients with multi-channel input', () => {
      const conv = new Conv2D(4, 4, 3, 2, 3, 'linear'); // 3 channels
      const inputSize = 4 * 4 * 3;
      const input = new Matrix(1, inputSize);
      for (let i = 0; i < inputSize; i++) input.data[i] = Math.random() * 2 - 1;

      const target = new Matrix(1, conv.outputSize);
      for (let i = 0; i < conv.outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');
      const output = conv.forward(input);
      conv.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, conv.filters.data);

      const err = relativeError(conv.dFilters.data, numGrads);
      assert.ok(err < 1e-3, `Conv2D multi-channel filter gradient error: ${err.toExponential(2)}`);
    });

    it('should match gradients with relu activation', () => {
      const conv = new Conv2D(4, 4, 1, 2, 3, 'relu');
      const input = new Matrix(1, 16);
      for (let i = 0; i < 16; i++) input.data[i] = Math.random() * 2 - 1;

      const target = new Matrix(1, conv.outputSize);
      for (let i = 0; i < conv.outputSize; i++) target.data[i] = Math.random();

      const loss = getLoss('mse');
      const output = conv.forward(input);
      conv.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = conv.forward(input);
        return loss.compute(out, target);
      }, conv.filters.data);

      const err = relativeError(conv.dFilters.data, numGrads);
      // Higher tolerance for relu (non-differentiable at 0)
      assert.ok(err < 1e-3, `Conv2D relu filter gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('RNN gradients', () => {
    it('should match numerical gradients for Wih (input-to-hidden)', () => {
      const rnn = new RNN(2, 3);
      // Sequence of 3 timesteps, input_size=2: total = 6
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = rnn.forward(input);
      rnn.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = rnn.forward(input);
        return loss.compute(out, target);
      }, rnn.Wih.data);

      const err = relativeError(rnn.dWih.data, numGrads);
      assert.ok(err < 1e-3, `RNN Wih gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for Whh (hidden-to-hidden)', () => {
      const rnn = new RNN(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = rnn.forward(input);
      rnn.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = rnn.forward(input);
        return loss.compute(out, target);
      }, rnn.Whh.data);

      const err = relativeError(rnn.dWhh.data, numGrads);
      assert.ok(err < 1e-3, `RNN Whh gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for bias', () => {
      const rnn = new RNN(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = rnn.forward(input);
      rnn.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = rnn.forward(input);
        return loss.compute(out, target);
      }, rnn.bh.data);

      const err = relativeError(rnn.dbh.data, numGrads);
      assert.ok(err < 1e-3, `RNN bias gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('LSTM gradients', () => {
    // LSTM has 4 gates (input, forget, cell candidate, output)
    // Each gate has its own weight matrix Wi/Wf/Wc/Wo and bias bi/bf/bc/bo
    // Gradients stored as _dWi, _dWf, _dWc, _dWo, _dbi, _dbf, _dbc, _dbo

    it('should match numerical gradients for input gate weights (Wi)', () => {
      const lstm = new LSTM(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = lstm.forward(input);
      lstm.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = lstm.forward(input);
        return loss.compute(out, target);
      }, lstm.Wi.data);

      const err = relativeError(lstm._dWi.data, numGrads);
      assert.ok(err < 1e-3, `LSTM Wi gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for forget gate weights (Wf)', () => {
      const lstm = new LSTM(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = lstm.forward(input);
      lstm.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = lstm.forward(input);
        return loss.compute(out, target);
      }, lstm.Wf.data);

      const err = relativeError(lstm._dWf.data, numGrads);
      assert.ok(err < 1e-3, `LSTM Wf gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for output gate weights (Wo)', () => {
      const lstm = new LSTM(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = lstm.forward(input);
      lstm.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = lstm.forward(input);
        return loss.compute(out, target);
      }, lstm.Wo.data);

      const err = relativeError(lstm._dWo.data, numGrads);
      assert.ok(err < 1e-3, `LSTM Wo gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for input gate bias (bi)', () => {
      const lstm = new LSTM(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = lstm.forward(input);
      lstm.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = lstm.forward(input);
        return loss.compute(out, target);
      }, lstm.bi.data);

      const err = relativeError(lstm._dbi.data, numGrads);
      assert.ok(err < 1e-3, `LSTM bi gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for forget gate bias (bf)', () => {
      const lstm = new LSTM(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = lstm.forward(input);
      lstm.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = lstm.forward(input);
        return loss.compute(out, target);
      }, lstm.bf.data);

      const err = relativeError(lstm._dbf.data, numGrads);
      assert.ok(err < 1e-3, `LSTM bf gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('GRU gradients', () => {
    it('should match numerical gradients for Wz (update gate)', () => {
      const gru = new GRU(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = gru.forward(input);
      gru.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = gru.forward(input);
        return loss.compute(out, target);
      }, gru.Wz.data);

      const err = relativeError(gru._dWz.data, numGrads);
      assert.ok(err < 1e-3, `GRU Wz gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for Wr (reset gate)', () => {
      const gru = new GRU(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = gru.forward(input);
      gru.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = gru.forward(input);
        return loss.compute(out, target);
      }, gru.Wr.data);

      const err = relativeError(gru._dWr.data, numGrads);
      assert.ok(err < 1e-3, `GRU Wr gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for Wh (candidate)', () => {
      const gru = new GRU(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = gru.forward(input);
      gru.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = gru.forward(input);
        return loss.compute(out, target);
      }, gru.Wh.data);

      const err = relativeError(gru._dWh.data, numGrads);
      assert.ok(err < 1e-3, `GRU Wh gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for bz bias', () => {
      const gru = new GRU(2, 3);
      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, -0.2, 0.4]]);
      const target = Matrix.fromArray([[1, 0, 0.5]]);
      const loss = getLoss('mse');

      const output = gru.forward(input);
      gru.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = gru.forward(input);
        return loss.compute(out, target);
      }, gru.bz.data);

      const err = relativeError(gru._dbz.data, numGrads);
      assert.ok(err < 1e-3, `GRU bz gradient error: ${err.toExponential(2)}`);
    });
  });

  describe('Gradient flow sanity checks', () => {
    it('should have non-zero gradients at all layers in deep network', () => {
      const net = new Network();
      net.dense(4, 8, 'leaky_relu')
         .dense(8, 8, 'leaky_relu')
         .dense(8, 4, 'leaky_relu')
         .dense(4, 2, 'sigmoid')
         .loss('mse');

      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1]]);
      const target = Matrix.fromArray([[1, 0]]);

      const output = net.forward(input);
      let grad = getLoss('mse').gradient(output, target);
      for (let i = net.layers.length - 1; i >= 0; i--) {
        grad = net.layers[i].backward(grad);
      }

      for (let i = 0; i < net.layers.length; i++) {
        const maxGrad = Math.max(...Array.from(net.layers[i].dWeights.data).map(Math.abs));
        assert.ok(maxGrad > 1e-10, `Layer ${i} has zero gradients (dead layer)`);
      }
    });

    it('should detect vanishing gradients in deep sigmoid network', () => {
      const net = new Network();
      for (let i = 0; i < 10; i++) {
        net.dense(8, 8, 'sigmoid');
      }
      net.dense(8, 2, 'sigmoid').loss('mse');

      const input = Matrix.fromArray([[0.5, -0.3, 0.8, 0.1, 0.2, -0.1, 0.3, 0.6]]);
      const target = Matrix.fromArray([[1, 0]]);

      const output = net.forward(input);
      let grad = getLoss('mse').gradient(output, target);
      for (let i = net.layers.length - 1; i >= 0; i--) {
        grad = net.layers[i].backward(grad);
      }

      // First layer gradients should be MUCH smaller than last layer
      const firstGrad = Math.max(...Array.from(net.layers[0].dWeights.data).map(Math.abs));
      const lastGrad = Math.max(...Array.from(net.layers[net.layers.length - 1].dWeights.data).map(Math.abs));

      // Vanishing gradient ratio should be > 100x for 10 sigmoid layers
      const ratio = lastGrad / (firstGrad + 1e-20);
      assert.ok(ratio > 10, `Expected vanishing gradients (ratio: ${ratio.toFixed(1)}x)`);
    });

    it('should verify gradient magnitude is proportional to loss', () => {
      const layer = new Dense(3, 2, 'sigmoid');
      const input = Matrix.fromArray([[0.5, -0.3, 0.8]]);
      const loss = getLoss('mse');

      // Target close to output → small gradients
      const output1 = layer.forward(input);
      const closeTarget = output1.clone();
      closeTarget.data[0] += 0.01;
      const grad1 = layer.backward(loss.gradient(output1, closeTarget));
      const mag1 = Math.max(...Array.from(layer.dWeights.data).map(Math.abs));

      // Target far from output → large gradients
      const farTarget = output1.map(v => 1 - v);
      layer.forward(input);
      layer.backward(loss.gradient(layer.a, farTarget));
      const mag2 = Math.max(...Array.from(layer.dWeights.data).map(Math.abs));

      assert.ok(mag2 > mag1 * 1.5, `Far target should have larger gradients: ${mag2.toExponential(2)} vs ${mag1.toExponential(2)}`);
    });
  });

  describe('BatchNorm gradients', () => {
    it('should match numerical gradients for gamma', () => {
      const bn = new BatchNorm(3);
      // Need batch > 1 for meaningful stats
      const input = Matrix.fromArray([
        [1.0, -0.5, 0.3],
        [0.2, 0.8, -0.1],
        [-0.3, 0.1, 0.6],
        [0.7, -0.2, 0.4]
      ]);
      const target = Matrix.fromArray([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
      ]);
      const loss = getLoss('mse');

      bn.training = true;
      const output = bn.forward(input);
      bn.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = bn.forward(input);
        return loss.compute(out, target);
      }, bn.gamma.data);

      const batchSize = input.rows;
      const analyticalGrads = bn.dGamma.mul(1.0 / batchSize).data;
      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-3, `BatchNorm gamma gradient error: ${err.toExponential(2)}`);
    });

    it('should match numerical gradients for beta', () => {
      const bn = new BatchNorm(3);
      const input = Matrix.fromArray([
        [1.0, -0.5, 0.3],
        [0.2, 0.8, -0.1],
        [-0.3, 0.1, 0.6],
        [0.7, -0.2, 0.4]
      ]);
      const target = Matrix.fromArray([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
      ]);
      const loss = getLoss('mse');

      bn.training = true;
      const output = bn.forward(input);
      bn.backward(loss.gradient(output, target));

      const numGrads = numericalGradient(() => {
        const out = bn.forward(input);
        return loss.compute(out, target);
      }, bn.beta.data);

      const batchSize = input.rows;
      const analyticalGrads = bn.dBeta.mul(1.0 / batchSize).data;
      const err = relativeError(analyticalGrads, numGrads);
      assert.ok(err < 1e-3, `BatchNorm beta gradient error: ${err.toExponential(2)}`);
    });
  });
});
