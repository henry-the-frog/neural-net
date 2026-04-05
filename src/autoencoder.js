// autoencoder.js — Autoencoder and Variational Autoencoder

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';
import { getLoss } from './loss.js';

/**
 * Autoencoder — learns compressed representations
 * Encoder compresses input → latent space, decoder reconstructs
 */
export class Autoencoder {
  constructor(inputSize, latentSize, hiddenSizes = []) {
    this.inputSize = inputSize;
    this.latentSize = latentSize;
    
    // Build encoder layers
    this.encoderLayers = [];
    let prevSize = inputSize;
    for (const h of hiddenSizes) {
      this.encoderLayers.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.encoderLayers.push(new Dense(prevSize, latentSize, 'linear'));
    
    // Build decoder layers (mirror of encoder)
    this.decoderLayers = [];
    prevSize = latentSize;
    for (const h of [...hiddenSizes].reverse()) {
      this.decoderLayers.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.decoderLayers.push(new Dense(prevSize, inputSize, 'sigmoid'));
    
    this.allLayers = [...this.encoderLayers, ...this.decoderLayers];
    this.lossFunction = getLoss('mse');
    this.training = true;
  }

  encode(input) {
    let x = input;
    for (const layer of this.encoderLayers) {
      x = layer.forward(x);
    }
    return x;
  }

  decode(latent) {
    let x = latent;
    for (const layer of this.decoderLayers) {
      x = layer.forward(x);
    }
    return x;
  }

  forward(input) {
    const latent = this.encode(input);
    return this.decode(latent);
  }

  trainBatch(input, learningRate = 0.001) {
    // Forward
    const output = this.forward(input);
    const loss = this.lossFunction.compute(output, input); // Reconstruct input

    // Backward
    let grad = this.lossFunction.gradient(output, input);
    for (let i = this.decoderLayers.length - 1; i >= 0; i--) {
      grad = this.decoderLayers[i].backward(grad);
    }
    for (let i = this.encoderLayers.length - 1; i >= 0; i--) {
      grad = this.encoderLayers[i].backward(grad);
    }

    // Update
    for (const layer of this.allLayers) {
      layer.update(learningRate, 0, 'sgd');
    }

    return loss;
  }

  train(data, { epochs = 100, learningRate = 0.001, batchSize = 32, verbose = false } = {}) {
    const n = data.rows;
    const history = [];

    for (const l of this.allLayers) l.training = true;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let batches = 0;

      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const batch = new Matrix(end - start, data.cols);
        for (let i = start; i < end; i++) {
          const idx = indices[i];
          for (let j = 0; j < data.cols; j++) batch.set(i - start, j, data.get(idx, j));
        }
        epochLoss += this.trainBatch(batch, learningRate);
        batches++;
      }

      epochLoss /= batches;
      history.push(epochLoss);

      if (verbose && (epoch % Math.max(1, Math.floor(epochs / 10)) === 0)) {
        console.log(`Epoch ${epoch + 1}/${epochs} — Loss: ${epochLoss.toFixed(6)}`);
      }
    }

    for (const l of this.allLayers) l.training = false;
    return history;
  }

  paramCount() {
    return this.allLayers.reduce((sum, l) => sum + l.paramCount(), 0);
  }
}

/**
 * Variational Autoencoder (VAE) — learns a generative model
 * Encoder outputs mean and log-variance, decoder samples from latent space
 */
export class VAE {
  constructor(inputSize, latentSize, hiddenSizes = []) {
    this.inputSize = inputSize;
    this.latentSize = latentSize;

    // Encoder → mean and log-variance
    this.encoderLayers = [];
    let prevSize = inputSize;
    for (const h of hiddenSizes) {
      this.encoderLayers.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.meanLayer = new Dense(prevSize, latentSize, 'linear');
    this.logvarLayer = new Dense(prevSize, latentSize, 'linear');

    // Decoder
    this.decoderLayers = [];
    prevSize = latentSize;
    for (const h of [...hiddenSizes].reverse()) {
      this.decoderLayers.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.decoderLayers.push(new Dense(prevSize, inputSize, 'sigmoid'));

    this.allLayers = [...this.encoderLayers, this.meanLayer, this.logvarLayer, ...this.decoderLayers];
    this.lossFunction = getLoss('mse');
    this.training = true;

    // Cache
    this._mean = null;
    this._logvar = null;
    this._z = null;
    this._epsilon = null;
  }

  encode(input) {
    let x = input;
    for (const layer of this.encoderLayers) {
      x = layer.forward(x);
    }
    this._mean = this.meanLayer.forward(x);
    this._logvar = this.logvarLayer.forward(x);
    return { mean: this._mean, logvar: this._logvar };
  }

  reparameterize(mean, logvar) {
    // z = mean + std * epsilon, where epsilon ~ N(0, 1)
    this._epsilon = Matrix.random(mean.rows, mean.cols);
    const std = logvar.map(v => Math.exp(0.5 * v));
    this._z = mean.add(std.mul(this._epsilon));
    return this._z;
  }

  decode(z) {
    let x = z;
    for (const layer of this.decoderLayers) {
      x = layer.forward(x);
    }
    return x;
  }

  forward(input) {
    const { mean, logvar } = this.encode(input);
    const z = this.reparameterize(mean, logvar);
    return this.decode(z);
  }

  // KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  klDivergence() {
    let kl = 0;
    for (let i = 0; i < this._mean.rows; i++) {
      for (let j = 0; j < this._mean.cols; j++) {
        const mu = this._mean.get(i, j);
        const lv = this._logvar.get(i, j);
        kl += -0.5 * (1 + lv - mu * mu - Math.exp(lv));
      }
    }
    return kl / this._mean.rows;
  }

  trainBatch(input, learningRate = 0.001, klWeight = 1.0) {
    const output = this.forward(input);
    const reconLoss = this.lossFunction.compute(output, input);
    const klLoss = this.klDivergence();
    const totalLoss = reconLoss + klWeight * klLoss;

    // Backward through decoder
    let grad = this.lossFunction.gradient(output, input);
    for (let i = this.decoderLayers.length - 1; i >= 0; i--) {
      grad = this.decoderLayers[i].backward(grad);
    }

    // Backward through reparameterization → mean and logvar
    // dz/dmean = 1, dz/dlogvar = 0.5 * epsilon * exp(0.5 * logvar)
    const dMean = grad.add(this._mean.mul(klWeight / input.rows));
    const std = this._logvar.map(v => Math.exp(0.5 * v));
    const dLogvar = grad.mul(this._epsilon).mul(std).mul(0.5)
      .add(this._logvar.map(v => 0.5 * klWeight * (Math.exp(v) - 1) / input.rows));

    this.meanLayer.backward(dMean);
    this.logvarLayer.backward(dLogvar);

    // Backward through encoder (combine gradients from mean and logvar)
    const gradFromMean = dMean.dot(this.meanLayer.weights.T());
    const gradFromLogvar = dLogvar.dot(this.logvarLayer.weights.T());
    let encGrad = gradFromMean.add(gradFromLogvar);
    for (let i = this.encoderLayers.length - 1; i >= 0; i--) {
      encGrad = this.encoderLayers[i].backward(encGrad);
    }

    // Update all layers
    for (const layer of this.allLayers) {
      layer.update(learningRate, 0, 'sgd');
    }

    return { total: totalLoss, reconstruction: reconLoss, kl: klLoss };
  }

  train(data, { epochs = 100, learningRate = 0.001, batchSize = 32, klWeight = 1.0, verbose = false } = {}) {
    const n = data.rows;
    const history = [];

    for (const l of this.allLayers) l.training = true;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let batches = 0;

      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const batch = new Matrix(end - start, data.cols);
        for (let i = start; i < end; i++) {
          const idx = indices[i];
          for (let j = 0; j < data.cols; j++) batch.set(i - start, j, data.get(idx, j));
        }
        const { total } = this.trainBatch(batch, learningRate, klWeight);
        epochLoss += total;
        batches++;
      }

      epochLoss /= batches;
      history.push(epochLoss);

      if (verbose && (epoch % Math.max(1, Math.floor(epochs / 10)) === 0)) {
        console.log(`Epoch ${epoch + 1}/${epochs} — Loss: ${epochLoss.toFixed(6)}`);
      }
    }

    for (const l of this.allLayers) l.training = false;
    return history;
  }

  // Generate new samples by decoding random latent vectors
  generate(numSamples = 1) {
    const z = Matrix.random(numSamples, this.latentSize);
    return this.decode(z);
  }

  paramCount() {
    return this.allLayers.reduce((sum, l) => sum + l.paramCount(), 0);
  }
}
