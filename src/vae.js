/**
 * vae.js — Variational Autoencoder
 * 
 * A generative model that learns a latent representation through:
 * 1. Encoder: maps input → latent distribution (μ, σ²)
 * 2. Reparameterization: z = μ + σ · ε, ε ~ N(0,1)
 * 3. Decoder: maps z → reconstruction
 * 
 * Loss = reconstruction_loss + KL_divergence(q(z|x) || p(z))
 * 
 * Where KL = -0.5 · Σ(1 + log(σ²) - μ² - σ²)
 * 
 * Based on: Kingma & Welling (2013), "Auto-Encoding Variational Bayes"
 */

import { Matrix } from './matrix.js';

function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
function sigmoidDeriv(y) { return y * (1 - y); }

/**
 * Simple dense layer for internal use by VAE
 */
class VAEDense {
  constructor(inputSize, outputSize, activation = 'sigmoid') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;

    // Xavier initialization
    const scale = Math.sqrt(2 / (inputSize + outputSize));
    this.W = new Matrix(outputSize, inputSize);
    for (let i = 0; i < this.W.data.length; i++) {
      this.W.data[i] = (Math.random() * 2 - 1) * scale;
    }
    this.b = new Matrix(outputSize, 1);

    // Cache for backward pass
    this._input = null;
    this._preActivation = null;
    this._output = null;

    // Gradient accumulators
    this.dW = Matrix.zeros(outputSize, inputSize);
    this.db = Matrix.zeros(outputSize, 1);
  }

  forward(input) {
    this._input = input;
    this._preActivation = this.W.dot(input).add(this.b);
    if (this.activation === 'sigmoid') {
      this._output = this._preActivation.map(x => sigmoid(x));
    } else if (this.activation === 'linear') {
      this._output = new Matrix(this._preActivation.rows, this._preActivation.cols,
        new Float64Array(this._preActivation.data));
    } else if (this.activation === 'relu') {
      this._output = this._preActivation.map(x => Math.max(0, x));
    }
    return this._output;
  }

  backward(dOutput) {
    let dPreAct;
    if (this.activation === 'sigmoid') {
      dPreAct = dOutput.map((d, i) => d * sigmoidDeriv(this._output.data[i]));
    } else if (this.activation === 'linear') {
      dPreAct = dOutput;
    } else if (this.activation === 'relu') {
      dPreAct = dOutput.map((d, i) => this._preActivation.data[i] > 0 ? d : 0);
    }

    // Gradient of weights: dW = dPreAct · input^T
    this.dW = dPreAct.dot(this._input.transpose());
    this.db = dPreAct;

    // Gradient to input: W^T · dPreAct
    return this.W.transpose().dot(dPreAct);
  }

  update(lr) {
    this.W = this.W.sub(this.dW.mul(lr));
    this.b = this.b.sub(this.db.mul(lr));
  }
}

/**
 * Variational Autoencoder
 */
export class VAE {
  /**
   * @param {number} inputSize - Dimension of input data
   * @param {number} hiddenSize - Encoder/decoder hidden layer size
   * @param {number} latentSize - Dimension of latent space
   * @param {Object} opts
   * @param {number} [opts.learningRate=0.001]
   * @param {number} [opts.beta=1] - Weight of KL divergence (β-VAE)
   */
  constructor(inputSize, hiddenSize, latentSize, opts = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.latentSize = latentSize;
    this.learningRate = opts.learningRate || 0.001;
    this.beta = opts.beta || 1;

    // Encoder: input → hidden → (μ, log_σ²)
    this.encHidden = new VAEDense(inputSize, hiddenSize, 'relu');
    this.encMu = new VAEDense(hiddenSize, latentSize, 'linear');
    this.encLogVar = new VAEDense(hiddenSize, latentSize, 'linear');

    // Decoder: z → hidden → reconstruction
    this.decHidden = new VAEDense(latentSize, hiddenSize, 'relu');
    this.decOutput = new VAEDense(hiddenSize, inputSize, 'sigmoid');
  }

  /**
   * Encode input to latent distribution parameters.
   * @param {Matrix} input - (inputSize × 1)
   * @returns {{ mu: Matrix, logVar: Matrix }}
   */
  encode(input) {
    const h = this.encHidden.forward(input);
    const mu = this.encMu.forward(h);
    const logVar = this.encLogVar.forward(h);
    return { mu, logVar };
  }

  /**
   * Reparameterization trick: z = μ + σ · ε
   * @param {Matrix} mu
   * @param {Matrix} logVar
   * @returns {{ z: Matrix, epsilon: Matrix }}
   */
  reparameterize(mu, logVar) {
    const epsilon = new Matrix(mu.rows, 1);
    for (let i = 0; i < epsilon.data.length; i++) {
      // Box-Muller transform for standard normal
      const u1 = Math.random();
      const u2 = Math.random();
      epsilon.data[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    }
    // z = mu + exp(0.5 * logVar) * epsilon
    const std = logVar.map(x => Math.exp(0.5 * x));
    const z = mu.add(std.map((s, i) => s * epsilon.data[i]));
    return { z, epsilon };
  }

  /**
   * Decode latent vector to reconstruction.
   * @param {Matrix} z - (latentSize × 1)
   * @returns {Matrix} Reconstruction (inputSize × 1)
   */
  decode(z) {
    const h = this.decHidden.forward(z);
    return this.decOutput.forward(h);
  }

  /**
   * Forward pass: encode → reparameterize → decode
   * @param {Matrix|number[]} input
   * @returns {{ reconstruction: Matrix, mu: Matrix, logVar: Matrix, z: Matrix }}
   */
  forward(input) {
    const x = Array.isArray(input)
      ? new Matrix(input.length, 1, new Float64Array(input))
      : input;

    const { mu, logVar } = this.encode(x);
    const { z, epsilon } = this.reparameterize(mu, logVar);
    const reconstruction = this.decode(z);

    this._lastInput = x;
    this._lastMu = mu;
    this._lastLogVar = logVar;
    this._lastZ = z;
    this._lastEpsilon = epsilon;

    return { reconstruction, mu, logVar, z };
  }

  /**
   * Compute VAE loss = reconstruction + β · KL divergence
   * @param {Matrix} input - Original input
   * @param {Matrix} reconstruction - Reconstructed output
   * @param {Matrix} mu - Latent mean
   * @param {Matrix} logVar - Latent log variance
   * @returns {{ total: number, recon: number, kl: number }}
   */
  computeLoss(input, reconstruction, mu, logVar) {
    // Binary cross-entropy reconstruction loss
    let reconLoss = 0;
    for (let i = 0; i < input.data.length; i++) {
      const x = input.data[i];
      const r = Math.max(1e-8, Math.min(1 - 1e-8, reconstruction.data[i]));
      reconLoss -= x * Math.log(r) + (1 - x) * Math.log(1 - r);
    }

    // KL divergence: -0.5 · Σ(1 + log(σ²) - μ² - σ²)
    let klLoss = 0;
    for (let i = 0; i < mu.data.length; i++) {
      klLoss += -0.5 * (1 + logVar.data[i] - mu.data[i] ** 2 - Math.exp(logVar.data[i]));
    }

    return {
      total: reconLoss + this.beta * klLoss,
      recon: reconLoss,
      kl: klLoss,
    };
  }

  /**
   * Train on a single sample (forward + backward + update).
   * @param {Matrix|number[]} input
   * @returns {{ loss: number, reconLoss: number, klLoss: number }}
   */
  trainStep(input) {
    const { reconstruction, mu, logVar, z } = this.forward(input);
    const x = this._lastInput;
    const { total, recon, kl } = this.computeLoss(x, reconstruction, mu, logVar);

    // Backward pass: compute gradients
    // dL/d_reconstruction = -(x/r - (1-x)/(1-r))
    const dRecon = new Matrix(x.rows, 1);
    for (let i = 0; i < x.data.length; i++) {
      const r = Math.max(1e-8, Math.min(1 - 1e-8, reconstruction.data[i]));
      dRecon.data[i] = -(x.data[i] / r - (1 - x.data[i]) / (1 - r));
    }

    // Decoder backward
    const dDecHidden = this.decOutput.backward(dRecon);
    const dZ = this.decHidden.backward(dDecHidden);

    // Gradient through reparameterization
    // dL/dμ = dL/dz + β · μ (from KL)
    const dMu = new Matrix(mu.rows, 1);
    for (let i = 0; i < mu.data.length; i++) {
      dMu.data[i] = dZ.data[i] + this.beta * mu.data[i];
    }

    // dL/d_logVar = 0.5 · dL/dz · ε · exp(0.5·logVar) + β · 0.5 · (exp(logVar) - 1)
    const dLogVar = new Matrix(logVar.rows, 1);
    for (let i = 0; i < logVar.data.length; i++) {
      const std = Math.exp(0.5 * logVar.data[i]);
      dLogVar.data[i] = 0.5 * dZ.data[i] * this._lastEpsilon.data[i] * std
        + this.beta * 0.5 * (Math.exp(logVar.data[i]) - 1);
    }

    // Encoder backward
    const dEncMuH = this.encMu.backward(dMu);
    const dEncLogVarH = this.encLogVar.backward(dLogVar);
    // Combined gradient to hidden
    const dH = dEncMuH.add(dEncLogVarH);
    this.encHidden.backward(dH);

    // Update all weights
    const lr = this.learningRate;
    this.encHidden.update(lr);
    this.encMu.update(lr);
    this.encLogVar.update(lr);
    this.decHidden.update(lr);
    this.decOutput.update(lr);

    return { loss: total, reconLoss: recon, klLoss: kl };
  }

  /**
   * Train on a dataset.
   * @param {Array} data - Array of inputs
   * @param {Object} opts
   * @param {number} [opts.epochs=10]
   * @param {Function} [opts.onEpoch]
   * @returns {{ history: { loss: number, recon: number, kl: number }[] }}
   */
  train(data, opts = {}) {
    const { epochs = 10, onEpoch } = opts;
    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0, totalRecon = 0, totalKL = 0;
      for (const input of data) {
        const { loss, reconLoss, klLoss } = this.trainStep(input);
        totalLoss += loss;
        totalRecon += reconLoss;
        totalKL += klLoss;
      }
      const n = data.length;
      const stats = {
        loss: totalLoss / n,
        recon: totalRecon / n,
        kl: totalKL / n,
      };
      history.push(stats);
      if (onEpoch) onEpoch({ epoch, ...stats });
    }

    return { history };
  }

  /**
   * Generate new samples by sampling from the prior z ~ N(0,1).
   * @param {number} [count=1]
   * @returns {Matrix[]}
   */
  generate(count = 1) {
    const samples = [];
    for (let i = 0; i < count; i++) {
      const z = new Matrix(this.latentSize, 1);
      for (let j = 0; j < this.latentSize; j++) {
        const u1 = Math.random();
        const u2 = Math.random();
        z.data[j] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
      }
      samples.push(this.decode(z));
    }
    return samples;
  }

  /**
   * Interpolate between two inputs in latent space.
   * @param {number[]|Matrix} input1
   * @param {number[]|Matrix} input2
   * @param {number} steps - Number of interpolation steps
   * @returns {Matrix[]}
   */
  interpolate(input1, input2, steps = 10) {
    const { mu: mu1 } = this.forward(input1);
    const { mu: mu2 } = this.forward(input2);

    const results = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const z = mu1.mul(1 - t).add(mu2.mul(t));
      results.push(this.decode(z));
    }
    return results;
  }
}
