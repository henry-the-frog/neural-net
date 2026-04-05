// diffusion.js — Simple DDPM (Denoising Diffusion Probabilistic Model)
//
// Implements a 1D/2D diffusion model using the neural network library.
// Based on Ho et al. 2020 "Denoising Diffusion Probabilistic Models"
//
// The model learns to reverse a gradual noising process:
//   Forward: x_0 → x_1 → ... → x_T (data → pure noise)
//   Reverse: x_T → x_{T-1} → ... → x_0 (noise → data)

import { Network } from './network.js';
import { Matrix } from './matrix.js';

/**
 * Noise schedule: controls how much noise is added at each timestep.
 * β_t goes from β_start to β_end linearly over T steps.
 * α_t = 1 - β_t
 * ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product)
 */
export class NoiseSchedule {
  constructor(T = 100, betaStart = 0.0001, betaEnd = 0.02) {
    this.T = T;
    this.betas = new Float64Array(T);
    this.alphas = new Float64Array(T);
    this.alphasCumprod = new Float64Array(T);

    // Linear schedule
    for (let t = 0; t < T; t++) {
      this.betas[t] = betaStart + (betaEnd - betaStart) * t / (T - 1);
      this.alphas[t] = 1 - this.betas[t];
    }

    // Cumulative product of alphas
    this.alphasCumprod[0] = this.alphas[0];
    for (let t = 1; t < T; t++) {
      this.alphasCumprod[t] = this.alphasCumprod[t - 1] * this.alphas[t];
    }
  }

  /**
   * Get noise parameters at timestep t
   */
  at(t) {
    return {
      beta: this.betas[t],
      alpha: this.alphas[t],
      alphaBar: this.alphasCumprod[t],
      sqrtAlphaBar: Math.sqrt(this.alphasCumprod[t]),
      sqrtOneMinusAlphaBar: Math.sqrt(1 - this.alphasCumprod[t]),
    };
  }

  /**
   * Add noise to data at timestep t (forward process)
   * x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
   */
  addNoise(x0, t) {
    const { sqrtAlphaBar, sqrtOneMinusAlphaBar } = this.at(t);
    const noise = randomNormal(x0.length);
    const xt = new Float64Array(x0.length);
    for (let i = 0; i < x0.length; i++) {
      xt[i] = sqrtAlphaBar * x0[i] + sqrtOneMinusAlphaBar * noise[i];
    }
    return { xt, noise };
  }
}

/**
 * Cosine noise schedule — smoother than linear
 */
export class CosineSchedule extends NoiseSchedule {
  constructor(T = 100, s = 0.008) {
    super(T); // Initialize parent arrays
    const f = (t) => Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    const f0 = f(0);
    for (let t = 0; t < T; t++) {
      this.alphasCumprod[t] = f(t + 1) / f0;
      this.alphas[t] = t === 0 ? this.alphasCumprod[0] : this.alphasCumprod[t] / this.alphasCumprod[t - 1];
      this.betas[t] = Math.min(1 - this.alphas[t], 0.999);
    }
  }
}

/**
 * Simple DDPM for 1D data.
 * Uses an MLP (from Network) as the denoising model.
 */
export class SimpleDiffusion {
  constructor(dataDim, options = {}) {
    this.dataDim = dataDim;
    this.T = options.T || 100;
    this.schedule = options.cosineSchedule
      ? new CosineSchedule(this.T)
      : new NoiseSchedule(this.T, options.betaStart, options.betaEnd);

    // Build denoising network: takes [x_t, t_embedding] → predicted noise ε
    const hiddenSize = options.hiddenSize || 64;
    const inputSize = dataDim + 16; // data + time embedding
    this.net = new Network()
      .dense(inputSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, dataDim, 'linear')
      .loss('mse');

    this.lr = options.lr || 0.001;
  }

  /**
   * Create sinusoidal time embedding (positional encoding)
   */
  timeEmbed(t) {
    const embed = new Float64Array(16);
    const tNorm = t / this.T;
    for (let i = 0; i < 8; i++) {
      const freq = Math.pow(10000, -i / 8);
      embed[i * 2] = Math.sin(tNorm * freq * Math.PI * 2);
      embed[i * 2 + 1] = Math.cos(tNorm * freq * Math.PI * 2);
    }
    return embed;
  }

  /**
   * Training step: sample data, add noise, predict noise
   * Returns the loss value.
   */
  trainStep(x0) {
    // Random timestep
    const t = Math.floor(Math.random() * this.T);

    // Add noise
    const { xt, noise } = this.schedule.addNoise(x0, t);

    // Build input: [x_t, time_embedding]
    const timeEmb = this.timeEmbed(t);
    const input = new Float64Array(this.dataDim + 16);
    input.set(xt);
    input.set(timeEmb, this.dataDim);

    // Forward pass — input as row vector (1×N)
    const inputArr = [Array.from(input)];
    const targetArr = [Array.from(noise)];

    // Set training mode and run batch
    for (const l of this.net.layers) l.training = true;
    const loss = this.net.trainBatch(inputArr, targetArr, this.lr);
    for (const l of this.net.layers) l.training = false;

    return loss;
  }

  /**
   * Train on a dataset for multiple epochs
   */
  train(dataset, epochs = 10, batchSize = 32) {
    const losses = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let steps = 0;
      for (let i = 0; i < dataset.length; i++) {
        epochLoss += this.trainStep(dataset[i]);
        steps++;
      }
      losses.push(epochLoss / steps);
    }
    return losses;
  }

  /**
   * Generate a sample by running the reverse diffusion process.
   * Starts from pure noise and iteratively denoises.
   */
  sample() {
    // Start from pure noise
    let xt = randomNormal(this.dataDim);

    // Reverse diffusion: t = T-1, T-2, ..., 0
    for (let t = this.T - 1; t >= 0; t--) {
      const { alpha, beta, alphaBar } = this.schedule.at(t);
      const sqrtOneMinusAlphaBar = Math.sqrt(1 - alphaBar);
      const sqrtAlpha = Math.sqrt(alpha);

      // Predict noise
      const timeEmb = this.timeEmbed(t);
      const input = new Float64Array(this.dataDim + 16);
      input.set(xt);
      input.set(timeEmb, this.dataDim);
      const inputMat = Matrix.fromArray([Array.from(input)]);
      const predictedNoise = this.net.forward(inputMat);
      // Extract predicted noise as flat array
      const eps = [];
      for (let j = 0; j < this.dataDim; j++) {
        eps.push(predictedNoise.get(0, j));
      }

      // Reverse step: x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ) + σ_t·z
      const result = new Float64Array(this.dataDim);
      for (let i = 0; i < this.dataDim; i++) {
        const mean = (1 / sqrtAlpha) * (xt[i] - (beta / sqrtOneMinusAlphaBar) * eps[i]);
        const sigma = t > 0 ? Math.sqrt(beta) : 0; // No noise at t=0
        const z = t > 0 ? gaussianRandom() : 0;
        result[i] = mean + sigma * z;
      }
      xt = result;
    }

    return xt;
  }
}

// --- Utility functions ---

function randomNormal(n) {
  const arr = new Float64Array(n);
  for (let i = 0; i < n; i++) arr[i] = gaussianRandom();
  return arr;
}

function gaussianRandom() {
  // Box-Muller transform
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function matToArray(mat) {
  if (mat.data) return Array.from(mat.data);
  if (mat.toArray) return mat.toArray().flat();
  return [];
}
