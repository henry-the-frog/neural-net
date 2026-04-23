// vae.js — Variational Autoencoder (Kingma & Welling, 2014)
// Learns a latent space representation with smooth, continuous structure.
//
// Encoder: x → μ, log(σ²) — produces mean and log-variance
// Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
// Decoder: z → x̂ — reconstructs the input
// Loss: L = reconstruction_loss + β * KL(q(z|x) || p(z))
//       KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

import { Matrix } from './matrix.js';

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function randn() {
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export class VAE {
  /**
   * @param {number} inputDim - Input dimension
   * @param {number} hiddenDim - Hidden layer dimension
   * @param {number} latentDim - Latent space dimension
   * @param {number} beta - KL weight (β-VAE, Higgins 2017)
   */
  constructor(inputDim, hiddenDim, latentDim, beta = 1.0) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.latentDim = latentDim;
    this.beta = beta;
    
    // Encoder: input → hidden → (mu, logvar)
    this.We1 = Matrix.random(inputDim, hiddenDim).map(v => v * Math.sqrt(2.0 / inputDim));
    this.be1 = new Float64Array(hiddenDim);
    this.Wmu = Matrix.random(hiddenDim, latentDim).map(v => v * Math.sqrt(2.0 / hiddenDim));
    this.bmu = new Float64Array(latentDim);
    this.Wlogvar = Matrix.random(hiddenDim, latentDim).map(v => v * Math.sqrt(2.0 / hiddenDim));
    this.blogvar = new Float64Array(latentDim);
    
    // Decoder: latent → hidden → output
    this.Wd1 = Matrix.random(latentDim, hiddenDim).map(v => v * Math.sqrt(2.0 / latentDim));
    this.bd1 = new Float64Array(hiddenDim);
    this.Wd2 = Matrix.random(hiddenDim, inputDim).map(v => v * Math.sqrt(2.0 / hiddenDim));
    this.bd2 = new Float64Array(inputDim);
  }

  /**
   * Encode input to latent distribution parameters.
   * @param {Float64Array} x - Input
   * @returns {{ mu: Float64Array, logvar: Float64Array }}
   */
  encode(x) {
    // Hidden layer with ReLU
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.be1[j];
      for (let i = 0; i < this.inputDim; i++) sum += x[i] * this.We1.get(i, j);
      hidden[j] = Math.max(0, sum);
    }
    
    const mu = new Float64Array(this.latentDim);
    const logvar = new Float64Array(this.latentDim);
    for (let j = 0; j < this.latentDim; j++) {
      let sumMu = this.bmu[j], sumLv = this.blogvar[j];
      for (let i = 0; i < this.hiddenDim; i++) {
        sumMu += hidden[i] * this.Wmu.get(i, j);
        sumLv += hidden[i] * this.Wlogvar.get(i, j);
      }
      mu[j] = sumMu;
      logvar[j] = sumLv;
    }
    
    return { mu, logvar };
  }

  /**
   * Reparameterization trick: z = μ + σ * ε.
   */
  reparameterize(mu, logvar) {
    const z = new Float64Array(this.latentDim);
    for (let i = 0; i < this.latentDim; i++) {
      const std = Math.exp(0.5 * logvar[i]);
      z[i] = mu[i] + std * randn();
    }
    return z;
  }

  /**
   * Decode latent vector to reconstruction.
   * @param {Float64Array} z - Latent vector
   * @returns {Float64Array} Reconstruction (sigmoid applied)
   */
  decode(z) {
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.bd1[j];
      for (let i = 0; i < this.latentDim; i++) sum += z[i] * this.Wd1.get(i, j);
      hidden[j] = Math.max(0, sum);
    }
    
    const output = new Float64Array(this.inputDim);
    for (let j = 0; j < this.inputDim; j++) {
      let sum = this.bd2[j];
      for (let i = 0; i < this.hiddenDim; i++) sum += hidden[i] * this.Wd2.get(i, j);
      output[j] = sigmoid(sum); // Sigmoid for [0,1] output
    }
    
    return output;
  }

  /**
   * Full forward pass: encode → reparameterize → decode.
   */
  forward(x) {
    const { mu, logvar } = this.encode(x);
    const z = this.reparameterize(mu, logvar);
    const reconstruction = this.decode(z);
    return { reconstruction, mu, logvar, z };
  }

  /**
   * Compute ELBO loss.
   * @param {Float64Array} x - Original input
   * @param {Float64Array} reconstruction - Decoded output
   * @param {Float64Array} mu - Latent mean
   * @param {Float64Array} logvar - Latent log-variance
   * @returns {{ total: number, recon: number, kl: number }}
   */
  loss(x, reconstruction, mu, logvar) {
    // Reconstruction loss (binary cross-entropy)
    let recon = 0;
    for (let i = 0; i < x.length; i++) {
      const p = Math.max(1e-10, Math.min(1 - 1e-10, reconstruction[i]));
      recon -= x[i] * Math.log(p) + (1 - x[i]) * Math.log(1 - p);
    }
    
    // KL divergence: -0.5 * Σ(1 + logvar - mu² - exp(logvar))
    let kl = 0;
    for (let i = 0; i < this.latentDim; i++) {
      kl += -0.5 * (1 + logvar[i] - mu[i] * mu[i] - Math.exp(logvar[i]));
    }
    
    return {
      total: recon + this.beta * kl,
      recon,
      kl,
    };
  }

  /**
   * Sample from the latent space.
   * @returns {Float64Array} Generated sample
   */
  sample() {
    const z = new Float64Array(this.latentDim).map(() => randn());
    return this.decode(z);
  }

  /**
   * Interpolate between two inputs in latent space.
   */
  interpolate(x1, x2, steps = 5) {
    const { mu: mu1 } = this.encode(x1);
    const { mu: mu2 } = this.encode(x2);
    
    const results = [];
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const z = new Float64Array(this.latentDim);
      for (let i = 0; i < this.latentDim; i++) {
        z[i] = mu1[i] * (1 - t) + mu2[i] * t;
      }
      results.push(this.decode(z));
    }
    return results;
  }
}
