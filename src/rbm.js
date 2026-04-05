/**
 * rbm.js — Restricted Boltzmann Machine
 * 
 * An energy-based generative model with visible and hidden layers.
 * No intra-layer connections (the "restricted" part).
 * 
 * Energy: E(v,h) = -a^T·v - b^T·h - v^T·W·h
 * 
 * Learning via Contrastive Divergence (CD-k):
 * 1. Clamp visible units to data
 * 2. Sample hidden units from p(h|v)
 * 3. Reconstruct visible units from p(v|h)
 * 4. Sample hidden again from reconstructed visible
 * 5. ΔW = lr · (v_data · h_data^T - v_recon · h_recon^T)
 * 
 * Based on: Hinton (2002), "Training Products of Experts by Minimizing CD"
 */

import { Matrix } from './matrix.js';

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Restricted Boltzmann Machine
 */
export class RBM {
  /**
   * @param {number} numVisible - Number of visible units
   * @param {number} numHidden - Number of hidden units
   * @param {Object} opts
   * @param {number} [opts.learningRate=0.01]
   * @param {number} [opts.momentum=0]
   * @param {number} [opts.weightDecay=0]
   * @param {number} [opts.cdSteps=1] - CD-k steps
   */
  constructor(numVisible, numHidden, opts = {}) {
    this.numVisible = numVisible;
    this.numHidden = numHidden;
    this.learningRate = opts.learningRate || 0.01;
    this.momentum = opts.momentum || 0;
    this.weightDecay = opts.weightDecay || 0;
    this.cdSteps = opts.cdSteps || 1;

    // Weights: numVisible × numHidden
    this.W = new Matrix(numVisible, numHidden);
    const scale = 0.1;
    for (let i = 0; i < this.W.data.length; i++) {
      this.W.data[i] = (Math.random() * 2 - 1) * scale;
    }

    // Visible biases
    this.a = new Matrix(numVisible, 1);
    // Hidden biases
    this.b = new Matrix(numHidden, 1);

    // Momentum terms
    this._dW = Matrix.zeros(numVisible, numHidden);
    this._da = Matrix.zeros(numVisible, 1);
    this._db = Matrix.zeros(numHidden, 1);
  }

  /**
   * Compute p(h=1|v) = sigmoid(W^T · v + b)
   * @param {Matrix} v - Visible state (numVisible × 1)
   * @returns {Matrix} Hidden probabilities (numHidden × 1)
   */
  hiddenProbs(v) {
    // W^T · v + b: (numHidden × numVisible) · (numVisible × 1) + (numHidden × 1)
    const activation = this.W.transpose().dot(v).add(this.b);
    return activation.map(x => sigmoid(x));
  }

  /**
   * Sample hidden units given visible.
   * @param {Matrix} v - Visible state
   * @returns {{ probs: Matrix, samples: Matrix }}
   */
  sampleHidden(v) {
    const probs = this.hiddenProbs(v);
    const samples = probs.map(p => Math.random() < p ? 1 : 0);
    return { probs, samples };
  }

  /**
   * Compute p(v=1|h) = sigmoid(W · h + a)
   * @param {Matrix} h - Hidden state (numHidden × 1)
   * @returns {Matrix} Visible probabilities (numVisible × 1)
   */
  visibleProbs(h) {
    // W · h + a: (numVisible × numHidden) · (numHidden × 1) + (numVisible × 1)
    const activation = this.W.dot(h).add(this.a);
    return activation.map(x => sigmoid(x));
  }

  /**
   * Sample visible units given hidden.
   * @param {Matrix} h - Hidden state
   * @returns {{ probs: Matrix, samples: Matrix }}
   */
  sampleVisible(h) {
    const probs = this.visibleProbs(h);
    const samples = probs.map(p => Math.random() < p ? 1 : 0);
    return { probs, samples };
  }

  /**
   * Contrastive Divergence training step (CD-k).
   * @param {Matrix} data - Visible data (numVisible × 1)
   * @returns {{ reconstructionError: number }}
   */
  trainStep(data) {
    // Positive phase: clamp visible to data
    const v0 = data;
    const { probs: h0Probs } = this.sampleHidden(v0);

    // Negative phase: Gibbs sampling for k steps
    let vk = v0;
    let hkProbs = h0Probs;
    for (let k = 0; k < this.cdSteps; k++) {
      const { probs: vProbs } = this.sampleVisible(hkProbs);
      vk = vProbs; // Use probabilities for visible reconstruction (less noisy)
      const { probs } = this.sampleHidden(vk);
      hkProbs = probs;
    }

    // Gradient: positive correlations - negative correlations
    // ΔW = v0 · h0^T - vk · hk^T
    const posGrad = v0.dot(h0Probs.transpose());    // (nVis × nHid)
    const negGrad = vk.dot(hkProbs.transpose());     // (nVis × nHid)
    const dW = posGrad.sub(negGrad);
    const da = v0.sub(vk);
    const db = h0Probs.sub(hkProbs);

    // Apply momentum
    this._dW = this._dW.mul(this.momentum).add(dW.mul(this.learningRate));
    this._da = this._da.mul(this.momentum).add(da.mul(this.learningRate));
    this._db = this._db.mul(this.momentum).add(db.mul(this.learningRate));

    // Update parameters
    this.W = this.W.add(this._dW);
    if (this.weightDecay > 0) {
      this.W = this.W.sub(this.W.mul(this.weightDecay * this.learningRate));
    }
    this.a = this.a.add(this._da);
    this.b = this.b.add(this._db);

    // Compute reconstruction error
    let error = 0;
    for (let i = 0; i < v0.data.length; i++) {
      error += (v0.data[i] - vk.data[i]) ** 2;
    }
    return { reconstructionError: error / v0.data.length };
  }

  /**
   * Train on a dataset.
   * @param {Matrix[]|number[][]} data - Array of visible vectors
   * @param {Object} opts
   * @param {number} [opts.epochs=10]
   * @param {Function} [opts.onEpoch]
   * @returns {{ history: number[] }} Avg reconstruction error per epoch
   */
  train(data, opts = {}) {
    const { epochs = 10, onEpoch } = opts;
    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalError = 0;
      for (let i = 0; i < data.length; i++) {
        const v = Array.isArray(data[i])
          ? new Matrix(data[i].length, 1, new Float64Array(data[i]))
          : data[i];
        const { reconstructionError } = this.trainStep(v);
        totalError += reconstructionError;
      }
      const avgError = totalError / data.length;
      history.push(avgError);
      if (onEpoch) onEpoch({ epoch, avgError });
    }

    return { history };
  }

  /**
   * Reconstruct a visible vector through one pass.
   * v → h → v_recon
   * @param {Matrix|number[]} visible
   * @returns {{ reconstruction: Matrix, hiddenActivation: Matrix }}
   */
  reconstruct(visible) {
    const v = Array.isArray(visible)
      ? new Matrix(visible.length, 1, new Float64Array(visible))
      : visible;
    const { probs: hProbs } = this.sampleHidden(v);
    const { probs: vRecon } = this.sampleVisible(hProbs);
    return { reconstruction: vRecon, hiddenActivation: hProbs };
  }

  /**
   * Generate a sample by running Gibbs sampling from random initialization.
   * @param {number} steps - Number of Gibbs steps
   * @returns {Matrix} Generated visible state
   */
  generate(steps = 100) {
    let h = new Matrix(this.numHidden, 1);
    for (let i = 0; i < h.data.length; i++) {
      h.data[i] = Math.random() < 0.5 ? 1 : 0;
    }

    for (let step = 0; step < steps; step++) {
      const { probs: vProbs } = this.sampleVisible(h);
      const { probs: hProbs } = this.sampleHidden(vProbs);
      h = hProbs;
    }

    const { probs } = this.sampleVisible(h);
    return probs;
  }

  /**
   * Compute the free energy of a visible vector.
   * F(v) = -a^T·v - Σ_j log(1 + exp(W_j^T · v + b_j))
   * Lower energy = more likely under the model.
   * @param {Matrix|number[]} visible
   * @returns {number}
   */
  freeEnergy(visible) {
    const v = Array.isArray(visible)
      ? new Matrix(visible.length, 1, new Float64Array(visible))
      : visible;

    // -a^T · v
    let energy = 0;
    for (let i = 0; i < this.numVisible; i++) {
      energy -= this.a.data[i] * v.data[i];
    }

    // -Σ log(1 + exp(W_j^T · v + b_j))
    const wx = this.W.transpose().dot(v).add(this.b);
    for (let j = 0; j < this.numHidden; j++) {
      energy -= Math.log(1 + Math.exp(wx.data[j]));
    }

    return energy;
  }

  /**
   * Extract learned features (hidden representations).
   * @param {Matrix|number[]} visible
   * @returns {number[]}
   */
  encode(visible) {
    const v = Array.isArray(visible)
      ? new Matrix(visible.length, 1, new Float64Array(visible))
      : visible;
    return Array.from(this.hiddenProbs(v).data);
  }

  /**
   * Decode from hidden to visible.
   * @param {number[]} hidden
   * @returns {number[]}
   */
  decode(hidden) {
    const h = new Matrix(hidden.length, 1, new Float64Array(hidden));
    return Array.from(this.visibleProbs(h).data);
  }
}
