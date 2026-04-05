/**
 * predictive-coding.js — Predictive Coding Network
 * 
 * Implements hierarchical predictive coding as described by Rao & Ballard (1999)
 * and connected to the Free Energy Principle (Friston, 2005).
 * 
 * Key idea: Each layer generates top-down predictions of the layer below.
 * Prediction errors propagate bottom-up and drive learning.
 * All learning is LOCAL — no backpropagation needed.
 * 
 * Architecture per layer:
 *   - Value nodes μ: current representation
 *   - Error nodes ε: prediction error = input - prediction
 *   - Weights W: generative model (predicts lower layer from higher)
 *   - Precision Π: inverse variance (confidence in predictions)
 */

import { Matrix } from './matrix.js';

// Activation functions
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function sigmoidDeriv(x) { const s = sigmoid(x); return s * (1 - s); }
function tanh(x) { return Math.tanh(x); }
function tanhDeriv(x) { const t = Math.tanh(x); return 1 - t * t; }
function relu(x) { return x > 0 ? x : 0; }
function reluDeriv(x) { return x > 0 ? 1 : 0; }
function identity(x) { return x; }
function identityDeriv(_x) { return 1; }

const ACTIVATIONS = {
  sigmoid: [sigmoid, sigmoidDeriv],
  tanh: [tanh, tanhDeriv],
  relu: [relu, reluDeriv],
  linear: [identity, identityDeriv],
};

/**
 * A single layer in the predictive coding hierarchy.
 * 
 * Contains value nodes (μ), error nodes (ε), and generative weights (W).
 * The generative model: prediction = f(W · μ_above + b)
 * Error: ε = input_from_below - prediction
 */
export class PredictiveCodingLayer {
  /**
   * @param {number} size - Number of value nodes in this layer
   * @param {number} inputSize - Size of the layer below (that this layer predicts)
   * @param {Object} opts
   * @param {string} [opts.activation='sigmoid'] - Activation function
   * @param {number} [opts.precision=1.0] - Initial precision (inverse variance)
   * @param {number} [opts.learningRate=0.01] - Weight learning rate
   * @param {number} [opts.inferenceRate=0.1] - Value node inference rate
   */
  constructor(size, inputSize, opts = {}) {
    this.size = size;
    this.inputSize = inputSize;

    const {
      activation = 'sigmoid',
      precision = 1.0,
      learningRate = 0.01,
      inferenceRate = 0.1,
    } = opts;

    const [act, actDeriv] = ACTIVATIONS[activation] || ACTIVATIONS.sigmoid;
    this.activation = act;
    this.activationDeriv = actDeriv;
    this.learningRate = learningRate;
    this.inferenceRate = inferenceRate;

    // Value nodes μ (the layer's representation)
    this.mu = new Matrix(size, 1);
    this._initRandom(this.mu, 0.1);

    // Error nodes ε (prediction error)
    this.epsilon = new Matrix(inputSize, 1);

    // Generative weights W: maps this layer → prediction of layer below
    // prediction = f(W · μ + b) where W is inputSize × size
    this.W = new Matrix(inputSize, size);
    this._initRandom(this.W, Math.sqrt(2 / (size + inputSize)));

    // Bias
    this.b = new Matrix(inputSize, 1);

    // Precision (scalar for simplicity, could be per-unit or full matrix)
    this.precision = precision;

    // Pre-activation cache (for derivative computation)
    this._preActivation = new Matrix(inputSize, 1);
  }

  /**
   * Initialize matrix with random values.
   */
  _initRandom(m, scale) {
    for (let i = 0; i < m.data.length; i++) {
      m.data[i] = (Math.random() * 2 - 1) * scale;
    }
  }

  /**
   * Generate a top-down prediction of the layer below.
   * prediction = f(W · μ + b)
   * @returns {Matrix} Predicted values (inputSize × 1)
   */
  predict() {
    // W (inputSize × size) · μ (size × 1) = (inputSize × 1)
    const preAct = this.W.dot(this.mu).add(this.b);
    this._preActivation = preAct;
    const result = new Matrix(preAct.rows, preAct.cols);
    for (let i = 0; i < result.data.length; i++) {
      result.data[i] = this.activation(preAct.data[i]);
    }
    return result;
  }

  /**
   * Compute prediction error given actual input from below.
   * ε = actual - predicted
   * @param {Matrix} actual - Actual input from the layer below
   * @returns {Matrix} Prediction error
   */
  computeError(actual) {
    const predicted = this.predict();
    this.epsilon = actual.sub(predicted);
    return this.epsilon;
  }

  /**
   * Update value nodes μ based on prediction errors.
   * 
   * The value update follows:
   * dμ/dt = -ε_own + W^T · Π · ε_below · f'(W·μ+b)
   * 
   * Where:
   * - ε_own: this layer's own prediction error (from layer above)
   * - ε_below: error at the layer below (this layer's prediction error)
   * - W^T: transpose of generative weights
   * - Π: precision (confidence)
   * - f': derivative of activation
   * 
   * @param {Matrix|null} errorFromAbove - Error signal from the layer above (null for top layer)
   */
  updateValues(errorFromAbove) {
    // Contribution from prediction error below (this layer's predictions were wrong)
    // W^T · (Π · ε_below · f'(pre_act))
    const scaledError = new Matrix(this.epsilon.rows, this.epsilon.cols);
    for (let i = 0; i < scaledError.data.length; i++) {
      scaledError.data[i] = this.precision * this.epsilon.data[i]
        * this.activationDeriv(this._preActivation.data[i]);
    }
    // W^T (size × inputSize) · scaledError (inputSize × 1) = (size × 1)
    const bottomUp = this.W.transpose().dot(scaledError);

    // Combine: move μ toward explaining the errors below
    let delta = bottomUp.mul(this.inferenceRate);

    // If there's error from above, also minimize that
    if (errorFromAbove) {
      // ε_above pushes μ toward what the layer above predicted
      delta = delta.sub(errorFromAbove.mul(this.inferenceRate));
    }

    this.mu = this.mu.add(delta);
  }

  /**
   * Update weights based on converged prediction errors.
   * This is the LEARNING step (after inference has converged).
   * 
   * dW/dt = Π · ε · f'(pre_act) · μ^T  (Hebbian-like!)
   * db/dt = Π · ε · f'(pre_act)
   */
  updateWeights() {
    // Compute gradient
    const scaledError = new Matrix(this.epsilon.rows, this.epsilon.cols);
    for (let i = 0; i < scaledError.data.length; i++) {
      scaledError.data[i] = this.precision * this.epsilon.data[i]
        * this.activationDeriv(this._preActivation.data[i]);
    }

    // dW = scaledError · μ^T (inputSize × 1) · (1 × size) = (inputSize × size)
    const dW = scaledError.dot(this.mu.transpose());
    this.W = this.W.add(dW.mul(this.learningRate));

    // db = scaledError
    this.b = this.b.add(scaledError.mul(this.learningRate));
  }

  /**
   * Get the current prediction error energy (squared error).
   */
  get energy() {
    let sum = 0;
    for (let i = 0; i < this.epsilon.data.length; i++) {
      sum += this.epsilon.data[i] * this.epsilon.data[i];
    }
    return 0.5 * this.precision * sum;
  }

  /**
   * Reset value nodes to small random values.
   */
  resetValues() {
    this._initRandom(this.mu, 0.1);
    for (let i = 0; i < this.epsilon.data.length; i++) {
      this.epsilon.data[i] = 0;
    }
  }

  /**
   * Clone this layer.
   */
  clone() {
    const layer = new PredictiveCodingLayer(this.size, this.inputSize, {
      precision: this.precision,
      learningRate: this.learningRate,
      inferenceRate: this.inferenceRate,
    });
    layer.W = new Matrix(this.W.rows, this.W.cols, new Float64Array(this.W.data));
    layer.b = new Matrix(this.b.rows, this.b.cols, new Float64Array(this.b.data));
    layer.mu = new Matrix(this.mu.rows, this.mu.cols, new Float64Array(this.mu.data));
    layer.activation = this.activation;
    layer.activationDeriv = this.activationDeriv;
    return layer;
  }
}

/**
 * Predictive Coding Network — hierarchical architecture.
 * 
 * Layers are stacked bottom-to-top:
 *   Layer 0: closest to sensory input (predicts input)
 *   Layer N: highest level of abstraction
 * 
 * Each layer generates predictions of the layer below it.
 * Prediction errors propagate upward.
 * All learning is local (no backpropagation).
 * 
 * Usage:
 *   const pc = new PredictiveCodingNetwork([inputSize, hidden1, hidden2, outputSize]);
 *   const output = pc.infer(input, { inferenceSteps: 50 });
 *   pc.learn(input); // Update weights based on converged state
 */
export class PredictiveCodingNetwork {
  /**
   * @param {number[]} layerSizes - Sizes for each layer [input, hidden..., top]
   * @param {Object} opts
   * @param {string} [opts.activation='sigmoid']
   * @param {number} [opts.inferenceSteps=50]
   * @param {number} [opts.learningRate=0.01]
   * @param {number} [opts.inferenceRate=0.1]
   * @param {number} [opts.precision=1.0]
   */
  constructor(layerSizes, opts = {}) {
    if (layerSizes.length < 2) throw new Error('Need at least 2 layer sizes');
    this.layerSizes = layerSizes;
    this.inferenceSteps = opts.inferenceSteps || 50;

    // Create layers (each layer predicts the layer below)
    // Layer i has size layerSizes[i+1] and predicts layerSizes[i]
    this.layers = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(new PredictiveCodingLayer(
        layerSizes[i + 1],  // this layer's size
        layerSizes[i],       // what it predicts (layer below)
        {
          activation: opts.activation || 'sigmoid',
          precision: opts.precision || 1.0,
          learningRate: opts.learningRate || 0.01,
          inferenceRate: opts.inferenceRate || 0.1,
        }
      ));
    }
  }

  /**
   * Run inference: clamp input, iterate until convergence.
   * 
   * Process:
   * 1. Clamp bottom layer to input
   * 2. For each inference step:
   *    a. Compute prediction errors bottom-up
   *    b. Update value nodes
   * 3. Return top layer's value (the network's "understanding")
   * 
   * @param {Matrix|number[]} input - Input data (inputSize × 1 or array)
   * @param {Object} opts
   * @param {number} [opts.steps] - Override inference steps
   * @returns {{ output: Matrix, energy: number, converged: boolean }}
   */
  infer(input, opts = {}) {
    const steps = opts.steps || this.inferenceSteps;
    const inputMat = Array.isArray(input) 
      ? new Matrix(input.length, 1, new Float64Array(input))
      : input;

    // Reset value nodes for fresh inference
    for (const layer of this.layers) {
      layer.resetValues();
    }

    let prevEnergy = Infinity;
    let converged = false;

    for (let step = 0; step < steps; step++) {
      // Compute prediction errors bottom-up
      // Layer 0 predicts the input
      this.layers[0].computeError(inputMat);

      // Higher layers predict the layer below's value nodes
      for (let i = 1; i < this.layers.length; i++) {
        this.layers[i].computeError(this.layers[i - 1].mu);
      }

      // Update value nodes
      for (let i = 0; i < this.layers.length; i++) {
        // Error from above: the error that the layer above computed about THIS layer
        const errorFromAbove = (i < this.layers.length - 1)
          ? this.layers[i + 1].epsilon // layer above's prediction error about us
          : null;

        // Wait — the error from above should be in the VALUE space of this layer
        // Layer i+1 predicts layer i's mu. Its epsilon = layer_i.mu - prediction_{i+1}
        // But layer i+1 stores epsilon of size layerSizes[i] (what it predicts)
        // which IS the right size for this layer's mu.
        // Actually: layer i+1 has epsilon of size layerSizes[i+1-1] = layerSizes[i]
        // Wait no: layer i+1 has inputSize = layerSizes[i+1] and size = layerSizes[i+2]
        // Its epsilon.rows = layerSizes[i+1] ≠ layerSizes[i+1] = layer i's size ✓

        // Hmm, let me re-check dimensions:
        // layer[i].size = layerSizes[i+1]
        // layer[i+1].inputSize = layerSizes[i+1] (what layer i+1 predicts)
        // layer[i+1].epsilon.rows = layerSizes[i+1] = layer[i].mu.rows ✓
        
        this.layers[i].updateValues(errorFromAbove);
      }

      // Check convergence
      const energy = this.totalEnergy();
      if (Math.abs(energy - prevEnergy) < 1e-6) {
        converged = true;
        break;
      }
      prevEnergy = energy;
    }

    return {
      output: this.layers[this.layers.length - 1].mu,
      energy: this.totalEnergy(),
      converged,
    };
  }

  /**
   * Learn from input (update weights after inference converges).
   * 
   * @param {Matrix|number[]} input - Input data
   * @param {Object} opts
   * @param {number} [opts.steps] - Inference steps before learning
   * @returns {{ energy: number }}
   */
  learn(input, opts = {}) {
    // First, run inference to convergence
    const result = this.infer(input, opts);

    // Then update weights at all layers (local learning)
    for (const layer of this.layers) {
      layer.updateWeights();
    }

    return { energy: result.energy };
  }

  /**
   * Train on multiple inputs.
   * @param {Matrix[]|number[][]} inputs - Array of input samples
   * @param {Object} opts
   * @param {number} [opts.epochs=1]
   * @param {Function} [opts.onEpoch] - Callback with { epoch, avgEnergy }
   * @returns {{ history: number[] }} Average energy per epoch
   */
  train(inputs, opts = {}) {
    const { epochs = 1, onEpoch } = opts;
    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalEnergy = 0;
      for (const input of inputs) {
        const result = this.learn(input, opts);
        totalEnergy += result.energy;
      }
      const avgEnergy = totalEnergy / inputs.length;
      history.push(avgEnergy);
      if (onEpoch) onEpoch({ epoch, avgEnergy });
    }

    return { history };
  }

  /**
   * Total free energy across all layers.
   */
  totalEnergy() {
    return this.layers.reduce((sum, l) => sum + l.energy, 0);
  }

  /**
   * Get the reconstruction of the input from the network's internal model.
   * Runs the top layer's prediction all the way down.
   */
  reconstruct() {
    return this.layers[0].predict();
  }

  /**
   * Get anomaly score for an input.
   * Higher score = more surprising (harder to predict).
   * @param {Matrix|number[]} input
   * @returns {number} Free energy (anomaly score)
   */
  anomalyScore(input) {
    const { energy } = this.infer(input);
    return energy;
  }
}
