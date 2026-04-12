// data.js — Data utilities for training neural networks
// Preprocessing, augmentation, splitting, and batching

import { Matrix } from './matrix.js';

/**
 * Shuffle dataset (inputs + targets together)
 * @param {Matrix} inputs
 * @param {Matrix} targets
 * @returns {{inputs: Matrix, targets: Matrix}}
 */
export function shuffle(inputs, targets) {
  const n = inputs.rows;
  const indices = Array.from({ length: n }, (_, i) => i);
  
  // Fisher-Yates shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  const newInputs = new Matrix(n, inputs.cols);
  const newTargets = new Matrix(n, targets.cols);
  
  for (let i = 0; i < n; i++) {
    const src = indices[i];
    for (let j = 0; j < inputs.cols; j++) {
      newInputs.set(i, j, inputs.get(src, j));
    }
    for (let j = 0; j < targets.cols; j++) {
      newTargets.set(i, j, targets.get(src, j));
    }
  }
  
  return { inputs: newInputs, targets: newTargets };
}

/**
 * Split dataset into train/test sets
 * @param {Matrix} inputs
 * @param {Matrix} targets
 * @param {number} testRatio - Fraction for test set (0-1)
 * @returns {{train: {inputs, targets}, test: {inputs, targets}}}
 */
export function trainTestSplit(inputs, targets, testRatio = 0.2) {
  const n = inputs.rows;
  const testSize = Math.floor(n * testRatio);
  const trainSize = n - testSize;
  
  // Shuffle first
  const { inputs: sInputs, targets: sTargets } = shuffle(inputs, targets);
  
  return {
    train: {
      inputs: sInputs.slice(0, trainSize),
      targets: sTargets.slice(0, trainSize),
    },
    test: {
      inputs: sInputs.slice(trainSize, n),
      targets: sTargets.slice(trainSize, n),
    }
  };
}

/**
 * Normalize data to zero mean, unit variance per feature
 * @param {Matrix} data
 * @returns {{normalized: Matrix, mean: Float64Array, std: Float64Array}}
 */
export function normalize(data) {
  const n = data.rows;
  const d = data.cols;
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  
  // Compute mean
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += data.get(i, j);
    mean[j] = sum / n;
  }
  
  // Compute std
  for (let j = 0; j < d; j++) {
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const diff = data.get(i, j) - mean[j];
      sumSq += diff * diff;
    }
    std[j] = Math.sqrt(sumSq / n) || 1; // Avoid division by zero
  }
  
  // Normalize
  const normalized = new Matrix(n, d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      normalized.set(i, j, (data.get(i, j) - mean[j]) / std[j]);
    }
  }
  
  return { normalized, mean, std };
}

/**
 * Apply normalization using pre-computed statistics
 * @param {Matrix} data
 * @param {Float64Array} mean
 * @param {Float64Array} std
 * @returns {Matrix}
 */
export function applyNormalization(data, mean, std) {
  const result = new Matrix(data.rows, data.cols);
  for (let i = 0; i < data.rows; i++) {
    for (let j = 0; j < data.cols; j++) {
      result.set(i, j, (data.get(i, j) - mean[j]) / std[j]);
    }
  }
  return result;
}

/**
 * Min-max scaling to [0, 1]
 * @param {Matrix} data
 * @returns {{scaled: Matrix, min: Float64Array, max: Float64Array}}
 */
export function minMaxScale(data) {
  const n = data.rows;
  const d = data.cols;
  const min = new Float64Array(d).fill(Infinity);
  const max = new Float64Array(d).fill(-Infinity);
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const v = data.get(i, j);
      if (v < min[j]) min[j] = v;
      if (v > max[j]) max[j] = v;
    }
  }
  
  const scaled = new Matrix(n, d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const range = max[j] - min[j] || 1;
      scaled.set(i, j, (data.get(i, j) - min[j]) / range);
    }
  }
  
  return { scaled, min, max };
}

/**
 * Add Gaussian noise to data (data augmentation)
 * @param {Matrix} data
 * @param {number} stddev - Standard deviation of noise
 * @returns {Matrix}
 */
export function addNoise(data, stddev = 0.01) {
  const result = new Matrix(data.rows, data.cols);
  for (let i = 0; i < data.data.length; i++) {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.random();
    const u2 = Math.random();
    const noise = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2) * stddev;
    result.data[i] = data.data[i] + noise;
  }
  return result;
}

/**
 * Create mini-batches from dataset
 * @param {Matrix} inputs
 * @param {Matrix} targets
 * @param {number} batchSize
 * @returns {Array<{inputs: Matrix, targets: Matrix}>}
 */
export function createBatches(inputs, targets, batchSize) {
  const batches = [];
  const n = inputs.rows;
  
  for (let start = 0; start < n; start += batchSize) {
    const end = Math.min(start + batchSize, n);
    batches.push({
      inputs: inputs.slice(start, end),
      targets: targets.slice(start, end),
    });
  }
  
  return batches;
}

/**
 * One-hot encode labels
 * @param {number[]} labels - Array of integer labels
 * @param {number} numClasses - Number of classes (auto-detected if not provided)
 * @returns {Matrix}
 */
export function oneHotEncode(labels, numClasses = null) {
  const nc = numClasses || Math.max(...labels) + 1;
  const result = Matrix.zeros(labels.length, nc);
  for (let i = 0; i < labels.length; i++) {
    result.set(i, labels[i], 1);
  }
  return result;
}
