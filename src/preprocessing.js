// preprocessing.js — Data normalization and encoding utilities
import { Matrix } from './matrix.js';

/**
 * StandardScaler: zero mean, unit variance normalization.
 */
export class StandardScaler {
  constructor() {
    this.mean = null;
    this.std = null;
    this.fitted = false;
  }

  fit(data) {
    const n = data.rows;
    const d = data.cols;
    this.mean = new Float64Array(d);
    this.std = new Float64Array(d);
    
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += data.get(i, j);
      this.mean[j] = sum / n;
      
      let sumSq = 0;
      for (let i = 0; i < n; i++) sumSq += (data.get(i, j) - this.mean[j]) ** 2;
      this.std[j] = Math.sqrt(sumSq / n) || 1; // Avoid division by zero
    }
    this.fitted = true;
    return this;
  }

  transform(data) {
    if (!this.fitted) throw new Error('Call fit() first');
    const result = Matrix.zeros(data.rows, data.cols);
    for (let i = 0; i < data.rows; i++) {
      for (let j = 0; j < data.cols; j++) {
        result.set(i, j, (data.get(i, j) - this.mean[j]) / this.std[j]);
      }
    }
    return result;
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }

  inverseTransform(data) {
    if (!this.fitted) throw new Error('Call fit() first');
    const result = Matrix.zeros(data.rows, data.cols);
    for (let i = 0; i < data.rows; i++) {
      for (let j = 0; j < data.cols; j++) {
        result.set(i, j, data.get(i, j) * this.std[j] + this.mean[j]);
      }
    }
    return result;
  }
}

/**
 * MinMaxScaler: scale to [0, 1] range.
 */
export class MinMaxScaler {
  constructor() {
    this.min = null;
    this.max = null;
    this.fitted = false;
  }

  fit(data) {
    const d = data.cols;
    this.min = new Float64Array(d).fill(Infinity);
    this.max = new Float64Array(d).fill(-Infinity);
    
    for (let i = 0; i < data.rows; i++) {
      for (let j = 0; j < d; j++) {
        const v = data.get(i, j);
        if (v < this.min[j]) this.min[j] = v;
        if (v > this.max[j]) this.max[j] = v;
      }
    }
    this.fitted = true;
    return this;
  }

  transform(data) {
    if (!this.fitted) throw new Error('Call fit() first');
    const result = Matrix.zeros(data.rows, data.cols);
    for (let i = 0; i < data.rows; i++) {
      for (let j = 0; j < data.cols; j++) {
        const range = this.max[j] - this.min[j] || 1;
        result.set(i, j, (data.get(i, j) - this.min[j]) / range);
      }
    }
    return result;
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }
}

/**
 * One-hot encode a vector of class labels.
 */
export function oneHotEncode(labels, numClasses = null) {
  if (!numClasses) numClasses = Math.max(...labels) + 1;
  const result = Matrix.zeros(labels.length, numClasses);
  for (let i = 0; i < labels.length; i++) {
    result.set(i, labels[i], 1);
  }
  return result;
}

/**
 * Train/test split.
 */
export function trainTestSplit(inputs, targets, testFraction = 0.2) {
  const n = inputs.rows;
  const testSize = Math.floor(n * testFraction);
  const trainSize = n - testSize;
  
  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  const extract = (matrix, idxs) => {
    const result = Matrix.zeros(idxs.length, matrix.cols);
    for (let i = 0; i < idxs.length; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.set(i, j, matrix.get(idxs[i], j));
      }
    }
    return result;
  };
  
  return {
    trainInputs: extract(inputs, indices.slice(0, trainSize)),
    trainTargets: extract(targets, indices.slice(0, trainSize)),
    testInputs: extract(inputs, indices.slice(trainSize)),
    testTargets: extract(targets, indices.slice(trainSize)),
  };
}
