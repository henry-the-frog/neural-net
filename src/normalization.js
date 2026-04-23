// normalization.js — All normalization techniques in one place
// LayerNorm, RMSNorm, InstanceNorm, BatchNorm comparison

import { Matrix } from './matrix.js';

export function layerNorm(x, eps = 1e-6) {
  const result = new Matrix(x.rows, x.cols);
  for (let i = 0; i < x.rows; i++) {
    let mean = 0;
    for (let j = 0; j < x.cols; j++) mean += x.get(i, j);
    mean /= x.cols;
    let variance = 0;
    for (let j = 0; j < x.cols; j++) variance += (x.get(i, j) - mean) ** 2;
    variance /= x.cols;
    for (let j = 0; j < x.cols; j++) {
      result.set(i, j, (x.get(i, j) - mean) / Math.sqrt(variance + eps));
    }
  }
  return result;
}

export function rmsNorm(x, eps = 1e-6) {
  const result = new Matrix(x.rows, x.cols);
  for (let i = 0; i < x.rows; i++) {
    let sumSq = 0;
    for (let j = 0; j < x.cols; j++) sumSq += x.get(i, j) ** 2;
    const rms = Math.sqrt(sumSq / x.cols + eps);
    for (let j = 0; j < x.cols; j++) result.set(i, j, x.get(i, j) / rms);
  }
  return result;
}

export function instanceNorm(x, eps = 1e-6) {
  // Same as layerNorm for 2D (batch × features)
  return layerNorm(x, eps);
}

/**
 * Compare all normalizations on same input.
 */
export function compareNorms(x) {
  return {
    layerNorm: layerNorm(x),
    rmsNorm: rmsNorm(x),
    instanceNorm: instanceNorm(x),
  };
}
