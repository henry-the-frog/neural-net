// Regularization techniques
import { Matrix } from './matrix.js';

/**
 * L1 regularization (Lasso) — sum of absolute weights
 */
export function l1Regularization(weights, lambda = 0.01) {
  let penalty = 0;
  if (weights instanceof Matrix) {
    for (let i = 0; i < weights.data.length; i++) {
      penalty += Math.abs(weights.data[i]);
    }
  } else if (Array.isArray(weights)) {
    for (const w of weights) penalty += Math.abs(w);
  }
  return lambda * penalty;
}

/**
 * L2 regularization (Ridge) — sum of squared weights
 */
export function l2Regularization(weights, lambda = 0.01) {
  let penalty = 0;
  if (weights instanceof Matrix) {
    for (let i = 0; i < weights.data.length; i++) {
      penalty += weights.data[i] ** 2;
    }
  } else if (Array.isArray(weights)) {
    for (const w of weights) penalty += w ** 2;
  }
  return 0.5 * lambda * penalty;
}

/**
 * Elastic net — combination of L1 and L2
 */
export function elasticNet(weights, lambda = 0.01, l1Ratio = 0.5) {
  return l1Ratio * l1Regularization(weights, lambda) + (1 - l1Ratio) * l2Regularization(weights, lambda);
}

/**
 * Weight decay — direct weight scaling (differs from L2 in optimizer context)
 */
export function weightDecay(weights, decayRate = 0.01) {
  if (weights instanceof Matrix) {
    const result = new Matrix(weights.rows, weights.cols);
    for (let i = 0; i < weights.data.length; i++) {
      result.data[i] = weights.data[i] * (1 - decayRate);
    }
    return result;
  }
  return weights.map(w => w * (1 - decayRate));
}

/**
 * Max norm constraint — clip weights if norm exceeds threshold
 */
export function maxNormConstraint(weights, maxNorm = 1.0) {
  if (!(weights instanceof Matrix)) return weights;
  let norm = 0;
  for (let i = 0; i < weights.data.length; i++) norm += weights.data[i] ** 2;
  norm = Math.sqrt(norm);
  if (norm <= maxNorm) return weights;
  const scale = maxNorm / norm;
  const result = new Matrix(weights.rows, weights.cols);
  for (let i = 0; i < weights.data.length; i++) result.data[i] = weights.data[i] * scale;
  return result;
}

/**
 * Spectral norm — approximate largest singular value
 */
export function spectralNorm(weights, iterations = 1) {
  if (!(weights instanceof Matrix)) return 0;
  let u = new Matrix(weights.rows, 1);
  for (let i = 0; i < u.data.length; i++) u.data[i] = Math.random();
  
  for (let i = 0; i < iterations; i++) {
    let v = weights.transpose().multiply(u);
    let vNorm = 0;
    for (let j = 0; j < v.data.length; j++) vNorm += v.data[j] ** 2;
    vNorm = Math.sqrt(vNorm) || 1;
    for (let j = 0; j < v.data.length; j++) v.data[j] /= vNorm;
    
    u = weights.multiply(v);
    let uNorm = 0;
    for (let j = 0; j < u.data.length; j++) uNorm += u.data[j] ** 2;
    uNorm = Math.sqrt(uNorm) || 1;
    for (let j = 0; j < u.data.length; j++) u.data[j] /= uNorm;
  }
  
  // sigma = u^T W v
  const Wv = weights.multiply(u);
  let sigma = 0;
  const uT = weights.transpose().multiply(u);
  for (let i = 0; i < weights.data.length; i++) sigma += weights.data[i] ** 2;
  sigma = Math.sqrt(sigma / (weights.rows * weights.cols));
  
  // Simple approximation: Frobenius norm / sqrt(min(m,n))
  let frobenius = 0;
  for (let i = 0; i < weights.data.length; i++) frobenius += weights.data[i] ** 2;
  return Math.sqrt(frobenius) / Math.sqrt(Math.min(weights.rows, weights.cols));
}

/**
 * Gradient clipping — clip gradients by norm or value
 */
export function gradientClipping(gradients, maxNorm = 1.0, mode = 'norm') {
  if (!(gradients instanceof Matrix)) return gradients;
  
  if (mode === 'value') {
    const result = new Matrix(gradients.rows, gradients.cols);
    for (let i = 0; i < gradients.data.length; i++) {
      result.data[i] = Math.max(-maxNorm, Math.min(maxNorm, gradients.data[i]));
    }
    return result;
  }
  
  // Norm clipping
  let norm = 0;
  for (let i = 0; i < gradients.data.length; i++) norm += gradients.data[i] ** 2;
  norm = Math.sqrt(norm);
  
  if (norm <= maxNorm) return gradients;
  const scale = maxNorm / norm;
  const result = new Matrix(gradients.rows, gradients.cols);
  for (let i = 0; i < gradients.data.length; i++) result.data[i] = gradients.data[i] * scale;
  return result;
}
