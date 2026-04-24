// regularization.js — Advanced Regularization Techniques
// Gradient Penalty (WGAN-GP, Gulrajani et al., 2017)
// Spectral Normalization (Miyato et al., 2018)
// Weight Decay, L1/L2 regularization

import { Matrix } from './matrix.js';

/**
 * Gradient penalty for WGAN-GP.
 * Enforces Lipschitz constraint by penalizing gradient norm ≠ 1.
 * GP = E[(||∇D(x̂)||₂ - 1)²] where x̂ = εx_real + (1-ε)x_fake
 * 
 * @param {function} discriminator - D(x) → scalar
 * @param {Float64Array} real - Real sample
 * @param {Float64Array} fake - Fake sample
 * @param {number} lambda - Gradient penalty weight (default 10)
 * @returns {{ penalty: number, interpolated: Float64Array }}
 */
export function gradientPenalty(discriminator, real, fake, lambda = 10) {
  const n = real.length;
  const epsilon = Math.random();
  
  // Interpolate between real and fake
  const interpolated = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    interpolated[i] = epsilon * real[i] + (1 - epsilon) * fake[i];
  }
  
  // Estimate gradient norm via finite differences
  const delta = 1e-5;
  const gradNorm = estimateGradientNorm(discriminator, interpolated, delta);
  
  // Penalty: (||grad||₂ - 1)²
  const penalty = lambda * (gradNorm - 1) ** 2;
  
  return { penalty, interpolated, gradNorm };
}

function estimateGradientNorm(fn, x, delta) {
  const n = x.length;
  let sumSq = 0;
  const baseVal = fn(x);
  
  for (let i = 0; i < n; i++) {
    const xPlus = new Float64Array(x);
    xPlus[i] += delta;
    const grad_i = (fn(xPlus) - baseVal) / delta;
    sumSq += grad_i * grad_i;
  }
  
  return Math.sqrt(sumSq);
}

/**
 * Spectral normalization: normalize weight matrix by its largest singular value.
 * Approximated via power iteration.
 * @param {Matrix} W - Weight matrix
 * @param {number} nIter - Number of power iterations (default 1)
 * @returns {{ normalized: Matrix, sigma: number }}
 */
export function spectralNormalization(W, nIter = 1) {
  const rows = W.rows;
  const cols = W.cols;
  
  // Initialize u (left singular vector estimate)
  let u = new Float64Array(rows);
  for (let i = 0; i < rows; i++) u[i] = Math.random() - 0.5;
  normalize(u);
  
  let v = new Float64Array(cols);
  
  for (let iter = 0; iter < nIter; iter++) {
    // v = W^T u / ||W^T u||
    v = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) v[j] += W.get(i, j) * u[i];
    }
    normalize(v);
    
    // u = W v / ||W v||
    u = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) u[i] += W.get(i, j) * v[j];
    }
    normalize(u);
  }
  
  // Spectral norm: σ = u^T W v
  let sigma = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      sigma += u[i] * W.get(i, j) * v[j];
    }
  }
  
  // Normalize W by sigma
  const normalized = new Matrix(rows, cols);
  for (let i = 0; i < W.data.length; i++) {
    normalized.data[i] = W.data[i] / sigma;
  }
  
  return { normalized, sigma };
}

function normalize(v) {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm + 1e-10);
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

/**
 * L1 regularization (Lasso): promotes sparsity.
 */
export function l1Penalty(weights, lambda = 0.01) {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) sum += Math.abs(weights[i]);
  return lambda * sum;
}

/**
 * L2 regularization (Ridge / weight decay).
 */
export function l2Penalty(weights, lambda = 0.01) {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) sum += weights[i] * weights[i];
  return 0.5 * lambda * sum;
}

/**
 * Elastic net: combination of L1 and L2.
 */
export function elasticNetPenalty(weights, lambda = 0.01, ratio = 0.5) {
  return ratio * l1Penalty(weights, lambda) + (1 - ratio) * l2Penalty(weights, lambda);
}

// === Matrix-based regularization API (returns { penalty, gradient }) ===

/**
 * L1 regularization on Matrix weights.
 * @param {Matrix} weights
 * @param {number} lambda
 * @returns {{ penalty: number, gradient: Matrix }}
 */
export function l1Regularization(weights, lambda = 0.01) {
  let penalty = 0;
  const gradient = new Matrix(weights.rows, weights.cols);
  for (let i = 0; i < weights.data.length; i++) {
    penalty += Math.abs(weights.data[i]);
    gradient.data[i] = lambda * Math.sign(weights.data[i]);
  }
  penalty *= lambda;
  return { penalty, gradient };
}

/**
 * L2 regularization on Matrix weights.
 */
export function l2Regularization(weights, lambda = 0.01) {
  let penalty = 0;
  const gradient = new Matrix(weights.rows, weights.cols);
  for (let i = 0; i < weights.data.length; i++) {
    penalty += weights.data[i] * weights.data[i];
    gradient.data[i] = lambda * weights.data[i];
  }
  penalty *= 0.5 * lambda;
  return { penalty, gradient };
}

/**
 * Elastic net regularization on Matrix weights.
 */
export function elasticNet(weights, lambda = 0.01, ratio = 0.5) {
  const l1 = l1Regularization(weights, lambda);
  const l2 = l2Regularization(weights, lambda);
  const gradient = new Matrix(weights.rows, weights.cols);
  for (let i = 0; i < weights.data.length; i++) {
    gradient.data[i] = ratio * l1.gradient.data[i] + (1 - ratio) * l2.gradient.data[i];
  }
  return { penalty: ratio * l1.penalty + (1 - ratio) * l2.penalty, gradient };
}

/**
 * Weight decay: multiply weights by (1 - decay).
 */
export function weightDecay(weights, decay = 0.01) {
  const result = new Matrix(weights.rows, weights.cols);
  for (let i = 0; i < weights.data.length; i++) {
    result.data[i] = weights.data[i] * (1 - decay);
  }
  return result;
}

/**
 * Max-norm constraint: clip weight vectors to max norm.
 */
export function maxNormConstraint(weights, maxNorm = 3.0) {
  const result = new Matrix(weights.rows, weights.cols);
  for (let r = 0; r < weights.rows; r++) {
    let norm = 0;
    for (let c = 0; c < weights.cols; c++) {
      norm += weights.get(r, c) ** 2;
    }
    norm = Math.sqrt(norm);
    const scale = norm > maxNorm ? maxNorm / norm : 1;
    for (let c = 0; c < weights.cols; c++) {
      result.set(r, c, weights.get(r, c) * scale);
    }
  }
  return result;
}

/**
 * Spectral norm alias.
 */
export const spectralNorm = spectralNormalization;

/**
 * Gradient clipping by global norm.
 */
export function gradientClipping(gradient, maxNorm = 1.0) {
  let norm = 0;
  for (let i = 0; i < gradient.data.length; i++) {
    norm += gradient.data[i] ** 2;
  }
  norm = Math.sqrt(norm);
  if (norm <= maxNorm) {
    const result = new Matrix(gradient.rows, gradient.cols);
    for (let i = 0; i < gradient.data.length; i++) result.data[i] = gradient.data[i];
    return result;
  }
  const scale = maxNorm / norm;
  const result = new Matrix(gradient.rows, gradient.cols);
  for (let i = 0; i < gradient.data.length; i++) {
    result.data[i] = gradient.data[i] * scale;
  }
  return result;
}
