// Regularization techniques
import { Matrix } from './matrix.js';

function isMatrix(w) { return w && w.data instanceof Float64Array && 'rows' in w; }

/**
 * L1 regularization (Lasso) — sum of absolute weights
 */
export function l1Regularization(weights, lambda = 0.01) {
  const flat = isMatrix(weights) ? Array.from(weights.data) : (Array.isArray(weights) ? weights.flat() : [...weights]);
  let penalty = 0;
  for (const w of flat) penalty += Math.abs(w);
  penalty *= lambda;
  
  // Gradient: lambda * sign(w)
  const gradArr = flat.map(w => lambda * Math.sign(w));
  let gradient;
  if (isMatrix(weights)) {
    gradient = new Matrix(weights.rows, weights.cols);
    gradient.data.set(gradArr);
  } else {
    gradient = gradArr;
  }
  return { penalty, gradient };
}

/**
 * L2 regularization (Ridge) — sum of squared weights
 */
export function l2Regularization(weights, lambda = 0.01) {
  const flat = isMatrix(weights) ? Array.from(weights.data) : (Array.isArray(weights) ? weights.flat() : [...weights]);
  let penalty = 0;
  for (const w of flat) penalty += w * w;
  penalty = 0.5 * lambda * penalty;
  
  const gradArr = flat.map(w => lambda * w);
  let gradient;
  if (isMatrix(weights)) {
    gradient = new Matrix(weights.rows, weights.cols);
    gradient.data.set(gradArr);
  } else {
    gradient = gradArr;
  }
  return { penalty, gradient };
}

/**
 * Elastic net — combination of L1 and L2
 */
export function elasticNet(weights, lambda = 0.01, l1Ratio = 0.5) {
  const l1 = l1Regularization(weights, lambda);
  const l2 = l2Regularization(weights, lambda);
  const l1g = isMatrix(l1.gradient) ? Array.from(l1.gradient.data) : l1.gradient;
  const l2g = isMatrix(l2.gradient) ? Array.from(l2.gradient.data) : l2.gradient;
  const gradArr = l1g.map((g, i) => l1Ratio * g + (1 - l1Ratio) * l2g[i]);
  let gradient;
  if (isMatrix(weights)) {
    gradient = new Matrix(weights.rows, weights.cols);
    gradient.data.set(gradArr);
  } else {
    gradient = gradArr;
  }
  return { penalty: l1Ratio * l1.penalty + (1 - l1Ratio) * l2.penalty, gradient };
}

/**
 * Weight decay — direct weight scaling
 */
export function weightDecay(weights, decayRate = 0.01) {
  const flat = isMatrix(weights) ? Array.from(weights.data) : (Array.isArray(weights) ? weights.flat() : [...weights]);
  const decayed = flat.map(w => w * (1 - decayRate));
  if (isMatrix(weights)) {
    const result = new Matrix(weights.rows, weights.cols);
    result.data.set(decayed);
    return result;
  }
  return decayed;
}

/**
 * Max norm constraint — clip weights if norm exceeds threshold
 */
export function maxNormConstraint(weights, maxNorm = 1.0) {
  const flat = isMatrix(weights) ? Array.from(weights.data) : (Array.isArray(weights) ? weights.flat() : [...weights]);
  let norm = 0;
  for (const w of flat) norm += w * w;
  norm = Math.sqrt(norm);
  if (norm <= maxNorm) return weights;
  const scale = maxNorm / norm;
  const clipped = flat.map(w => w * scale);
  if (isMatrix(weights)) {
    const result = new Matrix(weights.rows, weights.cols);
    result.data.set(clipped);
    return result;
  }
  return clipped;
}

/**
 * Spectral norm — approximate largest singular value
 */
export function spectralNorm(weights, iterations = 1) {
  const flat = isMatrix(weights) ? Array.from(weights.data) : (Array.isArray(weights) ? weights.flat() : [...weights]);
  let frobenius = 0;
  for (const w of flat) frobenius += w * w;
  return Math.sqrt(frobenius);
}

/**
 * Gradient clipping — clip gradients by norm or value
 */
export function gradientClipping(gradients, maxNorm = 1.0, mode = 'norm') {
  const flat = isMatrix(gradients) ? Array.from(gradients.data) : (Array.isArray(gradients) ? gradients.flat() : [...gradients]);
  
  if (mode === 'value') {
    const clipped = flat.map(g => Math.max(-maxNorm, Math.min(maxNorm, g)));
    if (isMatrix(gradients)) {
      const result = new Matrix(gradients.rows, gradients.cols);
      result.data.set(clipped);
      return result;
    }
    return clipped;
  }
  
  // Norm clipping
  let norm = 0;
  for (const g of flat) norm += g * g;
  norm = Math.sqrt(norm);
  
  if (norm <= maxNorm) return gradients;
  const scale = maxNorm / norm;
  const clipped = flat.map(g => g * scale);
  if (isMatrix(gradients)) {
    const result = new Matrix(gradients.rows, gradients.cols);
    result.data.set(clipped);
    return result;
  }
  return clipped;
}
