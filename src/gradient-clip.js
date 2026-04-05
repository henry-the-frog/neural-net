// gradient-clip.js — Gradient clipping utilities
//
// Prevents exploding gradients during training:
//   - clipByValue: clamp each gradient element to [-maxVal, maxVal]
//   - clipByNorm: scale gradient if its L2 norm exceeds maxNorm
//   - clipByGlobalNorm: scale all gradients together if combined norm exceeds threshold

import { Matrix } from './matrix.js';

/**
 * Clip gradient values to [-maxVal, maxVal]
 */
export function clipByValue(grad, maxVal) {
  const result = new Matrix(grad.rows, grad.cols);
  for (let i = 0; i < grad.data.length; i++) {
    result.data[i] = Math.max(-maxVal, Math.min(maxVal, grad.data[i]));
  }
  return result;
}

/**
 * Clip gradient by L2 norm: if ||grad|| > maxNorm, scale it down
 */
export function clipByNorm(grad, maxNorm) {
  const norm = l2Norm(grad);
  if (norm <= maxNorm) return grad;
  
  const scale = maxNorm / norm;
  const result = new Matrix(grad.rows, grad.cols);
  for (let i = 0; i < grad.data.length; i++) {
    result.data[i] = grad.data[i] * scale;
  }
  return result;
}

/**
 * Clip multiple gradients by global norm.
 * If the combined norm of all gradients exceeds maxNorm, scale all of them proportionally.
 */
export function clipByGlobalNorm(grads, maxNorm) {
  // Compute global norm
  let globalNormSq = 0;
  for (const grad of grads) {
    for (let i = 0; i < grad.data.length; i++) {
      globalNormSq += grad.data[i] * grad.data[i];
    }
  }
  const globalNorm = Math.sqrt(globalNormSq);
  
  if (globalNorm <= maxNorm) return { grads, globalNorm, clipped: false };
  
  const scale = maxNorm / globalNorm;
  const clippedGrads = grads.map(grad => {
    const result = new Matrix(grad.rows, grad.cols);
    for (let i = 0; i < grad.data.length; i++) {
      result.data[i] = grad.data[i] * scale;
    }
    return result;
  });
  
  return { grads: clippedGrads, globalNorm, clipped: true };
}

/**
 * Compute L2 norm of a matrix
 */
export function l2Norm(matrix) {
  let sum = 0;
  for (let i = 0; i < matrix.data.length; i++) {
    sum += matrix.data[i] * matrix.data[i];
  }
  return Math.sqrt(sum);
}

/**
 * Compute gradient statistics (for monitoring)
 */
export function gradientStats(grad) {
  let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
  const n = grad.data.length;
  for (let i = 0; i < n; i++) {
    const v = grad.data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  return { min, max, mean, variance, std: Math.sqrt(Math.max(0, variance)), norm: l2Norm(grad) };
}
