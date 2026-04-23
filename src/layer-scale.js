// layer-scale.js — Layer Scale (Touvron 2021) + Stochastic Depth (Huang 2016)
// Used in CaiT, DeiT-III, modern ViTs for stable deep training.

import { Matrix } from './matrix.js';

/**
 * Layer Scale: per-channel learnable scaling initialized to small value.
 * output = x + diag(λ) * f(x)
 * λ starts at initValue (e.g., 1e-6) and is learned.
 */
export class LayerScale {
  /**
   * @param {number} dim - Feature dimension
   * @param {number} initValue - Initial scale (typically 1e-4 to 1e-6)
   */
  constructor(dim, initValue = 1e-4) {
    this.dim = dim;
    this.gamma = new Float64Array(dim).fill(initValue);
  }

  /**
   * Apply layer scale to residual output.
   * @param {Matrix} residual - Block output f(x) (batch × dim)
   * @returns {Matrix} Scaled output
   */
  forward(residual) {
    const result = new Matrix(residual.rows, residual.cols);
    for (let i = 0; i < residual.rows; i++) {
      for (let j = 0; j < residual.cols; j++) {
        result.set(i, j, residual.get(i, j) * this.gamma[j]);
      }
    }
    return result;
  }
}

/**
 * Stochastic Depth (DropPath): randomly drop entire layers during training.
 * Each layer has a survival probability p.
 * During training: output = x + (1/p) * f(x) if surviving, else output = x
 * During inference: output = x + f(x) (no dropout)
 */
export class StochasticDepth {
  /**
   * @param {number} dropRate - Probability of dropping the layer (0-1)
   */
  constructor(dropRate = 0.1) {
    this.dropRate = dropRate;
    this.training = true;
  }

  /**
   * Apply stochastic depth to residual.
   * @param {Matrix} x - Input (identity path)
   * @param {Matrix} residual - Block output (to be dropped)
   * @returns {Matrix} Output
   */
  forward(x, residual) {
    const result = new Matrix(x.rows, x.cols);
    
    if (this.training && Math.random() < this.dropRate) {
      // Drop: return identity
      for (let i = 0; i < x.data.length; i++) result.data[i] = x.data[i];
    } else {
      // Keep: add residual (scaled by 1/(1-dropRate) during training)
      const scale = this.training ? 1 / (1 - this.dropRate) : 1;
      for (let i = 0; i < x.data.length; i++) {
        result.data[i] = x.data[i] + scale * residual.data[i];
      }
    }
    
    return result;
  }
}

/**
 * Linear dropRate schedule for stochastic depth.
 * Deeper layers have higher drop rates.
 * @param {number} nLayers - Total layers
 * @param {number} maxDropRate - Drop rate for the last layer
 * @returns {Array<number>} Per-layer drop rates
 */
export function linearDropSchedule(nLayers, maxDropRate = 0.1) {
  return Array.from({ length: nLayers }, (_, i) => maxDropRate * i / (nLayers - 1));
}
