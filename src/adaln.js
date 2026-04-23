// adaln.js — Adaptive Layer Normalization (DiT: Peebles & Xie, 2023)
// Used in Diffusion Transformers (DiT) for conditioning on timestep/class.
//
// Instead of fixed γ, β in LayerNorm, AdaLN predicts them from a conditioning signal:
//   γ, β = MLP(c)   where c = timestep embedding + class embedding
//   output = γ * LayerNorm(x) + β
//
// AdaLN-Zero: also predicts a scale factor α for the residual connection,
// initialized to zero so the model starts as identity.

import { Matrix } from './matrix.js';

/**
 * Standard Layer Normalization.
 */
export function layerNorm(x, eps = 1e-6) {
  const result = new Matrix(x.rows, x.cols);
  for (let i = 0; i < x.rows; i++) {
    let mean = 0;
    for (let j = 0; j < x.cols; j++) mean += x.get(i, j);
    mean /= x.cols;
    
    let variance = 0;
    for (let j = 0; j < x.cols; j++) variance += (x.get(i, j) - mean) ** 2;
    variance /= x.cols;
    
    const std = Math.sqrt(variance + eps);
    for (let j = 0; j < x.cols; j++) {
      result.set(i, j, (x.get(i, j) - mean) / std);
    }
  }
  return result;
}

export class AdaLN {
  /**
   * @param {number} dim - Feature dimension
   * @param {number} condDim - Conditioning dimension
   */
  constructor(dim, condDim) {
    this.dim = dim;
    this.condDim = condDim;
    
    // MLP to predict (γ, β) from conditioning
    // Output: 2 * dim (γ and β)
    this.Wc = Matrix.random(condDim, dim * 2).map(v => v * Math.sqrt(2.0 / condDim));
    this.bc = new Float64Array(dim * 2);
    // Initialize β to 0 and γ to 1
    for (let i = 0; i < dim; i++) this.bc[i] = 1; // γ init
    // β already 0
  }

  /**
   * @param {Matrix} x - Input (batch × dim)
   * @param {Float64Array} cond - Conditioning vector (condDim)
   * @returns {Matrix} Normalized and modulated output
   */
  forward(x, cond) {
    // Predict γ, β from conditioning
    const params = new Float64Array(this.dim * 2);
    for (let j = 0; j < this.dim * 2; j++) {
      let sum = this.bc[j];
      for (let i = 0; i < this.condDim; i++) {
        sum += cond[i] * this.Wc.get(i, j);
      }
      params[j] = sum;
    }
    
    const gamma = params.slice(0, this.dim);
    const beta = params.slice(this.dim);
    
    // Layer norm then modulate
    const normed = layerNorm(x);
    const output = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++) {
      for (let j = 0; j < x.cols; j++) {
        output.set(i, j, gamma[j] * normed.get(i, j) + beta[j]);
      }
    }
    
    return output;
  }
}

export class AdaLNZero {
  /**
   * AdaLN-Zero: predicts (γ, β, α) where α is initialized to 0.
   * α gates the residual connection, so the block starts as identity.
   * @param {number} dim - Feature dimension
   * @param {number} condDim - Conditioning dimension
   */
  constructor(dim, condDim) {
    this.dim = dim;
    this.condDim = condDim;
    
    // Predict (γ, β, α) — 3 * dim parameters
    this.Wc = Matrix.random(condDim, dim * 3).map(v => v * 0.001); // Small init for α → 0
    this.bc = new Float64Array(dim * 3);
    // γ init = 1
    for (let i = 0; i < dim; i++) this.bc[i] = 1;
    // β init = 0, α init = 0
  }

  /**
   * @param {Matrix} x - Input (batch × dim)
   * @param {Float64Array} cond - Conditioning vector
   * @returns {{ normed: Matrix, alpha: Float64Array }} Normalized output + residual gate
   */
  forward(x, cond) {
    const params = new Float64Array(this.dim * 3);
    for (let j = 0; j < this.dim * 3; j++) {
      let sum = this.bc[j];
      for (let i = 0; i < this.condDim; i++) {
        sum += cond[i] * this.Wc.get(i, j);
      }
      params[j] = sum;
    }
    
    const gamma = params.slice(0, this.dim);
    const beta = params.slice(this.dim, this.dim * 2);
    const alpha = params.slice(this.dim * 2);
    
    const normed = layerNorm(x);
    const output = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++) {
      for (let j = 0; j < x.cols; j++) {
        output.set(i, j, gamma[j] * normed.get(i, j) + beta[j]);
      }
    }
    
    return { normed: output, alpha };
  }

  /**
   * Apply residual with alpha gating: x + α * block_output
   */
  applyResidual(x, blockOutput, alpha) {
    const result = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++) {
      for (let j = 0; j < x.cols; j++) {
        result.set(i, j, x.get(i, j) + alpha[j] * blockOutput.get(i, j));
      }
    }
    return result;
  }
}
