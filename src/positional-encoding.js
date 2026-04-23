// positional-encoding.js — Positional Encodings
// Multiple approaches: sinusoidal (Vaswani 2017), learned, ALiBi (Press 2022)

import { Matrix } from './matrix.js';

/**
 * Sinusoidal positional encoding (Vaswani et al., 2017).
 * PE(pos, 2i) = sin(pos / 10000^(2i/dModel))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
 * @param {number} maxLen - Maximum sequence length
 * @param {number} dModel - Model dimension
 * @returns {Matrix} Positional encoding (maxLen × dModel)
 */
export function sinusoidalEncoding(maxLen, dModel) {
  const pe = new Matrix(maxLen, dModel);
  
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dModel; i += 2) {
      const angle = pos / Math.pow(10000, i / dModel);
      pe.set(pos, i, Math.sin(angle));
      if (i + 1 < dModel) {
        pe.set(pos, i + 1, Math.cos(angle));
      }
    }
  }
  
  return pe;
}

/**
 * Timestep embedding for diffusion models.
 * Same formula as sinusoidal PE but for a single timestep → vector.
 * @param {number} timestep - Diffusion timestep
 * @param {number} dim - Embedding dimension
 * @returns {Float64Array} Timestep embedding
 */
export function timestepEmbedding(timestep, dim) {
  const emb = new Float64Array(dim);
  const halfDim = Math.floor(dim / 2);
  
  for (let i = 0; i < halfDim; i++) {
    const angle = timestep / Math.pow(10000, i / halfDim);
    emb[i] = Math.sin(angle);
    emb[i + halfDim] = Math.cos(angle);
  }
  
  return emb;
}

/**
 * ALiBi (Attention with Linear Biases, Press et al., 2022).
 * No positional embeddings at all! Instead, add a linear bias to attention scores:
 * score(q, k) = q^T k - m * |i - j|
 * where m is a head-specific slope.
 * 
 * Key advantage: extrapolates to longer sequences without training.
 * 
 * @param {number} seqLen - Sequence length
 * @param {number} nHeads - Number of attention heads
 * @returns {Array<Matrix>} Per-head bias matrices (seqLen × seqLen)
 */
export function alibiSlopes(nHeads) {
  // Slopes: geometric sequence from 2^(-8/n) to 2^(-8)
  const slopes = new Float64Array(nHeads);
  const ratio = Math.pow(2, -8.0 / nHeads);
  let slope = ratio;
  for (let h = 0; h < nHeads; h++) {
    slopes[h] = slope;
    slope *= ratio;
  }
  return slopes;
}

/**
 * Compute ALiBi bias matrix for a given head slope.
 * @param {number} seqLen - Sequence length
 * @param {number} slope - Head-specific slope
 * @returns {Matrix} Bias matrix (seqLen × seqLen), causal mask included
 */
export function alibiBiasMatrix(seqLen, slope) {
  const bias = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      if (j > i) {
        // Causal mask: future tokens get -inf
        bias.set(i, j, -1e9);
      } else {
        // Linear distance bias
        bias.set(i, j, -slope * (i - j));
      }
    }
  }
  return bias;
}

/**
 * Learned positional embedding.
 * @param {number} maxLen - Maximum sequence length
 * @param {number} dModel - Model dimension
 * @returns {Matrix} Random initialized embedding (to be trained)
 */
export function learnedPositionalEmbedding(maxLen, dModel) {
  return Matrix.random(maxLen, dModel).map(v => v * 0.02);
}
