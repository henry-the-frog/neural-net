// rope.js — Rotary Position Embedding (RoPE)
// Used in: Llama 2/3, Mistral, Gemma, CodeLlama, etc.
// Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
//
// RoPE encodes position by rotating pairs of Q/K dimensions by position-dependent angles.
// Benefits over sinusoidal/learned embeddings:
// - Relative position information encoded in attention dot product
// - Extrapolation to longer sequences than seen in training
// - No additional parameters needed

import { Matrix } from './matrix.js';

/**
 * Precompute RoPE rotation frequencies for a given max sequence length.
 *
 * For each position t and dimension pair i:
 *   θ_i = t * base^(-2i/d)
 * where base=10000 (default) and d=headDim.
 *
 * @param {number} headDim - dimension per head (must be even)
 * @param {number} maxSeqLen - maximum sequence length to precompute
 * @param {number} base - frequency base (default 10000)
 * @returns {{ cos: Float64Array[], sin: Float64Array[] }} - cos[pos][i], sin[pos][i]
 */
export function precomputeRoPE(headDim, maxSeqLen, base = 10000) {
  if (headDim % 2 !== 0) throw new Error('headDim must be even for RoPE');

  const halfDim = headDim / 2;
  const cos = new Array(maxSeqLen);
  const sin = new Array(maxSeqLen);

  // Precompute inverse frequencies: θ_i = base^(-2i/d) for i = 0..halfDim-1
  const invFreqs = new Float64Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    invFreqs[i] = 1.0 / Math.pow(base, (2 * i) / headDim);
  }

  for (let pos = 0; pos < maxSeqLen; pos++) {
    cos[pos] = new Float64Array(halfDim);
    sin[pos] = new Float64Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * invFreqs[i];
      cos[pos][i] = Math.cos(angle);
      sin[pos][i] = Math.sin(angle);
    }
  }

  return { cos, sin, halfDim };
}

/**
 * Apply RoPE rotation to a vector at a given position.
 * Rotates pairs of dimensions: (x[2i], x[2i+1]) → rotated by θ_i.
 *
 * [ cos(θ)  -sin(θ) ] [ x_even ]   [ x_even * cos(θ) - x_odd * sin(θ) ]
 * [ sin(θ)   cos(θ) ] [ x_odd  ] = [ x_even * sin(θ) + x_odd * cos(θ) ]
 *
 * @param {Float64Array|number[]} vec - vector of length headDim
 * @param {number} pos - position in sequence
 * @param {{ cos, sin, halfDim }} rope - precomputed RoPE tables
 * @returns {Float64Array} rotated vector
 */
export function applyRoPE(vec, pos, rope) {
  const { cos, sin, halfDim } = rope;
  const result = new Float64Array(vec.length);

  for (let i = 0; i < halfDim; i++) {
    const even = vec[2 * i];
    const odd = vec[2 * i + 1];
    result[2 * i] = even * cos[pos][i] - odd * sin[pos][i];
    result[2 * i + 1] = even * sin[pos][i] + odd * cos[pos][i];
  }

  return result;
}

/**
 * Apply RoPE to an entire sequence of Q or K vectors.
 *
 * @param {Matrix} mat - [seqLen, headDim] matrix
 * @param {{ cos, sin, halfDim }} rope - precomputed RoPE tables
 * @param {number} offset - position offset for KV-cache (cached tokens count)
 * @returns {Matrix} rotated matrix
 */
export function applyRoPEToSequence(mat, rope, offset = 0) {
  const result = new Matrix(mat.rows, mat.cols);

  for (let t = 0; t < mat.rows; t++) {
    const pos = t + offset;
    for (let i = 0; i < rope.halfDim; i++) {
      const even = mat.get(t, 2 * i);
      const odd = mat.get(t, 2 * i + 1);
      result.set(t, 2 * i, even * rope.cos[pos][i] - odd * rope.sin[pos][i]);
      result.set(t, 2 * i + 1, even * rope.sin[pos][i] + odd * rope.cos[pos][i]);
    }
  }

  return result;
}

/**
 * Key property of RoPE: dot product of rotated vectors encodes relative position.
 *
 * <RoPE(q, m), RoPE(k, n)> depends only on (m - n), not on absolute positions.
 * This means the attention score between positions m and n is determined by
 * their relative distance, enabling better length generalization.
 */
export function demonstrateRelativeProperty(headDim, pos1, pos2, base = 10000) {
  const rope = precomputeRoPE(headDim, Math.max(pos1, pos2) + 1, base);
  const q = new Float64Array(headDim);
  const k = new Float64Array(headDim);
  for (let i = 0; i < headDim; i++) {
    q[i] = Math.random();
    k[i] = Math.random();
  }

  const rq = applyRoPE(q, pos1, rope);
  const rk = applyRoPE(k, pos2, rope);

  let dot = 0;
  for (let i = 0; i < headDim; i++) dot += rq[i] * rk[i];

  return dot;
}
