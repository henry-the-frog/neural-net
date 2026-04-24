// rope.js — Rotary Position Embedding (RoPE)
// Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
//
// Key idea: encode position by rotating query/key vectors in 2D pairs.
// For positions m,n: dot(RoPE(q,m), RoPE(k,n)) depends only on q·k and (m-n).
// This gives relative position sensitivity without explicit position embeddings.

import { Matrix } from './matrix.js';

/**
 * Precompute frequency basis for RoPE.
 * θ_i = 1 / (10000^(2i/d)) for i in [0, d/2)
 * @param {number} dim - Embedding dimension (must be even)
 * @param {number} maxLen - Maximum sequence length
 * @param {number} base - Base for frequency computation (default 10000)
 * @returns {{ cos: Matrix, sin: Matrix }} Precomputed cos/sin tables (maxLen × dim/2)
 */
export function precomputeFreqs(dim, maxLen, base = 10000) {
  if (dim % 2 !== 0) throw new Error('RoPE requires even embedding dimension');
  const halfDim = dim / 2;
  
  const cos = new Matrix(maxLen, halfDim);
  const sin = new Matrix(maxLen, halfDim);
  
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const theta = pos / Math.pow(base, (2 * i) / dim);
      cos.set(pos, i, Math.cos(theta));
      sin.set(pos, i, Math.sin(theta));
    }
  }
  
  return { cos, sin };
}

/**
 * Apply rotary position embedding to query or key vectors.
 * For each pair (x_{2i}, x_{2i+1}), rotate by angle θ_i * position:
 *   x'_{2i}   = x_{2i} * cos(θ) - x_{2i+1} * sin(θ)
 *   x'_{2i+1} = x_{2i} * sin(θ) + x_{2i+1} * cos(θ)
 * 
 * @param {Matrix} x - Input vectors (seqLen × dim)
 * @param {Matrix} freqCos - Cosine table from precomputeFreqs (seqLen × dim/2)
 * @param {Matrix} freqSin - Sine table from precomputeFreqs (seqLen × dim/2)
 * @param {number} offset - Position offset (for KV cache sliding window)
 * @returns {Matrix} Rotated vectors (seqLen × dim)
 */
export function applyRoPE(x, freqCos, freqSin, offset = 0) {
  const seqLen = x.rows;
  const dim = x.cols;
  const halfDim = dim / 2;
  const result = new Matrix(seqLen, dim);
  
  for (let pos = 0; pos < seqLen; pos++) {
    const freqPos = pos + offset;
    for (let i = 0; i < halfDim; i++) {
      const x0 = x.get(pos, 2 * i);
      const x1 = x.get(pos, 2 * i + 1);
      const c = freqCos.get(freqPos, i);
      const s = freqSin.get(freqPos, i);
      
      result.set(pos, 2 * i,     x0 * c - x1 * s);
      result.set(pos, 2 * i + 1, x0 * s + x1 * c);
    }
  }
  
  return result;
}

/**
 * Backward pass for RoPE: compute gradient of loss w.r.t. input x.
 * Since RoPE is a rotation (orthogonal), the backward is the inverse rotation.
 * 
 * @param {Matrix} dOutput - Gradient of loss w.r.t. rotated output
 * @param {Matrix} freqCos - Cosine table
 * @param {Matrix} freqSin - Sine table
 * @param {number} offset - Position offset
 * @returns {Matrix} Gradient w.r.t. input x
 */
export function applyRoPEBackward(dOutput, freqCos, freqSin, offset = 0) {
  const seqLen = dOutput.rows;
  const dim = dOutput.cols;
  const halfDim = dim / 2;
  const dInput = new Matrix(seqLen, dim);
  
  for (let pos = 0; pos < seqLen; pos++) {
    const freqPos = pos + offset;
    for (let i = 0; i < halfDim; i++) {
      const dy0 = dOutput.get(pos, 2 * i);
      const dy1 = dOutput.get(pos, 2 * i + 1);
      const c = freqCos.get(freqPos, i);
      const s = freqSin.get(freqPos, i);
      
      // Inverse rotation: R^(-1) = R^T (since R is orthogonal)
      // [cos θ, sin θ] [dy0]
      // [-sin θ, cos θ] [dy1]
      dInput.set(pos, 2 * i,     dy0 * c + dy1 * s);
      dInput.set(pos, 2 * i + 1, -dy0 * s + dy1 * c);
    }
  }
  
  return dInput;
}

/**
 * RoPE-enhanced attention: compute attention scores with rotary embeddings.
 * @param {Matrix} Q - Queries (seqLen × headDim)
 * @param {Matrix} K - Keys (seqLen × headDim)
 * @param {object} freqs - { cos, sin } from precomputeFreqs
 * @returns {{ Q_rot: Matrix, K_rot: Matrix }} Rotated Q and K
 */
/**
 * Alias for precomputeFreqs — used by gqa-attention.js
 */
export const precomputeRoPE = precomputeFreqs;

/**
 * Apply RoPE to a sequence given precomputed freqs object.
 * Wrapper for applyRoPE that unpacks {cos, sin} from freqs.
 * @param {Matrix} x - Input vectors (seqLen × dim)
 * @param {{ cos: Matrix, sin: Matrix }} freqs - Precomputed frequencies
 * @param {number} offset - Position offset
 * @returns {Matrix} Rotated vectors
 */
export function applyRoPEToSequence(x, freqs, offset = 0) {
  return applyRoPE(x, freqs.cos, freqs.sin, offset);
}

export function ropeAttention(Q, K, freqs) {
  const Q_rot = applyRoPE(Q, freqs.cos, freqs.sin);
  const K_rot = applyRoPE(K, freqs.cos, freqs.sin);
  return { Q_rot, K_rot };
}
