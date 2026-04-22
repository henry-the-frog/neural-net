// rope.js — Rotary Position Embeddings (Su et al., 2021)
//
// Used in GPT-NeoX, Llama, Mistral, and most modern LLMs.
// Key idea: encode position by rotating Q and K vectors in the complex plane.
// The attention score Q_m · K_n depends only on the relative position (m-n),
// not absolute positions.
//
// For each pair of dimensions (2i, 2i+1):
//   θ_i = 1 / base^(2i/d)
//   q'[2i]   = q[2i] * cos(m * θ_i) - q[2i+1] * sin(m * θ_i)
//   q'[2i+1] = q[2i] * sin(m * θ_i) + q[2i+1] * cos(m * θ_i)
// where m is the position index.

import { Matrix } from './matrix.js';

/**
 * Precompute rotation frequencies for given dimensions and max sequence length.
 * @param {number} dim — head dimension (must be even)
 * @param {number} maxSeqLen — maximum sequence length
 * @param {number} [base=10000] — frequency base
 * @returns {{ cos: Float64Array[], sin: Float64Array[] }}
 */
export function precomputeFreqs(dim, maxSeqLen, base = 10000) {
  if (dim % 2 !== 0) throw new Error('RoPE dimension must be even');
  
  const halfDim = dim / 2;
  const freqs = new Float64Array(halfDim);
  
  for (let i = 0; i < halfDim; i++) {
    freqs[i] = 1.0 / Math.pow(base, (2 * i) / dim);
  }
  
  const cosTable = [];
  const sinTable = [];
  
  for (let pos = 0; pos < maxSeqLen; pos++) {
    const cosRow = new Float64Array(halfDim);
    const sinRow = new Float64Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * freqs[i];
      cosRow[i] = Math.cos(angle);
      sinRow[i] = Math.sin(angle);
    }
    cosTable.push(cosRow);
    sinTable.push(sinRow);
  }
  
  return { cos: cosTable, sin: sinTable, freqs };
}

/**
 * Apply RoPE to a matrix of Q or K vectors.
 * @param {Matrix} x — [seqLen, dim] matrix of Q or K vectors
 * @param {{ cos: Float64Array[], sin: Float64Array[] }} freqTable — precomputed frequencies
 * @param {number} [offset=0] — position offset (for KV cache continuation)
 * @returns {Matrix} — rotated [seqLen, dim]
 */
export function applyRoPE(x, freqTable, offset = 0) {
  const seqLen = x.rows;
  const dim = x.cols;
  const halfDim = dim / 2;
  const result = new Matrix(seqLen, dim);
  
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = pos + offset;
    const cosRow = freqTable.cos[absPos];
    const sinRow = freqTable.sin[absPos];
    
    for (let i = 0; i < halfDim; i++) {
      const x0 = x.get(pos, 2 * i);
      const x1 = x.get(pos, 2 * i + 1);
      
      // Complex rotation: (x0 + ix1) * (cos + i*sin)
      result.set(pos, 2 * i,     x0 * cosRow[i] - x1 * sinRow[i]);
      result.set(pos, 2 * i + 1, x0 * sinRow[i] + x1 * cosRow[i]);
    }
  }
  
  return result;
}

/**
 * Apply inverse RoPE (for backward pass).
 * Since rotation by θ is undone by rotation by -θ:
 * just negate the sin terms.
 */
export function applyInverseRoPE(x, freqTable, offset = 0) {
  const seqLen = x.rows;
  const dim = x.cols;
  const halfDim = dim / 2;
  const result = new Matrix(seqLen, dim);
  
  for (let pos = 0; pos < seqLen; pos++) {
    const absPos = pos + offset;
    const cosRow = freqTable.cos[absPos];
    const sinRow = freqTable.sin[absPos];
    
    for (let i = 0; i < halfDim; i++) {
      const x0 = x.get(pos, 2 * i);
      const x1 = x.get(pos, 2 * i + 1);
      
      // Inverse rotation: (x0 + ix1) * (cos - i*sin)
      result.set(pos, 2 * i,     x0 * cosRow[i] + x1 * sinRow[i]);
      result.set(pos, 2 * i + 1, -x0 * sinRow[i] + x1 * cosRow[i]);
    }
  }
  
  return result;
}

/**
 * Key property of RoPE: the dot product between rotated Q at position m
 * and rotated K at position n depends only on (m-n).
 * 
 * This function verifies this property for a given pair of vectors.
 */
export function verifyRelativeProperty(q, k, freqTable, posM, posN) {
  const qRot = applyRoPE(q, freqTable, posM);
  const kRot = applyRoPE(k, freqTable, posN);
  
  // Dot product
  let dot = 0;
  for (let i = 0; i < qRot.cols; i++) {
    dot += qRot.get(0, i) * kRot.get(0, i);
  }
  return dot;
}

// Aliases for backward compatibility with gqa-attention.js
export { precomputeFreqs as precomputeRoPE };
export { applyRoPE as applyRoPEToSequence };
