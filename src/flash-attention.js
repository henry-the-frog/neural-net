// flash-attention.js — Flash Attention pattern (Dao et al., 2022)
// Fused softmax+matmul: computes attention without materializing full N×N matrix.
//
// In pure JS, we implement the "online softmax" trick:
// Process K/V in blocks, accumulating numerator and denominator incrementally.
// Memory: O(N*d) instead of O(N²) for the attention matrix.
//
// For small N (< 512), standard attention may be faster due to cache effects.

import { Matrix } from './matrix.js';

/**
 * Standard attention: Q @ K^T / sqrt(d) → softmax → @ V
 * Memory: O(N²) for attention matrix
 * @param {Matrix} Q - Queries (N × d)
 * @param {Matrix} K - Keys (N × d)
 * @param {Matrix} V - Values (N × d)
 * @param {boolean} causal - Apply causal mask
 * @returns {Matrix} Output (N × d)
 */
export function standardAttention(Q, K, V, causal = false) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1.0 / Math.sqrt(d);
  
  // Compute attention scores: Q @ K^T * scale
  const scores = new Matrix(N, N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (causal && j > i) {
        scores.set(i, j, -Infinity);
        continue;
      }
      let dot = 0;
      for (let k = 0; k < d; k++) {
        dot += Q.get(i, k) * K.get(j, k);
      }
      scores.set(i, j, dot * scale);
    }
  }
  
  // Softmax per row
  const attn = new Matrix(N, N);
  for (let i = 0; i < N; i++) {
    let max = -Infinity;
    for (let j = 0; j < N; j++) max = Math.max(max, scores.get(i, j));
    let sum = 0;
    for (let j = 0; j < N; j++) {
      const exp = Math.exp(scores.get(i, j) - max);
      attn.set(i, j, exp);
      sum += exp;
    }
    for (let j = 0; j < N; j++) attn.set(i, j, attn.get(i, j) / sum);
  }
  
  // Output: attn @ V
  const output = new Matrix(N, d);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < d; k++) {
      let sum = 0;
      for (let j = 0; j < N; j++) {
        sum += attn.get(i, j) * V.get(j, k);
      }
      output.set(i, k, sum);
    }
  }
  
  return output;
}

/**
 * Flash Attention: tiled online softmax attention.
 * Never materializes the full N×N attention matrix.
 * Memory: O(N*d + blockSize²) instead of O(N²)
 * 
 * Algorithm:
 * For each query row i:
 *   Initialize: max_i = -inf, sum_i = 0, output_i = 0
 *   For each block of K/V rows:
 *     Compute partial scores for this block
 *     Update max, sum, and output using online softmax trick
 * 
 * @param {Matrix} Q - Queries (N × d)
 * @param {Matrix} K - Keys (N × d)
 * @param {Matrix} V - Values (N × d)
 * @param {number} blockSize - Block size for tiling (default: 32)
 * @param {boolean} causal - Apply causal mask
 * @returns {Matrix} Output (N × d)
 */
export function flashAttention(Q, K, V, blockSize = 32, causal = false) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1.0 / Math.sqrt(d);
  
  // Output accumulators (per query row)
  const output = new Matrix(N, d); // Running weighted sum
  const maxScores = new Float64Array(N).fill(-Infinity); // Running max
  const sumExp = new Float64Array(N).fill(0); // Running sum of exp
  
  // Process K/V in blocks
  for (let jStart = 0; jStart < N; jStart += blockSize) {
    const jEnd = Math.min(jStart + blockSize, N);
    
    for (let i = 0; i < N; i++) {
      // Compute scores for this query against this block of keys
      let blockMax = -Infinity;
      const blockScores = new Float64Array(jEnd - jStart);
      
      for (let j = jStart; j < jEnd; j++) {
        if (causal && j > i) {
          blockScores[j - jStart] = -Infinity;
          continue;
        }
        let dot = 0;
        for (let k = 0; k < d; k++) {
          dot += Q.get(i, k) * K.get(j, k);
        }
        const score = dot * scale;
        blockScores[j - jStart] = score;
        blockMax = Math.max(blockMax, score);
      }
      
      // Online softmax update
      const prevMax = maxScores[i];
      const newMax = Math.max(prevMax, blockMax);
      
      // Rescale previous accumulator
      const rescale = Math.exp(prevMax - newMax);
      const rescaledSum = sumExp[i] * rescale;
      
      // Add contributions from this block
      let blockSum = 0;
      for (let j = 0; j < jEnd - jStart; j++) {
        blockSum += Math.exp(blockScores[j] - newMax);
      }
      
      // Update output: rescale old + add new block contribution
      for (let k = 0; k < d; k++) {
        let blockVal = 0;
        for (let j = 0; j < jEnd - jStart; j++) {
          const weight = Math.exp(blockScores[j] - newMax);
          blockVal += weight * V.get(jStart + j, k);
        }
        output.set(i, k, output.get(i, k) * rescale + blockVal);
      }
      
      maxScores[i] = newMax;
      sumExp[i] = rescaledSum + blockSum;
    }
  }
  
  // Normalize by sum of exponentials
  for (let i = 0; i < N; i++) {
    const s = sumExp[i];
    if (s > 0) {
      for (let k = 0; k < d; k++) {
        output.set(i, k, output.get(i, k) / s);
      }
    }
  }
  
  return output;
}

/**
 * Multi-head flash attention.
 * @param {Matrix} Q - Queries (N × d_model)
 * @param {Matrix} K - Keys (N × d_model)
 * @param {Matrix} V - Values (N × d_model)
 * @param {number} nHeads - Number of attention heads
 * @param {number} blockSize - Block size for flash attention
 * @param {boolean} causal - Apply causal mask
 * @returns {Matrix} Output (N × d_model)
 */
export function multiHeadFlashAttention(Q, K, V, nHeads, blockSize = 32, causal = false) {
  const N = Q.rows;
  const dModel = Q.cols;
  const headDim = dModel / nHeads;
  
  if (dModel % nHeads !== 0) {
    throw new Error(`d_model (${dModel}) must be divisible by nHeads (${nHeads})`);
  }
  
  const output = new Matrix(N, dModel);
  
  for (let h = 0; h < nHeads; h++) {
    // Extract head slice
    const qHead = new Matrix(N, headDim);
    const kHead = new Matrix(N, headDim);
    const vHead = new Matrix(N, headDim);
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < headDim; j++) {
        qHead.set(i, j, Q.get(i, h * headDim + j));
        kHead.set(i, j, K.get(i, h * headDim + j));
        vHead.set(i, j, V.get(i, h * headDim + j));
      }
    }
    
    // Run flash attention on this head
    const headOutput = flashAttention(qHead, kHead, vHead, blockSize, causal);
    
    // Write back to output
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < headDim; j++) {
        output.set(i, h * headDim + j, headOutput.get(i, j));
      }
    }
  }
  
  return output;
}
