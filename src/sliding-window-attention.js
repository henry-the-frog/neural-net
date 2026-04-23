// sliding-window-attention.js — Sliding Window Attention (Beltagy et al., 2020 / Mistral 2023)
// Each token only attends to the W nearest tokens (window size W).
// Reduces attention complexity from O(N²) to O(N*W).
// Key insight: information can still propagate across the full sequence
// through multiple layers (each layer shifts the window).

import { Matrix } from './matrix.js';

/**
 * Sliding Window Attention.
 * Each position i attends only to positions max(0, i-W+1) through i.
 * @param {Matrix} Q - Queries (N × d)
 * @param {Matrix} K - Keys (N × d)
 * @param {Matrix} V - Values (N × d)
 * @param {number} windowSize - Size of the sliding window
 * @returns {Matrix} Output (N × d)
 */
export function slidingWindowAttention(Q, K, V, windowSize) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1.0 / Math.sqrt(d);
  const output = new Matrix(N, d);
  
  for (let i = 0; i < N; i++) {
    // Window: positions [max(0, i-W+1), i]
    const windowStart = Math.max(0, i - windowSize + 1);
    const windowEnd = i + 1;
    const windowLen = windowEnd - windowStart;
    
    // Compute scores for positions in window
    let maxScore = -Infinity;
    const scores = new Float64Array(windowLen);
    
    for (let jIdx = 0; jIdx < windowLen; jIdx++) {
      const j = windowStart + jIdx;
      let dot = 0;
      for (let k = 0; k < d; k++) {
        dot += Q.get(i, k) * K.get(j, k);
      }
      scores[jIdx] = dot * scale;
      maxScore = Math.max(maxScore, scores[jIdx]);
    }
    
    // Softmax
    let sumExp = 0;
    for (let jIdx = 0; jIdx < windowLen; jIdx++) {
      scores[jIdx] = Math.exp(scores[jIdx] - maxScore);
      sumExp += scores[jIdx];
    }
    
    // Weighted sum of V
    for (let k = 0; k < d; k++) {
      let val = 0;
      for (let jIdx = 0; jIdx < windowLen; jIdx++) {
        val += (scores[jIdx] / sumExp) * V.get(windowStart + jIdx, k);
      }
      output.set(i, k, val);
    }
  }
  
  return output;
}

/**
 * Compute effective receptive field for sliding window across layers.
 * After L layers with window W, each token can see W*L positions back.
 * @param {number} windowSize - Per-layer window size
 * @param {number} numLayers - Number of attention layers
 * @returns {{ receptiveField: number, description: string }}
 */
export function effectiveReceptiveField(windowSize, numLayers) {
  const field = windowSize * numLayers;
  return {
    receptiveField: field,
    description: `Window=${windowSize}, Layers=${numLayers}: each token sees up to ${field} positions back`,
  };
}
