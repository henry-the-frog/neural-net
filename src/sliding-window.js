// sliding-window.js — Sliding Window Attention (Mistral-style)
// Each token attends only to the W nearest previous tokens.
// Combined with multi-layer stacking, effective context = W × num_layers.
//
// Benefits:
// - O(N × W) compute/memory instead of O(N²) for full attention
// - KV-cache bounded: only store W entries per layer
// - Enables very long sequences (Mistral: 128K context with W=4096)

import { Matrix } from './matrix.js';

/**
 * Sliding Window Attention
 *
 * @param {Matrix} Q - [seqLen, headDim]
 * @param {Matrix} K - [seqLen, headDim]
 * @param {Matrix} V - [seqLen, headDim]
 * @param {number} windowSize - number of previous tokens to attend to
 * @param {boolean} causal - apply causal mask (default true)
 * @returns {{ output: Matrix, stats: object }}
 */
export function slidingWindowAttention(Q, K, V, windowSize, causal = true) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1 / Math.sqrt(d);
  const output = new Matrix(N, d);

  for (let qi = 0; qi < N; qi++) {
    // Window: attend to positions max(0, qi - windowSize + 1) .. qi
    const start = causal ? Math.max(0, qi - windowSize + 1) : Math.max(0, qi - Math.floor(windowSize / 2));
    const end = causal ? qi + 1 : Math.min(N, qi + Math.ceil(windowSize / 2) + 1);
    const winLen = end - start;

    // Compute scores within window
    const scores = new Float64Array(winLen);
    let maxScore = -Infinity;
    for (let j = 0; j < winLen; j++) {
      let dot = 0;
      for (let dd = 0; dd < d; dd++) dot += Q.get(qi, dd) * K.get(start + j, dd);
      scores[j] = dot * scale;
      maxScore = Math.max(maxScore, scores[j]);
    }

    // Softmax
    let sum = 0;
    for (let j = 0; j < winLen; j++) {
      scores[j] = Math.exp(scores[j] - maxScore);
      sum += scores[j];
    }
    for (let j = 0; j < winLen; j++) scores[j] /= sum;

    // Weighted sum of V
    for (let dd = 0; dd < d; dd++) {
      let val = 0;
      for (let j = 0; j < winLen; j++) {
        val += scores[j] * V.get(start + j, dd);
      }
      output.set(qi, dd, val);
    }
  }

  return {
    output,
    stats: {
      windowSize,
      peakMemory: N * windowSize, // bounded, not N²
      effectiveContext: windowSize,
      method: 'sliding_window',
    }
  };
}

/**
 * Sliding Window KV-Cache: only stores the last W key-value pairs.
 * When the cache exceeds W entries, oldest entries are evicted.
 */
export class SlidingWindowKVCache {
  constructor(windowSize, headDim) {
    this.windowSize = windowSize;
    this.headDim = headDim;
    this.keys = [];   // circular buffer of Float64Array
    this.values = []; // circular buffer of Float64Array
    this.totalTokens = 0;
  }

  /**
   * Append a new token's K and V to the cache.
   * Evicts oldest if over window size.
   */
  append(k, v) {
    this.keys.push(Float64Array.from(k));
    this.values.push(Float64Array.from(v));
    this.totalTokens++;

    // Evict oldest entries beyond window
    while (this.keys.length > this.windowSize) {
      this.keys.shift();
      this.values.shift();
    }
  }

  /**
   * Get all cached K vectors as a Matrix.
   */
  getKeys() {
    const n = this.keys.length;
    const mat = new Matrix(n, this.headDim);
    for (let i = 0; i < n; i++)
      for (let d = 0; d < this.headDim; d++)
        mat.set(i, d, this.keys[i][d]);
    return mat;
  }

  /**
   * Get all cached V vectors as a Matrix.
   */
  getValues() {
    const n = this.values.length;
    const mat = new Matrix(n, this.headDim);
    for (let i = 0; i < n; i++)
      for (let d = 0; d < this.headDim; d++)
        mat.set(i, d, this.values[i][d]);
    return mat;
  }

  /**
   * Current cache size.
   */
  get size() {
    return this.keys.length;
  }

  /**
   * Clear cache.
   */
  clear() {
    this.keys = [];
    this.values = [];
    this.totalTokens = 0;
  }

  /**
   * Memory stats.
   */
  stats() {
    return {
      cached: this.keys.length,
      maxCapacity: this.windowSize,
      totalTokensSeen: this.totalTokens,
      evicted: this.totalTokens - this.keys.length,
      memoryElements: this.keys.length * this.headDim * 2, // K + V
    };
  }
}
