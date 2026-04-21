// attention-sink.js — Attention Sink KV-Cache (StreamingLLM)
// Paper: "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023)
//
// Observation: Transformers heavily attend to the first few tokens (attention sinks),
// even when those tokens are irrelevant to the query. Removing them from the
// KV-cache causes quality degradation.
//
// Solution: Keep the first K "sink" tokens in cache permanently,
// plus a sliding window of recent tokens. This enables infinite-length
// generation with bounded memory.
//
// Cache layout: [sink_0, sink_1, ..., sink_K, recent_{n-W}, ..., recent_n]

import { Matrix } from './matrix.js';

/**
 * Attention Sink KV-Cache
 * Keeps first `sinkSize` tokens permanently + sliding window of `windowSize` recent tokens.
 *
 * @param {number} headDim - dimension per head
 * @param {number} sinkSize - number of initial tokens to keep permanently
 * @param {number} windowSize - sliding window for recent tokens
 */
export class AttentionSinkCache {
  constructor(headDim, sinkSize = 4, windowSize = 512) {
    this.headDim = headDim;
    this.sinkSize = sinkSize;
    this.windowSize = windowSize;

    this.sinkKeys = [];     // first K tokens' keys (permanent)
    this.sinkValues = [];
    this.recentKeys = [];   // sliding window of recent keys
    this.recentValues = [];
    this.totalTokens = 0;
  }

  /**
   * Append a new token's K and V.
   */
  append(k, v) {
    this.totalTokens++;

    if (this.sinkKeys.length < this.sinkSize) {
      // Still filling sink slots
      this.sinkKeys.push(Float64Array.from(k));
      this.sinkValues.push(Float64Array.from(v));
    } else {
      // Add to recent window
      this.recentKeys.push(Float64Array.from(k));
      this.recentValues.push(Float64Array.from(v));

      // Evict oldest from recent if over window
      while (this.recentKeys.length > this.windowSize) {
        this.recentKeys.shift();
        this.recentValues.shift();
      }
    }
  }

  /**
   * Get all cached K vectors as Matrix: [sinks + recent, headDim]
   */
  getKeys() {
    return vecArrayToMatrix([...this.sinkKeys, ...this.recentKeys], this.headDim);
  }

  /**
   * Get all cached V vectors as Matrix.
   */
  getValues() {
    return vecArrayToMatrix([...this.sinkValues, ...this.recentValues], this.headDim);
  }

  /**
   * Total cached tokens.
   */
  get size() {
    return this.sinkKeys.length + this.recentKeys.length;
  }

  /**
   * Maximum cache capacity.
   */
  get capacity() {
    return this.sinkSize + this.windowSize;
  }

  clear() {
    this.sinkKeys = []; this.sinkValues = [];
    this.recentKeys = []; this.recentValues = [];
    this.totalTokens = 0;
  }

  stats() {
    return {
      sinkTokens: this.sinkKeys.length,
      recentTokens: this.recentKeys.length,
      totalCached: this.size,
      totalSeen: this.totalTokens,
      evicted: this.totalTokens - this.size,
      capacity: this.capacity,
      memoryElements: this.size * this.headDim * 2,
    };
  }
}

function vecArrayToMatrix(vecs, dim) {
  const n = vecs.length;
  if (n === 0) return new Matrix(0, dim);
  const mat = new Matrix(n, dim);
  for (let i = 0; i < n; i++)
    for (let d = 0; d < dim; d++)
      mat.set(i, d, vecs[i][d]);
  return mat;
}
