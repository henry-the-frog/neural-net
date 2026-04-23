// kv-cache.js — KV Cache for efficient autoregressive generation
// During generation, we don't need to recompute K,V for already-generated tokens.
// The KV cache stores previous K,V values and only computes new ones for new tokens.
//
// Memory: O(seqLen * d * numLayers) — grows linearly with generated tokens
// Speedup: O(1) per new token instead of O(seqLen)

import { Matrix } from './matrix.js';

export class KVCache {
  /**
   * @param {number} maxLen - Maximum sequence length
   * @param {number} numLayers - Number of attention layers
   * @param {number} numKVHeads - Number of KV heads per layer
   * @param {number} headDim - Dimension per head
   */
  constructor(maxLen, numLayers, numKVHeads, headDim) {
    this.maxLen = maxLen;
    this.numLayers = numLayers;
    this.numKVHeads = numKVHeads;
    this.headDim = headDim;
    this.seqLen = 0; // Current cached length
    
    // Preallocate cache: one K and V matrix per layer
    this.keys = [];
    this.values = [];
    for (let l = 0; l < numLayers; l++) {
      this.keys.push(new Matrix(maxLen, numKVHeads * headDim));
      this.values.push(new Matrix(maxLen, numKVHeads * headDim));
    }
  }

  /**
   * Append new K,V for a single token at a specific layer.
   * @param {number} layer - Layer index
   * @param {Float64Array} keyRow - Key values for this token (numKVHeads * headDim)
   * @param {Float64Array} valueRow - Value values for this token
   */
  append(layer, keyRow, valueRow) {
    if (this.seqLen >= this.maxLen) {
      throw new Error(`KV cache full: ${this.seqLen} >= ${this.maxLen}`);
    }
    const dim = this.numKVHeads * this.headDim;
    for (let i = 0; i < dim; i++) {
      this.keys[layer].set(this.seqLen, i, keyRow[i]);
      this.values[layer].set(this.seqLen, i, valueRow[i]);
    }
  }

  /**
   * Increment the sequence position (call after appending to all layers).
   */
  incrementSeqLen() {
    this.seqLen++;
  }

  /**
   * Get cached K values for a layer (up to current seqLen).
   * @param {number} layer - Layer index
   * @returns {Matrix} Cached keys (seqLen × dim)
   */
  getKeys(layer) {
    const dim = this.numKVHeads * this.headDim;
    const result = new Matrix(this.seqLen, dim);
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < dim; j++) {
        result.set(i, j, this.keys[layer].get(i, j));
      }
    }
    return result;
  }

  /**
   * Get cached V values for a layer (up to current seqLen).
   * @param {number} layer - Layer index
   * @returns {Matrix} Cached values (seqLen × dim)
   */
  getValues(layer) {
    const dim = this.numKVHeads * this.headDim;
    const result = new Matrix(this.seqLen, dim);
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < dim; j++) {
        result.set(i, j, this.values[layer].get(i, j));
      }
    }
    return result;
  }

  /**
   * Reset cache (start new generation).
   */
  reset() {
    this.seqLen = 0;
  }

  /**
   * Get memory usage in bytes (Float64: 8 bytes per element).
   */
  memoryBytes() {
    return this.numLayers * 2 * this.maxLen * this.numKVHeads * this.headDim * 8;
  }

  /**
   * Get utilization percentage.
   */
  utilization() {
    return (this.seqLen / this.maxLen * 100).toFixed(1) + '%';
  }
}
