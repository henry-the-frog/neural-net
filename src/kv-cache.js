// kv-cache.js — Key-Value Cache for Autoregressive Generation
//
// During autoregressive (token-by-token) generation, the KV cache stores
// previously computed K and V vectors so they don't need recomputation.
// 
// Without cache: each new token requires computing attention over ALL tokens
// With cache: only the new token's K,V are computed and appended
//
// Memory: O(seqLen × numKVHeads × headDim) per layer
// Speedup: O(seqLen) per token → O(1) per token for K,V computation

import { Matrix } from './matrix.js';

/**
 * KV Cache for a single attention layer.
 * Stores K and V matrices and supports incremental append.
 */
export class KVCache {
  /**
   * @param {number} numKVHeads — number of KV heads
   * @param {number} headDim — dimension per head
   * @param {number} [maxSeqLen=2048] — maximum sequence length
   */
  constructor(numKVHeads, headDim, maxSeqLen = 2048) {
    this.numKVHeads = numKVHeads;
    this.headDim = headDim;
    this.maxSeqLen = maxSeqLen;
    this.seqLen = 0;
    
    // Pre-allocate for max sequence length
    // keys[h] and values[h] are [maxSeqLen, headDim] but only seqLen rows are valid
    this.keys = [];
    this.values = [];
    for (let h = 0; h < numKVHeads; h++) {
      this.keys.push(new Matrix(maxSeqLen, headDim));
      this.values.push(new Matrix(maxSeqLen, headDim));
    }
  }
  
  /**
   * Append new K, V vectors for a single token.
   * @param {Matrix[]} newKeys — per-head K vectors, each [1, headDim]
   * @param {Matrix[]} newValues — per-head V vectors, each [1, headDim]
   */
  append(newKeys, newValues) {
    if (this.seqLen >= this.maxSeqLen) {
      throw new Error(`KV cache full (${this.maxSeqLen} tokens)`);
    }
    
    for (let h = 0; h < this.numKVHeads; h++) {
      for (let d = 0; d < this.headDim; d++) {
        this.keys[h].set(this.seqLen, d, newKeys[h].get(0, d));
        this.values[h].set(this.seqLen, d, newValues[h].get(0, d));
      }
    }
    this.seqLen++;
  }
  
  /**
   * Append multiple tokens at once (prefill phase).
   * @param {Matrix[]} newKeys — per-head K matrices, each [numTokens, headDim]
   * @param {Matrix[]} newValues — per-head V matrices, each [numTokens, headDim]
   */
  appendMultiple(newKeys, newValues) {
    const numTokens = newKeys[0].rows;
    if (this.seqLen + numTokens > this.maxSeqLen) {
      throw new Error(`KV cache overflow: ${this.seqLen + numTokens} > ${this.maxSeqLen}`);
    }
    
    for (let h = 0; h < this.numKVHeads; h++) {
      for (let t = 0; t < numTokens; t++) {
        for (let d = 0; d < this.headDim; d++) {
          this.keys[h].set(this.seqLen + t, d, newKeys[h].get(t, d));
          this.values[h].set(this.seqLen + t, d, newValues[h].get(t, d));
        }
      }
    }
    this.seqLen += numTokens;
  }
  
  /**
   * Get the full K matrix for a head (up to current seqLen).
   * @param {number} head — head index
   * @returns {Matrix} — [seqLen, headDim]
   */
  getKeys(head) {
    const result = new Matrix(this.seqLen, this.headDim);
    for (let t = 0; t < this.seqLen; t++)
      for (let d = 0; d < this.headDim; d++)
        result.set(t, d, this.keys[head].get(t, d));
    return result;
  }
  
  /**
   * Get the full V matrix for a head (up to current seqLen).
   */
  getValues(head) {
    const result = new Matrix(this.seqLen, this.headDim);
    for (let t = 0; t < this.seqLen; t++)
      for (let d = 0; d < this.headDim; d++)
        result.set(t, d, this.values[head].get(t, d));
    return result;
  }
  
  /**
   * Get current cache size in elements.
   */
  size() {
    return 2 * this.seqLen * this.numKVHeads * this.headDim;
  }
  
  /**
   * Reset the cache.
   */
  reset() {
    this.seqLen = 0;
  }
  
  /**
   * Clone the cache (for beam search).
   */
  clone() {
    const copy = new KVCache(this.numKVHeads, this.headDim, this.maxSeqLen);
    copy.seqLen = this.seqLen;
    for (let h = 0; h < this.numKVHeads; h++) {
      for (let t = 0; t < this.seqLen; t++)
        for (let d = 0; d < this.headDim; d++) {
          copy.keys[h].set(t, d, this.keys[h].get(t, d));
          copy.values[h].set(t, d, this.values[h].get(t, d));
        }
    }
    return copy;
  }
}

/**
 * Multi-layer KV cache for a full transformer model.
 */
export class ModelKVCache {
  constructor(numLayers, numKVHeads, headDim, maxSeqLen = 2048) {
    this.layers = [];
    for (let l = 0; l < numLayers; l++) {
      this.layers.push(new KVCache(numKVHeads, headDim, maxSeqLen));
    }
  }
  
  getLayer(layerIdx) {
    return this.layers[layerIdx];
  }
  
  seqLen() {
    return this.layers[0].seqLen;
  }
  
  totalSize() {
    return this.layers.reduce((sum, l) => sum + l.size(), 0);
  }
  
  reset() {
    for (const l of this.layers) l.reset();
  }
}
