// gqa-attention.js — Grouped Query Attention (GQA) with KV-Cache
// Used in modern LLMs: Llama 2/3, Mistral, Gemma, etc.
// GQA shares K,V heads across multiple Q heads to reduce memory.
// KV-cache stores computed K,V for past tokens to avoid recomputation during generation.

import { Matrix } from './matrix.js';

/**
 * Softmax over rows — each row is independently softmaxed.
 */
function softmaxRows(mat) {
  const result = new Matrix(mat.rows, mat.cols);
  for (let r = 0; r < mat.rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < mat.cols; c++) max = Math.max(max, mat.get(r, c));
    let sum = 0;
    for (let c = 0; c < mat.cols; c++) {
      const v = Math.exp(mat.get(r, c) - max);
      result.set(r, c, v);
      sum += v;
    }
    for (let c = 0; c < mat.cols; c++) result.set(r, c, result.get(r, c) / sum);
  }
  return result;
}

function extractCols(mat, startCol, numCols) {
  const out = new Matrix(mat.rows, numCols);
  for (let r = 0; r < mat.rows; r++)
    for (let c = 0; c < numCols; c++)
      out.set(r, c, mat.get(r, startCol + c));
  return out;
}

/**
 * Grouped Query Attention (GQA)
 *
 * Standard Multi-Head Attention: numQHeads = numKVHeads (each Q head has its own K,V)
 * Multi-Query Attention (MQA):   numKVHeads = 1 (all Q heads share one K,V)
 * GQA:                           numQHeads > numKVHeads, numQHeads % numKVHeads === 0
 *
 * Memory savings: KV-cache size is (numKVHeads / numQHeads) of standard MHA.
 * Quality: GQA ≈ MHA quality with much less KV memory for long sequences.
 *
 * @param {number} dModel - total model dimension
 * @param {number} numQHeads - number of query heads
 * @param {number} numKVHeads - number of key/value heads (must divide numQHeads evenly)
 */
export class GroupedQueryAttention {
  constructor(dModel, numQHeads, numKVHeads, { causal = true } = {}) {
    if (dModel % numQHeads !== 0) throw new Error('dModel must be divisible by numQHeads');
    if (numQHeads % numKVHeads !== 0) throw new Error('numQHeads must be divisible by numKVHeads');

    this.dModel = dModel;
    this.numQHeads = numQHeads;
    this.numKVHeads = numKVHeads;
    this.headDim = dModel / numQHeads;
    this.kvDim = numKVHeads * this.headDim;
    this.groupSize = numQHeads / numKVHeads; // Q heads per KV head
    this.causal = causal;

    // Q projection: dModel → dModel (all Q heads)
    const scale = Math.sqrt(2 / (dModel + dModel));
    this.Wq = Matrix.random(dModel, dModel).mul(scale);
    this.Wk = Matrix.random(dModel, this.kvDim).mul(scale);
    this.Wv = Matrix.random(dModel, this.kvDim).mul(scale);
    this.Wo = Matrix.random(dModel, dModel).mul(scale);

    this.bq = Matrix.zeros(1, dModel);
    this.bk = Matrix.zeros(1, this.kvDim);
    this.bv = Matrix.zeros(1, this.kvDim);
    this.bo = Matrix.zeros(1, dModel);

    this.outputSize = dModel;
    this._cache = null; // KV-cache: { K: Matrix, V: Matrix } per batch
  }

  /**
   * Reset KV-cache (call before new sequence generation).
   */
  clearCache() {
    this._cache = null;
  }

  /**
   * Forward pass with optional KV-cache for autoregressive generation.
   *
   * @param {Matrix} input - [batch, seqLen * dModel]
   * @param {boolean} useCache - if true, append to KV-cache and attend over full history
   * @returns {Matrix} [batch, seqLen * dModel]
   */
  forward(input, useCache = false) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const result = new Matrix(batchSize, seqLen * this.dModel);

    for (let b = 0; b < batchSize; b++) {
      // Extract sequence: [seqLen, dModel]
      const seq = new Matrix(seqLen, this.dModel);
      for (let t = 0; t < seqLen; t++)
        for (let d = 0; d < this.dModel; d++)
          seq.set(t, d, input.get(b, t * this.dModel + d));

      // Project Q, K, V
      const Q = seq.dot(this.Wq).add(this.bq); // [seqLen, dModel]
      let K = seq.dot(this.Wk).add(this.bk);    // [seqLen, kvDim]
      let V = seq.dot(this.Wv).add(this.bv);    // [seqLen, kvDim]

      // KV-cache: append new K,V to history
      if (useCache) {
        if (!this._cache) {
          this._cache = []; // per batch
        }
        if (!this._cache[b]) {
          this._cache[b] = { K, V };
        } else {
          // Concatenate: [prevLen + seqLen, kvDim]
          K = verticalConcat(this._cache[b].K, K);
          V = verticalConcat(this._cache[b].V, V);
          this._cache[b] = { K, V };
        }
      }

      const totalLen = K.rows; // with cache: includes past tokens

      // Compute attention per Q-head, sharing K,V heads via groups
      const headOutputs = [];
      for (let qh = 0; qh < this.numQHeads; qh++) {
        const kvh = Math.floor(qh / this.groupSize); // which KV head this Q head maps to
        const qOffset = qh * this.headDim;
        const kvOffset = kvh * this.headDim;

        // Extract head slices
        const Qh = extractCols(Q, qOffset, this.headDim);     // [seqLen, headDim]
        const Kh = extractCols(K, kvOffset, this.headDim);     // [totalLen, headDim]
        const Vh = extractCols(V, kvOffset, this.headDim);     // [totalLen, headDim]

        // Attention scores: [seqLen, totalLen]
        const scores = Qh.dot(Kh.T()).mul(1 / Math.sqrt(this.headDim));

        // Apply causal mask if needed
        if (this.causal) {
          const offset = totalLen - seqLen; // cache offset
          for (let i = 0; i < seqLen; i++)
            for (let j = 0; j < totalLen; j++)
              if (j > i + offset) scores.set(i, j, -1e9);
        }

        const attn = softmaxRows(scores);
        const context = attn.dot(Vh); // [seqLen, headDim]
        headOutputs.push(context);
      }

      // Concatenate all Q-heads: [seqLen, dModel]
      const concat = new Matrix(seqLen, this.dModel);
      for (let qh = 0; qh < this.numQHeads; qh++) {
        const offset = qh * this.headDim;
        for (let t = 0; t < seqLen; t++)
          for (let d = 0; d < this.headDim; d++)
            concat.set(t, offset + d, headOutputs[qh].get(t, d));
      }

      // Output projection
      const output = concat.dot(this.Wo).add(this.bo);

      for (let t = 0; t < seqLen; t++)
        for (let d = 0; d < this.dModel; d++)
          result.set(b, t * this.dModel + d, output.get(t, d));
    }

    return result;
  }

  /**
   * Get KV-cache memory usage stats.
   */
  cacheStats() {
    if (!this._cache) return { entries: 0, totalTokens: 0, memoryBytes: 0 };
    let totalTokens = 0;
    for (const entry of this._cache) {
      if (entry) totalTokens += entry.K.rows;
    }
    // Each token stores headDim floats × numKVHeads × 2 (K+V) × 8 bytes
    const memoryBytes = totalTokens * this.kvDim * 2 * 8;
    return { entries: this._cache.length, totalTokens, memoryBytes };
  }
}

/**
 * Vertically concatenate two matrices (stack rows).
 */
function verticalConcat(a, b) {
  const result = new Matrix(a.rows + b.rows, a.cols);
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < a.cols; c++)
      result.set(r, c, a.get(r, c));
  for (let r = 0; r < b.rows; r++)
    for (let c = 0; c < b.cols; c++)
      result.set(a.rows + r, c, b.get(r, c));
  return result;
}
