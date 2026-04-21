// kv-cache-compression.js — Quantized KV-Cache for Long-Context Inference
// At long context lengths, KV-cache dominates memory. Quantizing cached K/V
// to INT8 reduces memory by 4x (vs FP32) with minimal quality impact.
//
// Approach: per-head per-token quantization (each K/V vector gets its own scale)

import { Matrix } from './matrix.js';

/**
 * Quantized KV-Cache: stores K and V in INT8 format.
 * Each vector is independently quantized with its own scale factor.
 */
export class QuantizedKVCache {
  constructor(headDim, maxTokens = Infinity) {
    this.headDim = headDim;
    this.maxTokens = maxTokens;

    // Quantized storage
    this.kQuantized = [];  // Int8Array[]
    this.kScales = [];     // Float64[]
    this.vQuantized = [];
    this.vScales = [];

    this.totalTokens = 0;
  }

  /**
   * Append a token's K and V vectors.
   * Quantizes to INT8 for storage.
   */
  append(k, v) {
    const { quantized: kq, scale: ks } = quantizeVectorINT8(k);
    const { quantized: vq, scale: vs } = quantizeVectorINT8(v);

    this.kQuantized.push(kq);
    this.kScales.push(ks);
    this.vQuantized.push(vq);
    this.vScales.push(vs);
    this.totalTokens++;

    // Evict oldest if over capacity
    while (this.kQuantized.length > this.maxTokens) {
      this.kQuantized.shift();
      this.kScales.shift();
      this.vQuantized.shift();
      this.vScales.shift();
    }
  }

  /**
   * Get all K vectors as a dequantized Matrix.
   */
  getKeys() {
    return dequantizeVectors(this.kQuantized, this.kScales, this.headDim);
  }

  /**
   * Get all V vectors as a dequantized Matrix.
   */
  getValues() {
    return dequantizeVectors(this.vQuantized, this.vScales, this.headDim);
  }

  get size() { return this.kQuantized.length; }

  clear() {
    this.kQuantized = []; this.kScales = [];
    this.vQuantized = []; this.vScales = [];
    this.totalTokens = 0;
  }

  /**
   * Memory stats (bytes).
   * FP64: 8 bytes per element
   * INT8: 1 byte per element + 8 bytes per scale
   */
  stats() {
    const n = this.kQuantized.length;
    const fp64Memory = n * this.headDim * 2 * 8;      // K + V, FP64
    const int8Memory = n * this.headDim * 2 * 1        // quantized data
                     + n * 2 * 8;                      // scales (FP64)
    return {
      tokens: n,
      fp64Bytes: fp64Memory,
      int8Bytes: int8Memory,
      compressionRatio: fp64Memory > 0 ? (fp64Memory / int8Memory).toFixed(2) + 'x' : '0x',
      savedBytes: fp64Memory - int8Memory,
    };
  }
}

/**
 * Quantize a single vector to INT8 with absmax scaling.
 */
function quantizeVectorINT8(vec) {
  let absMax = 0;
  for (let i = 0; i < vec.length; i++) absMax = Math.max(absMax, Math.abs(vec[i]));
  const scale = absMax / 127 || 1;
  const quantized = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    quantized[i] = Math.max(-127, Math.min(127, Math.round(vec[i] / scale)));
  }
  return { quantized, scale };
}

/**
 * Dequantize multiple vectors into a Matrix.
 */
function dequantizeVectors(quantizedArr, scales, dim) {
  const n = quantizedArr.length;
  const mat = new Matrix(n, dim);
  for (let i = 0; i < n; i++) {
    const scale = scales[i];
    for (let d = 0; d < dim; d++) {
      mat.set(i, d, quantizedArr[i][d] * scale);
    }
  }
  return mat;
}

/**
 * Compare quantized vs unquantized KV-cache for attention quality.
 * Returns the mean absolute difference in attention output.
 */
export function compareQuantizedAttention(Q, K, V, headDim) {
  const N = Q.rows;
  const scale = 1 / Math.sqrt(headDim);

  // Standard attention
  const scores1 = Q.dot(K.T()).mul(scale);
  const attn1 = softmaxRows(scores1);
  const out1 = attn1.dot(V);

  // Quantized K, V
  const cache = new QuantizedKVCache(headDim);
  for (let i = 0; i < N; i++) {
    const k = new Float64Array(headDim);
    const v = new Float64Array(headDim);
    for (let d = 0; d < headDim; d++) {
      k[d] = K.get(i, d);
      v[d] = V.get(i, d);
    }
    cache.append(k, v);
  }

  const Kq = cache.getKeys();
  const Vq = cache.getValues();

  const scores2 = Q.dot(Kq.T()).mul(scale);
  const attn2 = softmaxRows(scores2);
  const out2 = attn2.dot(Vq);

  // Compute mean absolute error
  let totalError = 0;
  for (let r = 0; r < N; r++)
    for (let c = 0; c < headDim; c++)
      totalError += Math.abs(out1.get(r, c) - out2.get(r, c));

  return {
    mae: totalError / (N * headDim),
    stats: cache.stats(),
  };
}

function softmaxRows(mat) {
  const result = new Matrix(mat.rows, mat.cols);
  for (let r = 0; r < mat.rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < mat.cols; c++) max = Math.max(max, mat.get(r, c));
    let sum = 0;
    for (let c = 0; c < mat.cols; c++) {
      result.set(r, c, Math.exp(mat.get(r, c) - max));
      sum += result.get(r, c);
    }
    for (let c = 0; c < mat.cols; c++) result.set(r, c, result.get(r, c) / sum);
  }
  return result;
}
