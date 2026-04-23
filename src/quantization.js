// quantization.js — Weight Quantization for Model Compression
// Techniques used by: GPTQ, AWQ, GGML/GGUF, bitsandbytes
//
// Quantization reduces model size and inference cost by storing weights
// in lower precision (INT8, INT4) instead of FP32/FP16.
//
// Absmax quantization: scale = max(|W|), q = round(W / scale * (2^(bits-1) - 1))
// Zero-point quantization: q = round((W - min) / (max - min) * (2^bits - 1))
// Group quantization: quantize in groups of G (e.g., 128) for better accuracy

import { Matrix } from './matrix.js';

/**
 * Quantize a matrix to INT8 using absmax (symmetric) quantization.
 * @param {Matrix} mat - weight matrix
 * @returns {{ quantized: Int8Array, scale: number, rows: number, cols: number }}
 */
export function quantizeAbsmaxINT8(mat) {
  const data = [];
  let absMax = 0;
  for (let r = 0; r < mat.rows; r++)
    for (let c = 0; c < mat.cols; c++) {
      const v = mat.get(r, c);
      data.push(v);
      absMax = Math.max(absMax, Math.abs(v));
    }

  const scale = absMax / 127; // INT8 range: -127 to 127
  const quantized = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    quantized[i] = Math.max(-127, Math.min(127, Math.round(data[i] / scale)));
  }

  return { quantized, scale, rows: mat.rows, cols: mat.cols };
}

/**
 * Dequantize INT8 back to Matrix.
 */
export function dequantizeINT8(q) {
  const mat = new Matrix(q.rows, q.cols);
  for (let i = 0; i < q.quantized.length; i++) {
    const r = Math.floor(i / q.cols);
    const c = i % q.cols;
    mat.set(r, c, q.quantized[i] * q.scale);
  }
  return mat;
}

/**
 * Quantize a matrix to INT4 using group quantization.
 * Each group of `groupSize` values shares one scale factor.
 * Better accuracy than per-tensor quantization.
 *
 * INT4 range: -7 to 7 (4-bit signed)
 *
 * @param {Matrix} mat - weight matrix
 * @param {number} groupSize - values per group (default 128)
 * @returns {{ quantized: Int8Array, scales: Float64Array, groupSize: number, rows: number, cols: number }}
 */
export function quantizeGroupINT4(mat, groupSize = 128) {
  const totalElements = mat.rows * mat.cols;
  const numGroups = Math.ceil(totalElements / groupSize);

  // Pack 2 INT4 values per byte
  const packedSize = Math.ceil(totalElements / 2);
  const quantized = new Int8Array(packedSize);
  const scales = new Float64Array(numGroups);

  const data = [];
  for (let r = 0; r < mat.rows; r++)
    for (let c = 0; c < mat.cols; c++)
      data.push(mat.get(r, c));

  for (let g = 0; g < numGroups; g++) {
    const start = g * groupSize;
    const end = Math.min(start + groupSize, totalElements);

    // Find absmax for this group
    let absMax = 0;
    for (let i = start; i < end; i++) absMax = Math.max(absMax, Math.abs(data[i]));

    scales[g] = absMax / 7; // INT4 range: -7 to 7

    // Quantize this group
    for (let i = start; i < end; i++) {
      const q = Math.max(-7, Math.min(7, Math.round(data[i] / (scales[g] || 1))));
      const byteIdx = Math.floor(i / 2);
      if (i % 2 === 0) {
        quantized[byteIdx] = (q & 0x0F); // low nibble
      } else {
        quantized[byteIdx] |= ((q & 0x0F) << 4); // high nibble
      }
    }
  }

  return { quantized, scales, groupSize, rows: mat.rows, cols: mat.cols, totalElements };
}

/**
 * Dequantize INT4 group-quantized back to Matrix.
 */
export function dequantizeINT4(q) {
  const mat = new Matrix(q.rows, q.cols);
  let idx = 0;

  for (let g = 0; g < q.scales.length; g++) {
    const start = g * q.groupSize;
    const end = Math.min(start + q.groupSize, q.totalElements);
    const scale = q.scales[g];

    for (let i = start; i < end; i++) {
      const byteIdx = Math.floor(i / 2);
      let val;
      if (i % 2 === 0) {
        val = q.quantized[byteIdx] & 0x0F;
      } else {
        val = (q.quantized[byteIdx] >> 4) & 0x0F;
      }
      // Sign extend from 4 bits
      if (val >= 8) val -= 16;

      const r = Math.floor(i / q.cols);
      const c = i % q.cols;
      mat.set(r, c, val * scale);
      idx++;
    }
  }

  return mat;
}

/**
 * Compute quantization error.
 * Accepts either Matrix objects or plain arrays.
 */
export function quantizationError(original, dequantized) {
  // Array path
  if (Array.isArray(original)) {
    let sumSq = 0, maxError = 0;
    for (let i = 0; i < original.length; i++) {
      const err = Math.abs(original[i] - dequantized[i]);
      sumSq += err * err;
      maxError = Math.max(maxError, err);
    }
    const mse = sumSq / original.length;
    return { mse, rmse: Math.sqrt(mse), maxError };
  }
  // Matrix path
  let totalError = 0;
  let count = 0;
  for (let r = 0; r < original.rows; r++) {
    for (let c = 0; c < original.cols; c++) {
      totalError += Math.abs(original.get(r, c) - dequantized.get(r, c));
      count++;
    }
  }
  return totalError / count;
}

/**
 * Compute compression ratio.
 * FP64: 8 bytes per element
 * INT8: 1 byte per element + scale
 * INT4: 0.5 bytes per element + scales
 */
export function compressionRatio(mat, bits) {
  const fp64Bytes = mat.rows * mat.cols * 8;
  if (bits === 8) {
    const int8Bytes = mat.rows * mat.cols * 1 + 8; // +8 for scale
    return fp64Bytes / int8Bytes;
  } else if (bits === 4) {
    const int4Bytes = Math.ceil(mat.rows * mat.cols / 2) + 
                      Math.ceil(mat.rows * mat.cols / 128) * 8; // +scales
    return fp64Bytes / int4Bytes;
  }
  return 1;
}

// ========================
// Array-based quantization API (used by test/quantization.test.js)
// ========================

/**
 * Uniform quantization of a float array to N-bit integers.
 * @param {number[]} values - input values
 * @param {number} bits - bit width (e.g. 4, 8)
 * @param {boolean} symmetric - if true, symmetric around 0 (default: true)
 * @returns {{ quantized: number[], scale: number, zeroPoint: number }}
 */
export function quantize(values, bits, symmetric = true) {
  // Handle scalar input
  const isScalar = typeof values === 'number';
  if (isScalar) values = [values];
  
  const levels = (1 << bits) - 1;
  const min = Math.min(...values);
  const max = Math.max(...values);

  let scale, zeroPoint;
  if (symmetric) {
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1e-10;
    scale = (2 * absMax) / levels;
    zeroPoint = Math.round(levels / 2);
  } else {
    scale = (max - min) / levels || 1e-10;
    zeroPoint = Math.round(-min / scale);
  }

  const quantized = values.map(v => {
    const q = Math.round(v / scale + zeroPoint);
    return Math.max(0, Math.min(levels, q));
  });

  if (isScalar) {
    return { quantized: quantized[0], scale, zeroPoint, _bits: bits };
  }
  return { quantized, scale, zeroPoint, _bits: bits };
}

/**
 * Dequantize integers back to floats.
 * Accepts either (quantized, scale, zeroPoint) or ({quantized, scale, zeroPoint}, bits).
 */
export function dequantize(quantizedOrObj, scaleOrBits, zeroPoint) {
  // Handle object input: dequantize(result, bits) where result = quantize output
  if (typeof quantizedOrObj === 'object' && quantizedOrObj !== null && 'scale' in quantizedOrObj) {
    const obj = quantizedOrObj;
    const q = obj.quantized;
    if (typeof q === 'number') {
      return (q - obj.zeroPoint) * obj.scale;
    }
    return q.map(v => (v - obj.zeroPoint) * obj.scale);
  }
  // Array path: dequantize(quantized, scale, zeroPoint)
  if (Array.isArray(quantizedOrObj)) {
    return quantizedOrObj.map(q => (q - zeroPoint) * scaleOrBits);
  }
  // Scalar
  return (quantizedOrObj - zeroPoint) * scaleOrBits;
}

/**
 * Fake quantization: quantize then immediately dequantize (simulates quantization noise).
 */
export function fakeQuantize(values, bits) {
  const { quantized, scale, zeroPoint } = quantize(values, bits);
  return dequantize(quantized, scale, zeroPoint);
}

/**
 * Per-channel (per-row) quantization of a 2D array.
 */
export function quantizePerChannel(matrix, bits) {
  const quantized = [];
  const scales = [];
  const zeroPoints = [];

  for (const row of matrix) {
    const result = quantize(row, bits);
    quantized.push(result.quantized);
    scales.push(result.scale);
    zeroPoints.push(result.zeroPoint);
  }

  return { quantized, scales, zeroPoints };
}

/**
 * Dequantize per-channel.
 */
export function dequantizePerChannel(quantized, scales, zeroPoints) {
  return quantized.map((row, i) => dequantize(row, scales[i], zeroPoints[i]));
}

/**
 * Dynamic quantization: find the minimum bits needed for a target error.
 */
export function dynamicQuantize(values, targetRMSE = 0.01) {
  for (let bits = 2; bits <= 16; bits++) {
    const fq = fakeQuantize(values, bits);
    const { rmse } = quantizationError(values, fq);
    if (rmse <= targetRMSE) {
      return { bits, error: rmse, quantized: fq };
    }
  }
  return { bits: 16, error: 0, quantized: fakeQuantize(values, 16) };
}

/**
 * K-means weight clustering (codebook quantization).
 */
export function clusterWeights(weights, k, maxIter = 50) {
  // Initialize centroids via uniform sampling
  const sorted = [...weights].sort((a, b) => a - b);
  let centroids = [];
  for (let i = 0; i < k; i++) {
    centroids.push(sorted[Math.floor(i * sorted.length / k)]);
  }

  let assignments = new Array(weights.length);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign each weight to nearest centroid
    for (let i = 0; i < weights.length; i++) {
      let minDist = Infinity, bestC = 0;
      for (let c = 0; c < k; c++) {
        const d = Math.abs(weights[i] - centroids[c]);
        if (d < minDist) { minDist = d; bestC = c; }
      }
      assignments[i] = bestC;
    }

    // Recompute centroids
    const sums = new Array(k).fill(0);
    const counts = new Array(k).fill(0);
    for (let i = 0; i < weights.length; i++) {
      sums[assignments[i]] += weights[i];
      counts[assignments[i]]++;
    }
    const newCentroids = centroids.map((c, i) => counts[i] > 0 ? sums[i] / counts[i] : c);
    
    // Check convergence
    let converged = true;
    for (let i = 0; i < k; i++) {
      if (Math.abs(newCentroids[i] - centroids[i]) > 1e-8) { converged = false; break; }
    }
    centroids = newCentroids;
    if (converged) break;
  }

  // Map weights to centroids
  const quantized = weights.map((_, i) => centroids[assignments[i]]);
  const bitsPerWeight = Math.log2(k);
  const compressionRatio = bitsPerWeight / 32;

  return { centroids, quantized, assignments, compressionRatio };
}

/**
 * Quantize a weight matrix to N-bit.
 * @param {Matrix} mat - weight matrix
 * @param {number} bits - bit width
 * @returns {{ quantized: Int8Array, scale: number, zeroPoint: number }}
 */
export function quantizeWeights(mat, bits) {
  const values = Array.from(mat.data);
  return quantize(values, bits);
}

/**
 * Calculate bits required to represent N levels.
 * @param {number} levels - number of distinct levels
 * @returns {number} bits needed
 */
export function bitsRequired(levels) {
  return Math.ceil(Math.log2(levels));
}
