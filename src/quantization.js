// quantization.js — Post-training Quantization utilities
// Reduces model size by converting FP64/FP32 weights to lower precision.
// INT8 quantization: 4x memory reduction with minimal quality loss.

import { Matrix } from './matrix.js';

/**
 * Absmax quantization: scale to [-127, 127] range using absolute maximum.
 * Simple and fast. Works well for normally distributed weights.
 * @param {Matrix} weights - FP64 weight matrix
 * @returns {{ quantized: Int8Array, scale: number, shape: [number, number] }}
 */
export function quantizeAbsmax(weights) {
  let absMax = 0;
  for (let i = 0; i < weights.data.length; i++) {
    absMax = Math.max(absMax, Math.abs(weights.data[i]));
  }
  
  const scale = absMax / 127;
  const quantized = new Int8Array(weights.data.length);
  
  for (let i = 0; i < weights.data.length; i++) {
    quantized[i] = Math.round(weights.data[i] / scale);
  }
  
  return { quantized, scale, shape: [weights.rows, weights.cols] };
}

/**
 * Dequantize absmax-quantized weights back to FP64.
 */
export function dequantizeAbsmax({ quantized, scale, shape }) {
  const result = new Matrix(shape[0], shape[1]);
  for (let i = 0; i < quantized.length; i++) {
    result.data[i] = quantized[i] * scale;
  }
  return result;
}

/**
 * Per-channel quantization: quantize each row independently.
 * Better quality than absmax for matrices with variable row magnitudes.
 */
export function quantizePerChannel(weights, bits = 8) {
  // Handle both Matrix and plain 2D array
  if (weights.data && weights.rows) {
    // Matrix version
    const scales = new Float64Array(weights.rows);
    const quantized = new Int8Array(weights.data.length);
    
    for (let i = 0; i < weights.rows; i++) {
      let rowMax = 0;
      for (let j = 0; j < weights.cols; j++) {
        rowMax = Math.max(rowMax, Math.abs(weights.get(i, j)));
      }
      scales[i] = rowMax / 127 || 1e-10;
      
      for (let j = 0; j < weights.cols; j++) {
        quantized[i * weights.cols + j] = Math.round(weights.get(i, j) / scales[i]);
      }
    }
    
    return { quantized, scales, shape: [weights.rows, weights.cols] };
  }
  
  // Plain 2D array version
  const rows = weights.length;
  const cols = weights[0].length;
  const levels = (1 << bits) - 1;
  const halfLevels = Math.floor(levels / 2);
  const scales = new Array(rows);
  const zeroPoints = new Array(rows);
  const quantized = weights.map((row, i) => {
    const absMax = Math.max(...row.map(Math.abs)) || 1e-10;
    scales[i] = (2 * absMax) / levels;
    zeroPoints[i] = halfLevels;
    return row.map(v => {
      const q = Math.round(v / scales[i] + zeroPoints[i]);
      return Math.max(0, Math.min(levels, q));
    });
  });
  
  return { quantized, scales, zeroPoints, shape: [rows, cols] };
}

export function dequantizePerChannel(quantizedOrObj, scales, zeroPoints) {
  // Handle old API: dequantizePerChannel({ quantized, scales, shape })
  if (quantizedOrObj && typeof quantizedOrObj === 'object' && !Array.isArray(quantizedOrObj) && quantizedOrObj.quantized !== undefined) {
    const obj = quantizedOrObj;
    if (obj.zeroPoints) {
      return dequantizePerChannel(obj.quantized, obj.scales, obj.zeroPoints);
    }
    const result = new Matrix(obj.shape[0], obj.shape[1]);
    const cols = obj.shape[1];
    for (let i = 0; i < obj.shape[0]; i++) {
      for (let j = 0; j < cols; j++) {
        result.set(i, j, obj.quantized[i * cols + j] * obj.scales[i]);
      }
    }
    return result;
  }
  // Handle both formats with separate args
  if (zeroPoints && Array.isArray(quantizedOrObj[0])) {
    return quantizedOrObj.map((row, i) =>
      row.map(q => (q - zeroPoints[i]) * scales[i])
    );
  }
  // Matrix version (separate args)
  if (scales) {
    const totalLen = quantizedOrObj.length;
    const numRows = scales.length;
    const numCols = totalLen / numRows;
    const result = new Matrix(numRows, numCols);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols; j++) {
        result.set(i, j, quantizedOrObj[i * numCols + j] * scales[i]);
      }
    }
    return result;
  }
  return quantizedOrObj;
}

/**
 * Compute quantization error metrics.
 */
export function quantizationError(original, dequantized) {
  // Handle both Matrix objects and plain arrays
  const origData = original.data || original;
  const deqData = dequantized.data || dequantized;
  let mse = 0, maxError = 0;
  for (let i = 0; i < origData.length; i++) {
    const err = Math.abs(origData[i] - deqData[i]);
    mse += err * err;
    maxError = Math.max(maxError, err);
  }
  mse /= origData.length;
  return {
    mse,
    rmse: Math.sqrt(mse),
    maxError,
    snr: 10 * Math.log10(origData.reduce((s, v) => s + v * v, 0) / origData.length / (mse || 1e-10)),
  };
}

/**
 * Compute compression ratio.
 */
export function compressionRatio(original, quantized) {
  const originalBytes = original.data.length * 8; // Float64
  const quantizedBytes = quantized.quantized.length * 1 + (quantized.scale ? 8 : quantized.scales.length * 8);
  return {
    originalBytes,
    quantizedBytes,
    ratio: (originalBytes / quantizedBytes).toFixed(1) + 'x',
  };
}

// === Array-based quantization API (for tests and simple usage) ===

/**
 * Uniform quantization of a flat array to N-bit integers.
 * @param {number[]} values - Input values
 * @param {number} bits - Bit width (e.g. 8, 4)
 * @param {boolean} symmetric - Symmetric quantization (default true)
 * @returns {{ quantized: number[], scale: number, zeroPoint: number }}
 */
export function quantize(values, bits = 8, symmetric = true) {
  // Handle scalar input — assume [-1, 1] range
  if (typeof values === 'number') {
    const levels = (1 << bits) - 1;
    const halfLevels = Math.floor(levels / 2);
    return Math.round(values * halfLevels + halfLevels);
  }
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
  
  return { quantized, scale, zeroPoint };
}

/**
 * Dequantize integer values back to floats.
 */
export function dequantize(quantized, scaleOrBits, zeroPoint) {
  // Handle scalar input with bits (dequantize(value, bits))
  if (typeof quantized === 'number' && zeroPoint === undefined) {
    // Reconstruct: value was quantized to [-1, 1] range with N bits
    const bits = scaleOrBits;
    const levels = (1 << bits) - 1;
    const halfLevels = Math.floor(levels / 2);
    return (quantized - halfLevels) / halfLevels;
  }
  return quantized.map(q => (q - zeroPoint) * scaleOrBits);
}

/**
 * Fake quantization: quantize then immediately dequantize (for QAT simulation).
 */
export function fakeQuantize(values, bits = 8) {
  const { quantized, scale, zeroPoint } = quantize(values, bits);
  return dequantize(quantized, scale, zeroPoint);
}

/**
 * Dynamic quantization: choose minimum bits to achieve target error.
 */
export function dynamicQuantize(values, targetError = 0.05) {
  const range = Math.max(...values) - Math.min(...values) || 1;
  for (let bits = 2; bits <= 16; bits++) {
    const fq = fakeQuantize(values, bits);
    const { rmse } = quantizationError(values, fq);
    if (rmse < targetError * range || bits === 16) {
      return { bits, quantized: quantize(values, bits), error: rmse };
    }
  }
  return { bits: 16, quantized: quantize(values, 16), error: 0 };
}

/**
 * Compute number of bits required to represent n levels.
 */
export function bitsRequired(n) {
  return Math.ceil(Math.log2(n));
}

/**
 * Quantize a Matrix's weights to N-bit.
 */
export function quantizeWeights(matrix, bits = 8) {
  const values = Array.from(matrix.data);
  const result = quantize(values, bits);
  return result;
}

/**
 * K-means weight clustering (codebook quantization).
 * @param {number[]} weights - Flat weight array
 * @param {number} k - Number of clusters
 * @param {number} iterations - K-means iterations
 */
export function clusterWeights(weights, k = 16, iterations = 20) {
  // Initialize centroids with k evenly spaced values in weight range
  const min = Math.min(...weights);
  const max = Math.max(...weights);
  let centroids = Array.from({ length: k }, (_, i) => min + (max - min) * (i + 0.5) / k);
  
  let assignments = new Array(weights.length).fill(0);
  
  for (let iter = 0; iter < iterations; iter++) {
    // Assign each weight to nearest centroid
    for (let i = 0; i < weights.length; i++) {
      let bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const d = Math.abs(weights[i] - centroids[c]);
        if (d < bestDist) { bestDist = d; assignments[i] = c; }
      }
    }
    // Update centroids
    const sums = new Array(k).fill(0);
    const counts = new Array(k).fill(0);
    for (let i = 0; i < weights.length; i++) {
      sums[assignments[i]] += weights[i];
      counts[assignments[i]]++;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) centroids[c] = sums[c] / counts[c];
    }
  }
  
  const quantized = assignments.map(a => centroids[a]);
  const codebookBits = Math.ceil(Math.log2(k));
  
  return {
    centroids,
    assignments,
    quantized,
    compressionRatio: codebookBits / 32,
  };
}
