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
export function quantizePerChannel(weights) {
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

export function dequantizePerChannel({ quantized, scales, shape }) {
  const result = new Matrix(shape[0], shape[1]);
  const cols = shape[1];
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < cols; j++) {
      result.set(i, j, quantized[i * cols + j] * scales[i]);
    }
  }
  return result;
}

/**
 * Compute quantization error metrics.
 */
export function quantizationError(original, dequantized) {
  let mse = 0, maxError = 0;
  for (let i = 0; i < original.data.length; i++) {
    const err = Math.abs(original.data[i] - dequantized.data[i]);
    mse += err * err;
    maxError = Math.max(maxError, err);
  }
  mse /= original.data.length;
  return {
    mse,
    rmse: Math.sqrt(mse),
    maxError,
    snr: 10 * Math.log10(original.data.reduce((s, v) => s + v * v, 0) / original.data.length / mse),
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
