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
 * Compute quantization error (mean absolute error).
 */
export function quantizationError(original, dequantized) {
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
