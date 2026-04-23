// attention-masks.js — Attention Mask Utilities
// Common mask patterns used in transformers.

import { Matrix } from './matrix.js';

const NEG_INF = -1e9;

/**
 * Causal mask: attend only to past and current tokens.
 * Used in decoder-only models (GPT, LLaMA).
 */
export function causalMask(seqLen) {
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j <= i; j++) mask.set(i, j, 0);
    for (let j = i + 1; j < seqLen; j++) mask.set(i, j, NEG_INF);
  }
  return mask;
}

/**
 * Sliding window causal mask (Mistral-style).
 * Each token attends to at most `windowSize` past tokens.
 */
export function slidingWindowMask(seqLen, windowSize) {
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      if (j > i || i - j > windowSize) {
        mask.set(i, j, NEG_INF);
      } else {
        mask.set(i, j, 0);
      }
    }
  }
  return mask;
}

/**
 * Prefix mask: first `prefixLen` tokens can attend to each other (bidirectional),
 * remaining tokens use causal mask but can attend to prefix.
 * Used in prefix-tuning and some encoder-decoder setups.
 */
export function prefixMask(seqLen, prefixLen) {
  const mask = new Matrix(seqLen, seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      if (i < prefixLen && j < prefixLen) {
        mask.set(i, j, 0); // Prefix tokens: bidirectional
      } else if (i >= prefixLen && j <= i) {
        mask.set(i, j, 0); // Non-prefix: causal + prefix access
      } else {
        mask.set(i, j, NEG_INF);
      }
    }
  }
  return mask;
}

/**
 * Block-sparse mask: divide into blocks, each block attends to itself + adjacent blocks.
 * Used in BigBird, Longformer for O(n√n) attention.
 */
export function blockSparseMask(seqLen, blockSize) {
  const mask = new Matrix(seqLen, seqLen);
  mask.data.fill(NEG_INF);
  
  const nBlocks = Math.ceil(seqLen / blockSize);
  for (let bi = 0; bi < nBlocks; bi++) {
    for (let bj = Math.max(0, bi - 1); bj <= Math.min(nBlocks - 1, bi + 1); bj++) {
      for (let i = bi * blockSize; i < Math.min((bi + 1) * blockSize, seqLen); i++) {
        for (let j = bj * blockSize; j < Math.min((bj + 1) * blockSize, seqLen); j++) {
          mask.set(i, j, 0);
        }
      }
    }
  }
  
  return mask;
}

/**
 * Padding mask: mask out padding tokens.
 * @param {Array<number>} lengths - Actual lengths of each sequence
 * @param {number} maxLen - Maximum sequence length
 */
export function paddingMask(lengths, maxLen) {
  const batch = lengths.length;
  const masks = [];
  for (let b = 0; b < batch; b++) {
    const mask = new Float64Array(maxLen);
    for (let j = 0; j < maxLen; j++) {
      mask[j] = j < lengths[b] ? 0 : NEG_INF;
    }
    masks.push(mask);
  }
  return masks;
}

/**
 * Combine two masks (e.g., causal + padding).
 */
export function combineMasks(mask1, mask2) {
  const result = new Matrix(mask1.rows, mask1.cols);
  for (let i = 0; i < mask1.data.length; i++) {
    result.data[i] = Math.min(mask1.data[i] + mask2.data[i], 0);
  }
  return result;
}
