// relative-position-bias.js — T5-style Relative Position Bias (Raffel et al., 2020)
// Instead of absolute positions, learn a bias table indexed by relative distance.
// Distances are bucketed logarithmically for efficiency.
// 
// Attention: score(i,j) = q_i^T k_j + bias(i-j)

import { Matrix } from './matrix.js';

/**
 * Compute relative position bucket (T5 bucketing scheme).
 * Bidirectional: bucket 0-7 for exact positions 0-7,
 * buckets 8-15 for logarithmically spaced larger distances.
 * @param {number} relativePos - Relative position (i - j)
 * @param {boolean} bidirectional - Whether to use separate buckets for negative distances
 * @param {number} numBuckets - Total number of buckets (default 32)
 * @param {number} maxDistance - Maximum distance to bucket (default 128)
 * @returns {number} Bucket index
 */
export function relativePosiBucket(relativePos, bidirectional = true, numBuckets = 32, maxDistance = 128) {
  let ret = 0;
  let n = -relativePos;
  
  if (bidirectional) {
    numBuckets = Math.floor(numBuckets / 2);
    ret += n < 0 ? numBuckets : 0;
    n = Math.abs(n);
  } else {
    n = Math.max(n, 0);
  }
  
  // Half of buckets are for exact positions
  const maxExact = Math.floor(numBuckets / 2);
  
  if (n < maxExact) {
    ret += n;
  } else {
    // Logarithmic bucketing for larger distances
    const logN = Math.log(n / maxExact) / Math.log(maxDistance / maxExact);
    const bucket = maxExact + Math.floor(logN * (numBuckets - maxExact));
    ret += Math.min(bucket, numBuckets - 1);
  }
  
  return ret;
}

/**
 * Compute relative position bias matrix.
 * @param {number} seqLen - Sequence length
 * @param {number} nHeads - Number of attention heads
 * @param {boolean} bidirectional
 * @param {number} numBuckets
 * @returns {{ biasMatrix: Matrix, biasTable: Matrix }}
 */
export function computeRelativePositionBias(seqLen, nHeads, bidirectional = true, numBuckets = 32) {
  // Learnable bias table: (numBuckets × nHeads)
  const biasTable = Matrix.random(numBuckets, nHeads).map(v => v * 0.02);
  
  // Compute bias for each (i, j) position pair, per head
  // Returns: (seqLen × seqLen) per head → we flatten to (seqLen × seqLen) averaging heads
  const biasMatrices = [];
  
  for (let h = 0; h < nHeads; h++) {
    const bias = new Matrix(seqLen, seqLen);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        const bucket = relativePosiBucket(i - j, bidirectional, numBuckets);
        bias.set(i, j, biasTable.get(bucket, h));
      }
    }
    biasMatrices.push(bias);
  }
  
  return { biasMatrices, biasTable };
}

/**
 * Apply relative position bias to attention scores.
 * @param {Matrix} scores - Attention scores (seqLen × seqLen)
 * @param {Matrix} bias - Position bias (seqLen × seqLen)
 * @returns {Matrix} Biased scores
 */
export function applyRelativePositionBias(scores, bias) {
  const result = new Matrix(scores.rows, scores.cols);
  for (let i = 0; i < scores.data.length; i++) {
    result.data[i] = scores.data[i] + bias.data[i];
  }
  return result;
}
