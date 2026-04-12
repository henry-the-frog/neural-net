// sparse-attention.js — Efficient Attention Patterns for Long Sequences
// Implementations of various sparse attention masks and patterns

// ===== Attention Masks =====

// Full attention: every token attends to every token
export function fullAttentionMask(seqLen) {
  return Array.from({ length: seqLen }, () => new Array(seqLen).fill(true));
}

// Causal (autoregressive): token i can only attend to j where j <= i
export function causalMask(seqLen) {
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) => j <= i)
  );
}

// Local window: each token attends to w tokens on each side
export function localWindowMask(seqLen, windowSize) {
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) => Math.abs(i - j) <= windowSize)
  );
}

// Strided: attend to every k-th token
export function stridedMask(seqLen, stride) {
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) => j % stride === 0 || i === j)
  );
}

// Dilated: attend to positions at increasing distances (1, 2, 4, 8, ...)
export function dilatedMask(seqLen, maxDilation = null) {
  const maxD = maxDilation || Math.floor(Math.log2(seqLen));
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) => {
      if (i === j) return true;
      const diff = Math.abs(i - j);
      // Check if diff is a power of 2 up to maxDilation
      for (let d = 0; d <= maxD; d++) {
        if (diff === Math.pow(2, d)) return true;
      }
      return false;
    })
  );
}

// Global tokens: specified tokens attend to and are attended by all
export function globalTokenMask(seqLen, globalIndices) {
  const globalSet = new Set(globalIndices);
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) =>
      globalSet.has(i) || globalSet.has(j) || i === j
    )
  );
}

// Random: each token randomly attends to k other tokens
export function randomMask(seqLen, numRandom) {
  return Array.from({ length: seqLen }, (_, i) => {
    const row = new Array(seqLen).fill(false);
    row[i] = true; // Always attend to self
    const candidates = Array.from({ length: seqLen }, (_, j) => j).filter(j => j !== i);
    // Random sample
    for (let r = 0; r < Math.min(numRandom, candidates.length); r++) {
      const idx = Math.floor(Math.random() * candidates.length);
      row[candidates[idx]] = true;
      candidates.splice(idx, 1);
    }
    return row;
  });
}

// Combine masks with OR
export function combineMasks(...masks) {
  const seqLen = masks[0].length;
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) =>
      masks.some(m => m[i][j])
    )
  );
}

// Combine masks with AND
export function intersectMasks(...masks) {
  const seqLen = masks[0].length;
  return Array.from({ length: seqLen }, (_, i) =>
    Array.from({ length: seqLen }, (_, j) =>
      masks.every(m => m[i][j])
    )
  );
}

// ===== Longformer-style attention =====
// Combines: local window + global tokens + (optional) random
export function longformerMask(seqLen, { windowSize = 3, globalIndices = [0], numRandom = 0 } = {}) {
  const local = localWindowMask(seqLen, windowSize);
  const global = globalTokenMask(seqLen, globalIndices);
  const masks = [local, global];
  if (numRandom > 0) masks.push(randomMask(seqLen, numRandom));
  return combineMasks(...masks);
}

// BigBird-style: local + global + random
export function bigBirdMask(seqLen, { windowSize = 3, numGlobal = 2, numRandom = 3 } = {}) {
  const globalIndices = Array.from({ length: numGlobal }, (_, i) => i);
  return longformerMask(seqLen, { windowSize, globalIndices, numRandom });
}

// ===== Sparse Attention Computation =====

// Apply attention with mask
export function sparseAttention(queries, keys, values, mask, scale = null) {
  const seqLen = queries.length;
  const dim = queries[0].length;
  const sc = scale || 1 / Math.sqrt(dim);

  const output = [];
  const attentionWeights = [];

  for (let i = 0; i < seqLen; i++) {
    // Compute attention scores only for allowed positions
    const scores = new Array(seqLen).fill(-Infinity);
    for (let j = 0; j < seqLen; j++) {
      if (mask[i][j]) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += queries[i][d] * keys[j][d];
        scores[j] = dot * sc;
      }
    }

    // Softmax over valid positions
    const maxScore = Math.max(...scores.filter(s => s !== -Infinity));
    const exps = scores.map(s => s === -Infinity ? 0 : Math.exp(s - maxScore));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const weights = exps.map(e => sumExp > 0 ? e / sumExp : 0);
    attentionWeights.push(weights);

    // Weighted sum of values
    const out = new Array(dim).fill(0);
    for (let j = 0; j < seqLen; j++) {
      for (let d = 0; d < dim; d++) {
        out[d] += weights[j] * values[j][d];
      }
    }
    output.push(out);
  }

  return { output, weights: attentionWeights };
}

// ===== Mask Statistics =====
export function maskDensity(mask) {
  const total = mask.length * mask[0].length;
  const active = mask.flat().filter(Boolean).length;
  return active / total;
}

export function maskSparsity(mask) {
  return 1 - maskDensity(mask);
}

// Average number of attended positions per token
export function avgAttendedPositions(mask) {
  const perToken = mask.map(row => row.filter(Boolean).length);
  return perToken.reduce((a, b) => a + b, 0) / mask.length;
}
