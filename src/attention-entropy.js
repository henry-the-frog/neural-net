// attention-entropy.js — Attention entropy analysis
// Measures how "focused" or "diffuse" attention patterns are.

export function attentionEntropy(weights) {
  // weights: array of attention probabilities (should sum to 1)
  let entropy = 0;
  for (const w of weights) {
    if (w > 1e-10) entropy -= w * Math.log2(w);
  }
  return entropy;
}

export function averageAttentionEntropy(attnMatrix) {
  // attnMatrix: 2D array [seqLen][seqLen] of attention weights
  let totalEntropy = 0;
  for (const row of attnMatrix) {
    totalEntropy += attentionEntropy(row);
  }
  return totalEntropy / attnMatrix.length;
}

// Maximum possible entropy for seqLen tokens
export function maxEntropy(seqLen) {
  return Math.log2(seqLen);
}

// Attention head redundancy: measure similarity between heads
export function headRedundancy(heads) {
  // heads: array of attention matrices (one per head)
  let totalSim = 0, count = 0;
  for (let i = 0; i < heads.length; i++) {
    for (let j = i + 1; j < heads.length; j++) {
      totalSim += cosineSim(heads[i].flat(), heads[j].flat());
      count++;
    }
  }
  return count > 0 ? totalSim / count : 0;
}

function cosineSim(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA * normB) + 1e-8);
}
