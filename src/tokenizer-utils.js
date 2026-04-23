// tokenizer-utils.js — Common tokenizer utilities
// Shared helper functions for BPE, WordPiece, and other tokenizers

/**
 * Repetition penalty for language model logits (Keskar et al., 2019).
 * Reduces probability of tokens that already appear in the generated text.
 * @param {Float64Array} logits - Raw logits
 * @param {Array<number>} generatedTokens - Already generated token IDs
 * @param {number} penalty - Repetition penalty factor (1.0 = no penalty, >1 = penalize)
 * @returns {Float64Array} Modified logits
 */
export function applyRepetitionPenalty(logits, generatedTokens, penalty = 1.2) {
  const modified = new Float64Array(logits);
  const seen = new Set(generatedTokens);
  
  for (const tokenId of seen) {
    if (tokenId >= 0 && tokenId < modified.length) {
      if (modified[tokenId] > 0) {
        modified[tokenId] /= penalty;
      } else {
        modified[tokenId] *= penalty;
      }
    }
  }
  
  return modified;
}

/**
 * Frequency penalty: reduces logits proportional to token frequency.
 * @param {Float64Array} logits - Raw logits
 * @param {Array<number>} generatedTokens - Already generated tokens
 * @param {number} freqPenalty - Frequency penalty (0 = none, 1 = strong)
 * @param {number} presencePenalty - Presence penalty (0 = none, 1 = strong)
 * @returns {Float64Array} Modified logits
 */
export function applyFrequencyPenalty(logits, generatedTokens, freqPenalty = 0.5, presencePenalty = 0.5) {
  const modified = new Float64Array(logits);
  const freq = new Map();
  
  for (const t of generatedTokens) {
    freq.set(t, (freq.get(t) || 0) + 1);
  }
  
  for (const [tokenId, count] of freq) {
    if (tokenId >= 0 && tokenId < modified.length) {
      modified[tokenId] -= freqPenalty * count + presencePenalty;
    }
  }
  
  return modified;
}

/**
 * Temperature with min-p filtering (Together AI, 2023).
 * Filters out tokens whose probability is less than min_p * max_probability.
 * More principled than top-k for variable-confidence distributions.
 */
export function minPFilter(logits, minP = 0.05, temperature = 1.0) {
  const n = logits.length;
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
  
  const max = Math.max(...scaled);
  const probs = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    probs[i] = Math.exp(scaled[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  
  const maxProb = Math.max(...probs);
  const threshold = minP * maxProb;
  
  // Set logits of filtered tokens to -infinity
  const filtered = new Float64Array(logits);
  for (let i = 0; i < n; i++) {
    if (probs[i] < threshold) filtered[i] = -Infinity;
  }
  
  return filtered;
}
