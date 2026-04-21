// sampling.js — Token Sampling Strategies for Language Model Generation
// Implements: greedy, temperature, top-k, top-p (nucleus), and combined sampling

/**
 * Apply temperature scaling to logits.
 * Higher temperature → more random (flatter distribution)
 * Lower temperature → more deterministic (sharper distribution)
 * Temperature = 1.0 → unchanged
 *
 * @param {Float64Array|number[]} logits
 * @param {number} temperature
 * @returns {Float64Array}
 */
export function applyTemperature(logits, temperature) {
  if (temperature <= 0) throw new Error('Temperature must be > 0');
  if (temperature === 1.0) return Float64Array.from(logits);
  const result = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    result[i] = logits[i] / temperature;
  }
  return result;
}

/**
 * Convert logits to probabilities via softmax.
 * @param {Float64Array|number[]} logits
 * @returns {Float64Array} probabilities (sum to 1)
 */
export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

/**
 * Top-k filtering: keep only the k highest-probability tokens, zero out the rest.
 * @param {Float64Array} logits
 * @param {number} k - number of tokens to keep
 * @returns {Float64Array} filtered logits (non-kept set to -Infinity)
 */
export function topK(logits, k) {
  if (k >= logits.length) return Float64Array.from(logits);
  if (k <= 0) throw new Error('k must be > 0');

  // Find the k-th largest value
  const sorted = Array.from(logits).sort((a, b) => b - a);
  const threshold = sorted[k - 1];

  const result = new Float64Array(logits.length);
  let count = 0;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] >= threshold && count < k) {
      result[i] = logits[i];
      count++;
    } else {
      result[i] = -Infinity;
    }
  }
  return result;
}

/**
 * Top-p (nucleus) sampling: keep the smallest set of tokens whose cumulative
 * probability exceeds p.
 *
 * @param {Float64Array} logits
 * @param {number} p - cumulative probability threshold (0.0 to 1.0)
 * @returns {Float64Array} filtered logits
 */
export function topP(logits, p) {
  if (p >= 1.0) return Float64Array.from(logits);
  if (p <= 0) throw new Error('p must be > 0');

  const probs = softmax(logits);

  // Sort by probability descending, track indices
  const indexed = Array.from(probs).map((prob, idx) => ({ prob, idx }));
  indexed.sort((a, b) => b.prob - a.prob);

  // Find cutoff: smallest set summing to >= p
  let cumulative = 0;
  const keep = new Set();
  for (const { prob, idx } of indexed) {
    keep.add(idx);
    cumulative += prob;
    if (cumulative >= p) break;
  }

  const result = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    result[i] = keep.has(i) ? logits[i] : -Infinity;
  }
  return result;
}

/**
 * Sample from a probability distribution.
 * @param {Float64Array} probs - probability distribution (must sum to ~1)
 * @returns {number} sampled index
 */
export function sampleFromProbs(probs) {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1; // numerical safety
}

/**
 * Greedy decoding: return the token with highest logit.
 * @param {Float64Array|number[]} logits
 * @returns {number} token index
 */
export function greedySample(logits) {
  let maxIdx = 0, maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
  }
  return maxIdx;
}

/**
 * Combined sampling: apply temperature → top-k → top-p → sample.
 * This is how most modern LLM inference works.
 *
 * @param {Float64Array|number[]} logits - raw logits from model
 * @param {object} opts
 * @param {number} [opts.temperature=1.0] - temperature scaling
 * @param {number} [opts.topK=0] - top-k filter (0 = disabled)
 * @param {number} [opts.topP=1.0] - top-p filter (1.0 = disabled)
 * @param {boolean} [opts.greedy=false] - force greedy decoding
 * @returns {number} sampled token index
 */
export function sample(logits, { temperature = 1.0, topK: k = 0, topP: p = 1.0, greedy = false } = {}) {
  if (greedy) return greedySample(logits);

  let filtered = applyTemperature(logits, temperature);
  if (k > 0) filtered = topK(filtered, k);
  if (p < 1.0) filtered = topP(filtered, p);

  const probs = softmax(filtered);
  return sampleFromProbs(probs);
}

/**
 * Repetition penalty: reduce logits of tokens that appear in the context.
 * Used to prevent repetitive generation.
 *
 * @param {Float64Array|number[]} logits
 * @param {number[]} context - recent token IDs
 * @param {number} penalty - penalty factor (> 1.0 reduces, 1.0 = no penalty)
 * @returns {Float64Array}
 */
export function applyRepetitionPenalty(logits, context, penalty = 1.2) {
  if (penalty === 1.0) return Float64Array.from(logits);
  const result = Float64Array.from(logits);
  const seen = new Set(context);
  for (const tokenId of seen) {
    if (tokenId < result.length) {
      if (result[tokenId] > 0) {
        result[tokenId] /= penalty;
      } else {
        result[tokenId] *= penalty;
      }
    }
  }
  return result;
}
