// top-p-sampling.js — Nucleus (Top-p) Sampling (Holtzman et al., 2020)
// Instead of sampling from all tokens (random) or only top-k tokens,
// sample from the smallest set of tokens whose cumulative probability exceeds p.
// This dynamically adjusts the number of candidate tokens based on confidence.

/**
 * Top-p (nucleus) sampling.
 * @param {Float64Array|Array<number>} logits - Raw logits
 * @param {number} p - Cumulative probability threshold (typically 0.9-0.95)
 * @param {number} temperature - Sampling temperature
 * @returns {number} Sampled token index
 */
export function topPSample(logits, p = 0.9, temperature = 1.0) {
  const n = logits.length;
  
  // Apply temperature
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
  
  // Softmax
  const max = Math.max(...scaled);
  const probs = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    probs[i] = Math.exp(scaled[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  
  // Sort by probability (descending)
  const indexed = Array.from(probs).map((prob, idx) => ({ prob, idx }));
  indexed.sort((a, b) => b.prob - a.prob);
  
  // Find nucleus: smallest set with cumulative prob >= p
  let cumulative = 0;
  const nucleus = [];
  for (const item of indexed) {
    nucleus.push(item);
    cumulative += item.prob;
    if (cumulative >= p) break;
  }
  
  // Renormalize within nucleus
  const nucleusSum = nucleus.reduce((s, item) => s + item.prob, 0);
  
  // Sample from nucleus
  const r = Math.random() * nucleusSum;
  let acc = 0;
  for (const item of nucleus) {
    acc += item.prob;
    if (r < acc) return item.idx;
  }
  return nucleus[nucleus.length - 1].idx;
}

/**
 * Top-k sampling.
 * @param {Float64Array|Array<number>} logits - Raw logits
 * @param {number} k - Number of top tokens to consider
 * @param {number} temperature - Sampling temperature
 * @returns {number} Sampled token index
 */
export function topKSample(logits, k = 40, temperature = 1.0) {
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
  
  const indexed = Array.from(probs).map((prob, idx) => ({ prob, idx }));
  indexed.sort((a, b) => b.prob - a.prob);
  const topK = indexed.slice(0, Math.min(k, n));
  
  const topSum = topK.reduce((s, item) => s + item.prob, 0);
  const r = Math.random() * topSum;
  let acc = 0;
  for (const item of topK) {
    acc += item.prob;
    if (r < acc) return item.idx;
  }
  return topK[topK.length - 1].idx;
}

/**
 * Combined top-k + top-p sampling (used by many LLMs).
 * First apply top-k filter, then apply top-p within the top-k set.
 */
export function topKPSample(logits, k = 40, p = 0.9, temperature = 1.0) {
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
  
  // Top-k filter
  const indexed = Array.from(probs).map((prob, idx) => ({ prob, idx }));
  indexed.sort((a, b) => b.prob - a.prob);
  const topK = indexed.slice(0, Math.min(k, n));
  
  // Top-p filter within top-k
  let cumulative = 0;
  const nucleus = [];
  for (const item of topK) {
    nucleus.push(item);
    cumulative += item.prob;
    if (cumulative >= p) break;
  }
  
  const nucleusSum = nucleus.reduce((s, item) => s + item.prob, 0);
  const r = Math.random() * nucleusSum;
  let acc = 0;
  for (const item of nucleus) {
    acc += item.prob;
    if (r < acc) return item.idx;
  }
  return nucleus[nucleus.length - 1].idx;
}
