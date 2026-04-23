// scaling-laws.js — Chinchilla scaling laws (Hoffmann et al., 2022)
// Predict optimal model size and token count for a given compute budget.

export function chinchillaOptimal(computeBudget) {
  // C = 6 * N * D where N = params, D = tokens
  // Optimal: N ∝ C^0.5, D ∝ C^0.5
  // Chinchilla: N ≈ (C/20)^(0.5), D ≈ 20 * N
  const optimalParams = Math.sqrt(computeBudget / 20);
  const optimalTokens = 20 * optimalParams;
  return { params: optimalParams, tokens: optimalTokens };
}

export function computeForTraining(params, tokens) {
  return 6 * params * tokens; // Approximate FLOPs
}

export function isOverTrained(params, tokens) {
  // Chinchilla ratio: tokens/params should be ~20
  return tokens / params > 20;
}

export function isUnderTrained(params, tokens) {
  return tokens / params < 20;
}
