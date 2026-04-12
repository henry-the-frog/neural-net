// pruning.js — Neural Network Pruning
// Remove unnecessary weights for efficient inference

// ===== Magnitude Pruning =====
// Remove weights below threshold (absolute value)
export function magnitudePrune(weights, sparsity = 0.5) {
  const flat = weights.flat ? weights.flat() : [...weights];
  const sorted = flat.map(Math.abs).sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * sparsity)];

  const mask = Array.isArray(weights[0])
    ? weights.map(row => row.map(w => Math.abs(w) > threshold ? 1 : 0))
    : weights.map(w => Math.abs(w) > threshold ? 1 : 0);

  const pruned = Array.isArray(weights[0])
    ? weights.map((row, i) => row.map((w, j) => w * mask[i][j]))
    : weights.map((w, i) => w * mask[i]);

  return { pruned, mask, threshold, actualSparsity: countSparsity(pruned) };
}

// ===== Structured Pruning =====
// Remove entire neurons/channels based on L1/L2 norm
export function structuredPrune(weightMatrix, sparsity = 0.3, norm = 'l1') {
  const norms = weightMatrix.map(row => {
    if (norm === 'l1') return row.reduce((s, v) => s + Math.abs(v), 0);
    return Math.sqrt(row.reduce((s, v) => s + v * v, 0));
  });

  const sorted = [...norms].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(norms.length * sparsity)];

  const mask = norms.map(n => n > threshold ? 1 : 0);
  const pruned = weightMatrix.map((row, i) => mask[i] ? [...row] : row.map(() => 0));

  return { pruned, mask, threshold, removedChannels: mask.filter(m => m === 0).length };
}

// ===== Lottery Ticket Hypothesis =====
// Find winning ticket: train → prune → reset to initial weights → retrain
export function findWinningTicket(initialWeights, trainedWeights, sparsity = 0.5) {
  // Get pruning mask from trained weights
  const { mask } = magnitudePrune(trainedWeights, sparsity);

  // Apply mask to INITIAL weights (the lottery ticket)
  const ticket = Array.isArray(initialWeights[0])
    ? initialWeights.map((row, i) => row.map((w, j) => w * mask[i][j]))
    : initialWeights.map((w, i) => w * mask[i]);

  return { ticket, mask, sparsity };
}

// ===== Gradual Pruning Schedule =====
// Gradually increase sparsity during training
export function pruningSchedule(initialSparsity, targetSparsity, totalSteps, currentStep) {
  if (currentStep >= totalSteps) return targetSparsity;
  // Cubic schedule (Zhu & Gupta, 2017)
  const progress = currentStep / totalSteps;
  return targetSparsity + (initialSparsity - targetSparsity) * Math.pow(1 - progress, 3);
}

// ===== Utility =====
export function countSparsity(weights) {
  const flat = Array.isArray(weights[0]) ? weights.flat() : weights;
  const zeros = flat.filter(w => w === 0).length;
  return zeros / flat.length;
}

export function countNonZero(weights) {
  const flat = Array.isArray(weights[0]) ? weights.flat() : weights;
  return flat.filter(w => w !== 0).length;
}

export function compressionRatio(originalSize, sparsity) {
  return 1 / (1 - sparsity);
}
