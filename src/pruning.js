// pruning.js — Neural Network Pruning
// Remove unnecessary weights for efficient inference

function toArray(weights) {
  if (weights && weights.data instanceof Float64Array) {
    // Matrix: convert to flat array
    return Array.from(weights.data);
  }
  if (Array.isArray(weights[0])) return weights.flat();
  return Array.isArray(weights) ? weights : [...weights];
}

function isMatrix(w) { return w && w.data instanceof Float64Array && 'rows' in w && 'cols' in w; }

// ===== Magnitude Pruning =====
export function magnitudePrune(weights, sparsity = 0.5) {
  const flat = toArray(weights);
  const sorted = flat.map(Math.abs).sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * sparsity)];
  
  const prunedFlat = flat.map(w => Math.abs(w) >= threshold ? w : 0);
  
  if (isMatrix(weights)) {
    // Return Matrix-like with .data
    return { data: new Float64Array(prunedFlat), rows: weights.rows, cols: weights.cols, actualSparsity: prunedFlat.filter(w => w === 0).length / prunedFlat.length };
  }

  const mask = Array.isArray(weights[0])
    ? weights.map(row => row.map(w => Math.abs(w) > threshold ? 1 : 0))
    : flat.map(w => Math.abs(w) > threshold ? 1 : 0);

  const pruned = Array.isArray(weights[0])
    ? weights.map((row, i) => row.map((w, j) => w * mask[i][j]))
    : flat.map((w, i) => w * mask[i]);

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
  const flat = toArray(weights);
  const zeros = flat.filter(w => w === 0).length;
  return zeros / flat.length;
}

export function countNonZero(weights) {
  const flat = toArray(weights);
  return flat.filter(w => w !== 0).length;
}

export function compressionRatio(originalSize, sparsity) {
  return 1 / (1 - sparsity);
}

// Aliases and additions
export const sparsity = countSparsity;

export function randomPrune(weights, ratio = 0.5) {
  if (isMatrix(weights)) {
    const data = new Float64Array(weights.data.length);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() > ratio ? weights.data[i] : 0;
    return { data, rows: weights.rows, cols: weights.cols, actualSparsity: Array.from(data).filter(w => w === 0).length / data.length };
  }
  if (Array.isArray(weights[0])) {
    const pruned = weights.map(row => row.map(w => Math.random() > ratio ? w : 0));
    return { pruned, actualSparsity: countSparsity(pruned) };
  }
  const pruned = weights.map(w => Math.random() > ratio ? w : 0);
  return { pruned, actualSparsity: countSparsity(pruned) };
}

export function gradualPrune(weights, currentSparsity, targetSparsity, step, totalSteps) {
  const s = currentSparsity + (targetSparsity - currentSparsity) * Math.min(1, step / totalSteps);
  return magnitudePrune(weights, s);
}

export class StructuredPruner {
  constructor(sparsity = 0.3) { this.sparsity = sparsity; }
  prune(weights) { return structuredPrune(weights, this.sparsity); }
}
