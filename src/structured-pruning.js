// structured-pruning.js — Model pruning techniques
export function magnitudePrune(weights, sparsity = 0.5) {
  const sorted = [...weights].map(Math.abs).sort((a, b) => a - b);
  const threshold = sorted[Math.floor(weights.length * sparsity)];
  return weights.map(w => Math.abs(w) < threshold ? 0 : w);
}

export function computeSparsity(weights) {
  const zeros = weights.filter(w => w === 0).length;
  return zeros / weights.length;
}

export function topKSparsify(weights, k) {
  const indexed = weights.map((w, i) => ({ w: Math.abs(w), i }));
  indexed.sort((a, b) => b.w - a.w);
  const topIndices = new Set(indexed.slice(0, k).map(x => x.i));
  return weights.map((w, i) => topIndices.has(i) ? w : 0);
}
