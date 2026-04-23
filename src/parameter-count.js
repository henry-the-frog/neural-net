// parameter-count.js — Count parameters in neural network architectures
export function transformerParams({ vocabSize, dModel, nHeads, nLayers, dFF, tiedWeights = false }) {
  const embedding = vocabSize * dModel;
  const lmHead = tiedWeights ? 0 : vocabSize * dModel;
  const attention = nLayers * (4 * dModel * dModel); // Q, K, V, O
  const ffn = nLayers * (2 * dModel * dFF + dFF + dModel); // 2 linear layers + biases
  const norm = nLayers * (2 * dModel); // 2 norms per layer
  return embedding + lmHead + attention + ffn + norm;
}

export function formatParams(count) {
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return `${count}`;
}
