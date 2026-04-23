// perplexity.js — Perplexity metric for language models
export function perplexity(logProbs) {
  // PPL = exp(-1/N * Σ log P(token_i))
  const avgLogProb = logProbs.reduce((s, lp) => s + lp, 0) / logProbs.length;
  return Math.exp(-avgLogProb);
}

export function bitsPerByte(logProbs, numBytes) {
  // BPB = -Σ log2 P(token_i) / numBytes
  const totalBits = logProbs.reduce((s, lp) => s + (-lp / Math.LN2), 0);
  return totalBits / numBytes;
}
