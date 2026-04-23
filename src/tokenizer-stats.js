// tokenizer-stats.js — Tokenizer analysis and statistics
export function compressionRatio(text, tokenIds) {
  const textBytes = new TextEncoder().encode(text).length;
  return textBytes / tokenIds.length;
}

export function fertilities(texts, tokenizer) {
  // Average tokens per word
  return texts.map(text => {
    const words = text.split(/\s+/).filter(Boolean);
    const tokens = tokenizer(text);
    return tokens.length / Math.max(words.length, 1);
  });
}

export function tokenFrequencies(allTokenIds) {
  const freq = new Map();
  for (const id of allTokenIds) freq.set(id, (freq.get(id) || 0) + 1);
  return [...freq.entries()].sort((a, b) => b[1] - a[1]);
}
