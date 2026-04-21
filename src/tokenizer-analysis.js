// tokenizer-analysis.js — Tokenizer Vocabulary and Compression Analysis
// Tools for understanding how BPE tokenization affects text.

import { BPETokenizer } from './bpe.js';

/**
 * Analyze tokenization statistics for a text.
 */
export function analyzeTokenization(tokenizer, text) {
  const tokens = tokenizer.encode(text);
  const chars = text.length;
  const numTokens = tokens.length;
  const avgCharsPerToken = chars / numTokens;

  // Token frequency distribution
  const freq = new Map();
  for (const t of tokens) {
    freq.set(t, (freq.get(t) || 0) + 1);
  }

  // Most common tokens
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1]);
  const topTokens = sorted.slice(0, 10).map(([id, count]) => ({
    id,
    token: tokenizer.idToToken[id],
    count,
    pct: (count / numTokens * 100).toFixed(1) + '%',
  }));

  return {
    characters: chars,
    tokens: numTokens,
    compressionRatio: (chars / numTokens).toFixed(2),
    uniqueTokens: freq.size,
    vocabUtilization: (freq.size / tokenizer.size * 100).toFixed(1) + '%',
    topTokens,
  };
}

/**
 * Analyze merge rules: which merges produce the biggest compression gains.
 */
export function analyzeMerges(tokenizer) {
  return tokenizer.merges.map((merge, idx) => ({
    step: idx + 1,
    pair: merge.pair.join(' + '),
    result: merge.merged,
    resultLength: merge.merged.length,
  }));
}

/**
 * Compare compression across different text types.
 */
export function compareTexts(tokenizer, texts) {
  return texts.map(({ label, text }) => {
    const analysis = analyzeTokenization(tokenizer, text);
    return {
      label,
      ...analysis,
    };
  });
}

/**
 * Estimate vocabulary coverage: what fraction of the input can be
 * encoded without falling back to unknown tokens.
 */
export function vocabularyCoverage(tokenizer, text) {
  const chars = new Set(text);
  let covered = 0;
  let uncovered = 0;
  for (const ch of chars) {
    if (tokenizer.vocab.has(ch)) covered++;
    else uncovered++;
  }
  return {
    totalUniqueChars: chars.size,
    covered,
    uncovered,
    coverage: (covered / chars.size * 100).toFixed(1) + '%',
  };
}
