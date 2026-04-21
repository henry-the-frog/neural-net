// beam-search.js — Beam Search for Text Generation
// Alternative to greedy/sampling: maintains K "beams" (hypotheses) at each step.
// Each beam is extended with the top-K tokens, then pruned to keep only
// the K highest-scoring overall sequences.
//
// Beam search often produces higher-quality output than greedy for tasks
// like translation, but can be repetitive for open-ended generation.

import { softmax } from './sampling.js';

/**
 * Beam search generation.
 *
 * @param {object} model - model with forward([tokenIds]) returning logits
 * @param {number[]} prompt - initial token IDs
 * @param {number} maxNewTokens - max tokens to generate
 * @param {number} beamWidth - number of beams to maintain
 * @param {number} vocabSize - vocabulary size
 * @param {object} [opts]
 * @param {number} [opts.lengthPenalty=1.0] - penalize shorter sequences (>1 favors longer)
 * @param {number} [opts.eosToken=-1] - end-of-sequence token
 * @returns {{ sequence: number[], score: number, allBeams: Array }}
 */
export function beamSearch(model, prompt, maxNewTokens, beamWidth, vocabSize, opts = {}) {
  const { lengthPenalty = 1.0, eosToken = -1 } = opts;

  // Initialize with single beam containing the prompt
  let beams = [{ tokens: [...prompt], score: 0, done: false }];

  for (let step = 0; step < maxNewTokens; step++) {
    const candidates = [];

    for (const beam of beams) {
      if (beam.done) {
        candidates.push(beam);
        continue;
      }

      // Get logits for the current beam's sequence
      const logits = model.forward([beam.tokens]);
      const lastPos = beam.tokens.length - 1;
      const posLogits = new Float64Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        posLogits[v] = logits.get(0, lastPos * vocabSize + v);
      }
      
      const logProbs = logSoftmax(posLogits);

      // Expand beam with top beamWidth tokens
      const topK = getTopK(logProbs, beamWidth);
      for (const { token, logProb } of topK) {
        const newTokens = [...beam.tokens, token];
        const newScore = beam.score + logProb;
        const done = token === eosToken;
        candidates.push({ tokens: newTokens, score: newScore, done });
      }
    }

    // Score with length penalty and select top beams
    beams = candidates
      .map(b => ({
        ...b,
        normalizedScore: b.score / Math.pow(b.tokens.length - prompt.length + 1, lengthPenalty),
      }))
      .sort((a, b) => b.normalizedScore - a.normalizedScore)
      .slice(0, beamWidth);

    // Early stop if all beams are done
    if (beams.every(b => b.done)) break;
  }

  // Return best beam
  const best = beams[0];
  return {
    sequence: best.tokens,
    score: best.score,
    normalizedScore: best.normalizedScore,
    allBeams: beams.map(b => ({
      tokens: b.tokens,
      score: b.score,
      length: b.tokens.length - prompt.length,
    })),
  };
}

/**
 * Log-softmax: log(softmax(x)) computed numerically stably.
 */
function logSoftmax(logits) {
  const max = Math.max(...logits);
  let logSumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    logSumExp += Math.exp(logits[i] - max);
  }
  logSumExp = Math.log(logSumExp) + max;

  const result = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    result[i] = logits[i] - logSumExp;
  }
  return result;
}

/**
 * Get top-K tokens by log probability.
 */
function getTopK(logProbs, k) {
  const indexed = Array.from(logProbs).map((lp, idx) => ({ token: idx, logProb: lp }));
  indexed.sort((a, b) => b.logProb - a.logProb);
  return indexed.slice(0, k);
}
