// perplexity.js — Perplexity Evaluation for Language Models
// Perplexity is THE standard metric for language models.
// PPL = exp(-1/N * Σ log P(token_i | context))
// Lower is better. Random model on vocab V has PPL = V.

import { softmax } from './sampling.js';

/**
 * Compute perplexity of a model on a sequence.
 *
 * @param {object} model - model with forward([tokenIds]) → logits
 * @param {number[]} sequence - token IDs to evaluate
 * @param {number} vocabSize
 * @returns {{ perplexity: number, avgLogProb: number, numTokens: number }}
 */
export function computePerplexity(model, sequence, vocabSize) {
  if (sequence.length < 2) return { perplexity: Infinity, avgLogProb: -Infinity, numTokens: 0 };

  const logits = model.forward([sequence]);
  let totalLogProb = 0;
  let numTokens = 0;

  // For each position, compute log P(next_token | prefix)
  for (let t = 0; t < sequence.length - 1; t++) {
    const posLogits = new Float64Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      posLogits[v] = logits.get(0, t * vocabSize + v);
    }
    const probs = softmax(posLogits);
    const targetToken = sequence[t + 1];
    totalLogProb += Math.log(Math.max(probs[targetToken], 1e-15));
    numTokens++;
  }

  const avgLogProb = totalLogProb / numTokens;
  const perplexity = Math.exp(-avgLogProb);

  return { perplexity, avgLogProb, numTokens };
}

/**
 * Compute perplexity over multiple sequences (corpus-level).
 */
export function corpusPerplexity(model, sequences, vocabSize) {
  let totalLogProb = 0;
  let totalTokens = 0;

  for (const seq of sequences) {
    const result = computePerplexity(model, seq, vocabSize);
    totalLogProb += result.avgLogProb * result.numTokens;
    totalTokens += result.numTokens;
  }

  const avgLogProb = totalTokens > 0 ? totalLogProb / totalTokens : -Infinity;
  return {
    perplexity: Math.exp(-avgLogProb),
    avgLogProb,
    totalTokens,
    numSequences: sequences.length,
  };
}

/**
 * Compare two models on the same data.
 */
export function compareModels(model1, model2, sequences, vocabSize) {
  const ppl1 = corpusPerplexity(model1, sequences, vocabSize);
  const ppl2 = corpusPerplexity(model2, sequences, vocabSize);

  return {
    model1: ppl1,
    model2: ppl2,
    winner: ppl1.perplexity < ppl2.perplexity ? 'model1' : 'model2',
    improvement: ((ppl1.perplexity - ppl2.perplexity) / ppl1.perplexity * 100).toFixed(1) + '%',
  };
}

/**
 * Theoretical baseline perplexities.
 */
export function theoreticalBaselines(vocabSize) {
  return {
    random: vocabSize,           // PPL = V for uniform random
    unigram: Math.sqrt(vocabSize), // rough estimate for unigram model
    perfect: 1.0,                // PPL = 1 for perfect prediction
  };
}
