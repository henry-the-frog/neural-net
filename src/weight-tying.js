// weight-tying.js — Weight Tying (Press & Wolf, 2017)
// Shares the embedding matrix with the output (language model head) matrix.
// This reduces parameters by vocabSize * dModel and often improves quality.
//
// Theory: the embedding matrix maps tokens to dModel space.
// The LM head maps dModel space back to token probabilities.
// These are approximately inverse operations → sharing makes sense.

import { Matrix } from './matrix.js';

/**
 * Create tied weight matrices for embedding and LM head.
 * @param {number} vocabSize - Vocabulary size
 * @param {number} dModel - Model dimension
 * @returns {{ embedding: Matrix, lmHead: function }}
 */
export function createTiedWeights(vocabSize, dModel) {
  const embedding = Matrix.random(vocabSize, dModel).map(v => v * 0.02);
  
  // LM head is transpose of embedding: logits = hidden @ embedding^T
  function lmHead(hidden) {
    // hidden: (seqLen × dModel), embedding: (vocabSize × dModel)
    // logits: (seqLen × vocabSize) = hidden @ embedding^T
    const seqLen = hidden.rows;
    const logits = new Matrix(seqLen, vocabSize);
    for (let i = 0; i < seqLen; i++) {
      for (let v = 0; v < vocabSize; v++) {
        let sum = 0;
        for (let d = 0; d < dModel; d++) {
          sum += hidden.get(i, d) * embedding.get(v, d);
        }
        logits.set(i, v, sum);
      }
    }
    return logits;
  }
  
  return { embedding, lmHead };
}

/**
 * Calculate parameter savings from weight tying.
 */
export function tyingSavings(vocabSize, dModel) {
  const untied = vocabSize * dModel * 2; // embedding + LM head
  const tied = vocabSize * dModel; // shared
  return {
    untied,
    tied,
    saved: untied - tied,
    savings: ((1 - tied / untied) * 100).toFixed(1) + '%',
  };
}
