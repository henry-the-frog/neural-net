// clip.js — CLIP Loss (Radford et al., 2021)
// Contrastive Language-Image Pre-training.
// Aligns text and image embeddings in a shared space.
//
// Given N (image, text) pairs, maximizes similarity of matching pairs
// and minimizes similarity of non-matching pairs.
// Loss is symmetric: both image→text and text→image cross-entropy.

import { cosineSimilarity } from './contrastive.js';

/**
 * CLIP contrastive loss.
 * @param {Array<Float64Array>} imageEmbs - Image embeddings (N × D)
 * @param {Array<Float64Array>} textEmbs - Text embeddings (N × D)
 * @param {number} temperature - Learned temperature (logit_scale = 1/τ)
 * @returns {{ loss: number, i2tAccuracy: number, t2iAccuracy: number }}
 */
export function clipLoss(imageEmbs, textEmbs, temperature = 0.07) {
  const N = imageEmbs.length;
  const logitScale = 1 / temperature;
  
  // Compute similarity matrix: N × N
  const logits = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      logits[i][j] = cosineSimilarity(imageEmbs[i], textEmbs[j]) * logitScale;
    }
  }
  
  // Image → Text cross-entropy (each row)
  let i2tLoss = 0;
  let i2tCorrect = 0;
  for (let i = 0; i < N; i++) {
    const { loss, isCorrect } = crossEntropyRow(logits[i], i);
    i2tLoss += loss;
    if (isCorrect) i2tCorrect++;
  }
  
  // Text → Image cross-entropy (each column)
  let t2iLoss = 0;
  let t2iCorrect = 0;
  for (let j = 0; j < N; j++) {
    const col = new Float64Array(N);
    for (let i = 0; i < N; i++) col[i] = logits[i][j];
    const { loss, isCorrect } = crossEntropyRow(col, j);
    t2iLoss += loss;
    if (isCorrect) t2iCorrect++;
  }
  
  return {
    loss: (i2tLoss + t2iLoss) / (2 * N),
    i2tAccuracy: i2tCorrect / N,
    t2iAccuracy: t2iCorrect / N,
  };
}

function crossEntropyRow(logits, target) {
  const N = logits.length;
  const max = Math.max(...logits);
  let sumExp = 0;
  for (let i = 0; i < N; i++) sumExp += Math.exp(logits[i] - max);
  const logSumExp = Math.log(sumExp) + max;
  
  const loss = -(logits[target] - logSumExp);
  
  let argmax = 0;
  for (let i = 1; i < N; i++) if (logits[i] > logits[argmax]) argmax = i;
  
  return { loss, isCorrect: argmax === target };
}

/**
 * Zero-shot classification using CLIP embeddings.
 * @param {Float64Array} imageEmb - Image embedding
 * @param {Array<Float64Array>} classEmbs - Text embeddings for each class
 * @returns {{ classIdx: number, scores: Float64Array }}
 */
export function zeroShotClassify(imageEmb, classEmbs) {
  const scores = new Float64Array(classEmbs.length);
  for (let i = 0; i < classEmbs.length; i++) {
    scores[i] = cosineSimilarity(imageEmb, classEmbs[i]);
  }
  
  let maxIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[maxIdx]) maxIdx = i;
  }
  
  return { classIdx: maxIdx, scores };
}
