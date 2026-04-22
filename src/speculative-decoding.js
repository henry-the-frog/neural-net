// speculative-decoding.js — Speculative Decoding (Leviathan et al., 2022)
//
// Speed up autoregressive generation by:
// 1. Draft model generates γ tokens quickly (small, fast model)
// 2. Target model verifies all γ tokens in parallel (single forward pass)
// 3. Accept matching tokens, reject where draft diverges
//
// Expected speedup: γ * (1 - rejection_rate) ≈ 2-3x for well-matched draft/target
//
// This is a generic implementation that works with any forward() function.

import { Matrix } from './matrix.js';

/**
 * Speculative decoding step.
 * 
 * @param {Function} draftForward — draft model forward: (tokens) → logits [seqLen, vocabSize]
 * @param {Function} targetForward — target model forward: (tokens) → logits [seqLen, vocabSize]
 * @param {number[]} prefix — current token sequence
 * @param {number} gamma — number of speculative tokens
 * @param {Function} [sample] — sampling function: (logits) → tokenId
 * @returns {{ tokens: number[], accepted: number, rejected: boolean }}
 */
export function speculativeStep(draftForward, targetForward, prefix, gamma, sample = argmax) {
  // 1. Draft model generates γ tokens autoregressively
  const draftTokens = [];
  let draftPrefix = [...prefix];
  
  for (let i = 0; i < gamma; i++) {
    const logits = draftForward(draftPrefix);
    const lastLogits = getLastRow(logits);
    const token = sample(lastLogits);
    draftTokens.push(token);
    draftPrefix.push(token);
  }
  
  // 2. Target model verifies all tokens in one forward pass
  const verifyPrefix = [...prefix, ...draftTokens];
  const targetLogits = targetForward(verifyPrefix);
  
  // 3. Check which draft tokens match target's distribution
  const accepted = [];
  let allAccepted = true;
  
  for (let i = 0; i < gamma; i++) {
    const targetIdx = prefix.length + i; // Position where target predicts token i+1
    const targetRow = getRow(targetLogits, targetIdx);
    const targetToken = sample(targetRow);
    
    if (targetToken === draftTokens[i]) {
      accepted.push(draftTokens[i]);
    } else {
      // Reject this and all subsequent draft tokens
      // Use target's prediction instead
      accepted.push(targetToken);
      allAccepted = false;
      break;
    }
  }
  
  // If all draft tokens accepted, also get target's prediction for next token
  if (allAccepted) {
    const lastRow = getRow(targetLogits, prefix.length + gamma - 1);
    if (lastRow) {
      accepted.push(sample(lastRow));
    }
  }
  
  return {
    tokens: accepted,
    accepted: allAccepted ? gamma : accepted.length - 1, // -1 because we added target's correction
    totalGenerated: accepted.length,
    rejectedAt: allAccepted ? -1 : accepted.length - 1,
  };
}

/**
 * Full speculative generation loop.
 */
export function speculativeGenerate(draftForward, targetForward, prefix, maxTokens, gamma = 4, sample = argmax) {
  const generated = [...prefix];
  let totalAccepted = 0;
  let totalSteps = 0;
  
  while (generated.length < prefix.length + maxTokens) {
    const result = speculativeStep(
      draftForward, targetForward, generated, gamma, sample
    );
    
    for (const token of result.tokens) {
      if (generated.length >= prefix.length + maxTokens) break;
      generated.push(token);
    }
    
    totalAccepted += result.accepted;
    totalSteps++;
  }
  
  return {
    tokens: generated.slice(prefix.length),
    totalSteps, // Number of target model forward passes
    totalAccepted,
    avgAcceptance: totalAccepted / totalSteps,
    speedup: generated.length / totalSteps, // Tokens per target forward pass
  };
}

function argmax(logits) {
  if (logits instanceof Matrix) {
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < logits.cols; i++) {
      if (logits.get(0, i) > maxVal) { maxVal = logits.get(0, i); maxIdx = i; }
    }
    return maxIdx;
  }
  if (Array.isArray(logits) || logits instanceof Float64Array) {
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
    }
    return maxIdx;
  }
  return 0;
}

function getLastRow(logits) {
  if (logits instanceof Matrix) {
    const row = new Matrix(1, logits.cols);
    for (let j = 0; j < logits.cols; j++) row.set(0, j, logits.get(logits.rows - 1, j));
    return row;
  }
  return logits;
}

function getRow(logits, idx) {
  if (logits instanceof Matrix && idx < logits.rows) {
    const row = new Matrix(1, logits.cols);
    for (let j = 0; j < logits.cols; j++) row.set(0, j, logits.get(idx, j));
    return row;
  }
  return null;
}
