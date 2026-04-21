// speculative-decoding.js — Speculative Decoding for LLM Inference
// Paper: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
//
// Key idea: A small, fast "draft" model generates K candidate tokens.
// The large "target" model verifies all K tokens in a single forward pass.
// Accepted tokens are free (we got K-token throughput from 1 target forward pass).
// Rejected tokens get re-sampled from the target distribution.
//
// Result: generates same distribution as the target model, but faster
// (when draft model acceptance rate is high enough).

import { softmax } from './sampling.js';

/**
 * Speculative decoding: generate tokens using draft+target model pair.
 *
 * @param {object} draft - draft model { forward(tokenIds) → logits matrix }
 * @param {object} target - target model { forward(tokenIds) → logits matrix }
 * @param {number[]} prompt - initial token IDs
 * @param {number} maxNewTokens - total new tokens to generate
 * @param {number} K - speculation length (tokens to draft per iteration)
 * @param {number} vocabSize - vocabulary size
 * @returns {{ tokens: number[], stats: object }}
 */
export function speculativeDecode(draft, target, prompt, maxNewTokens, K, vocabSize) {
  const tokens = [...prompt];
  let totalDraftForwards = 0;
  let totalTargetForwards = 0;
  let totalAccepted = 0;
  let totalDrafted = 0;

  while (tokens.length - prompt.length < maxNewTokens) {
    const remaining = maxNewTokens - (tokens.length - prompt.length);
    const specLen = Math.min(K, remaining);

    // Step 1: Draft model generates K candidate tokens (greedy for simplicity)
    const draftTokens = [];
    const draftLogits = []; // store draft logits for each position

    let draftInput = [...tokens];
    for (let i = 0; i < specLen; i++) {
      const logits = draft.forward([draftInput]);
      totalDraftForwards++;

      const lastPos = draftInput.length - 1;
      const posLogits = extractPositionLogits(logits, lastPos, vocabSize);
      draftLogits.push(posLogits);

      const nextToken = argmax(posLogits);
      draftTokens.push(nextToken);
      draftInput = [...draftInput, nextToken];
    }
    totalDrafted += specLen;

    // Step 2: Target model verifies all K+1 positions in ONE forward pass
    // (all tokens including the K drafted ones)
    const verifyInput = [...tokens, ...draftTokens];
    const targetLogits = target.forward([verifyInput]);
    totalTargetForwards++;

    // Step 3: Compare draft and target distributions, accept/reject
    let accepted = 0;
    for (let i = 0; i < specLen; i++) {
      const pos = tokens.length + i - 1; // position in the verify sequence
      const targetPos = extractPositionLogits(targetLogits, pos, vocabSize);
      const draftPos = draftLogits[i];

      const draftProbs = softmax(draftPos);
      const targetProbs = softmax(targetPos);

      const draftedToken = draftTokens[i];
      const acceptanceProb = Math.min(1, targetProbs[draftedToken] / Math.max(draftProbs[draftedToken], 1e-15));

      if (Math.random() < acceptanceProb) {
        // Accept this token
        tokens.push(draftedToken);
        accepted++;
      } else {
        // Reject: sample from adjusted distribution
        // p'(x) = max(0, p_target(x) - p_draft(x)) / Z
        const adjusted = new Float64Array(vocabSize);
        let sum = 0;
        for (let v = 0; v < vocabSize; v++) {
          adjusted[v] = Math.max(0, targetProbs[v] - draftProbs[v]);
          sum += adjusted[v];
        }
        if (sum > 0) {
          for (let v = 0; v < vocabSize; v++) adjusted[v] /= sum;
        } else {
          // Fallback to target distribution
          for (let v = 0; v < vocabSize; v++) adjusted[v] = targetProbs[v];
        }

        const sampled = sampleFromDist(adjusted);
        tokens.push(sampled);
        break; // Stop speculation at first rejection
      }
    }

    totalAccepted += accepted;

    // If all K were accepted, sample one more from the target's next position
    if (accepted === specLen && tokens.length - prompt.length < maxNewTokens) {
      const lastPos = tokens.length - 1;
      const nextLogits = extractPositionLogits(targetLogits, lastPos, vocabSize);
      const nextProbs = softmax(nextLogits);
      tokens.push(sampleFromDist(nextProbs));
    }
  }

  return {
    tokens: tokens.slice(0, prompt.length + maxNewTokens),
    stats: {
      draftForwards: totalDraftForwards,
      targetForwards: totalTargetForwards,
      totalAccepted,
      totalDrafted,
      acceptanceRate: totalDrafted > 0 ? (totalAccepted / totalDrafted * 100).toFixed(1) + '%' : '0%',
      // Speedup: without speculation, would need maxNewTokens target forwards
      // With speculation: totalTargetForwards target forwards
      speedup: totalTargetForwards > 0 ? (maxNewTokens / totalTargetForwards).toFixed(2) + 'x' : '1x',
    }
  };
}

// --- Helpers ---

function extractPositionLogits(logitsMatrix, pos, vocabSize) {
  const result = new Float64Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    result[v] = logitsMatrix.get(0, pos * vocabSize + v);
  }
  return result;
}

function argmax(arr) {
  let maxIdx = 0, maxVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
  }
  return maxIdx;
}

function sampleFromDist(probs) {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}
