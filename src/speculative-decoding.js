// speculative-decoding.js — Speculative Decoding (Leviathan et al., 2023)
// Uses a smaller "draft" model to generate K candidate tokens,
// then the larger "target" model verifies them all in a single forward pass.
// Accepted tokens don't need recomputation. Rejected tokens fall back to target.
//
// Speedup: ~2-3x for similar draft/target quality.
// Guarantee: exact same distribution as target-only generation.

/**
 * Speculative decoding step.
 * 
 * 1. Draft model generates K candidate tokens autoregressively
 * 2. Target model scores all K tokens in a single batch
 * 3. Accept/reject each token based on probability ratio
 * 4. Return accepted prefix + one target-sampled token
 * 
 * @param {function} draftForward - Draft model: tokens → logits for next token
 * @param {function} targetForward - Target model: tokens → logits for each position
 * @param {Array<number>} prompt - Input token sequence
 * @param {number} K - Number of speculative tokens
 * @param {number} temperature - Sampling temperature
 * @returns {{ tokens: number[], accepted: number, total: number }}
 */
export function speculativeDecodeStep(draftForward, targetForward, prompt, K = 4, temperature = 1.0) {
  const draftTokens = [];
  let currentPrompt = [...prompt];
  
  // Step 1: Draft model generates K candidate tokens
  for (let i = 0; i < K; i++) {
    const logits = draftForward(currentPrompt);
    const token = sample(logits, temperature);
    draftTokens.push(token);
    currentPrompt.push(token);
  }
  
  // Step 2: Target model scores all positions in one batch
  const fullSequence = [...prompt, ...draftTokens];
  const targetLogits = targetForward(fullSequence); // Returns logits for each position
  
  // Step 3: Accept/reject using modified rejection sampling
  const accepted = [];
  let allAccepted = true;
  
  for (let i = 0; i < K; i++) {
    const pos = prompt.length + i;
    const targetProbs = softmax(targetLogits[pos - 1], temperature);
    const draftProbs = softmax(draftForward([...prompt, ...draftTokens.slice(0, i)]), temperature);
    
    const draftToken = draftTokens[i];
    const pTarget = targetProbs[draftToken] || 1e-10;
    const pDraft = draftProbs[draftToken] || 1e-10;
    
    // Accept with probability min(1, p_target / p_draft)
    const acceptProb = Math.min(1, pTarget / pDraft);
    
    if (Math.random() < acceptProb) {
      accepted.push(draftToken);
    } else {
      // Reject: sample from adjusted target distribution
      // p_adjusted = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
      const adjustedProbs = new Float64Array(targetProbs.length);
      let adjSum = 0;
      for (let j = 0; j < targetProbs.length; j++) {
        adjustedProbs[j] = Math.max(0, targetProbs[j] - draftProbs[j]);
        adjSum += adjustedProbs[j];
      }
      if (adjSum > 0) {
        for (let j = 0; j < adjustedProbs.length; j++) adjustedProbs[j] /= adjSum;
      } else {
        // Fallback to target distribution
        for (let j = 0; j < targetProbs.length; j++) adjustedProbs[j] = targetProbs[j];
      }
      
      accepted.push(sampleFromProbs(adjustedProbs));
      allAccepted = false;
      break;
    }
  }
  
  // If all K tokens were accepted, sample one more from target
  if (allAccepted) {
    const finalProbs = softmax(targetLogits[fullSequence.length - 1], temperature);
    accepted.push(sampleFromProbs(finalProbs));
  }
  
  return {
    tokens: accepted,
    accepted: allAccepted ? K : accepted.length - 1,
    total: K,
    acceptanceRate: (allAccepted ? K : accepted.length - 1) / K,
  };
}

// --- Helper functions ---

function softmax(logits, temperature = 1.0) {
  if (!logits || logits.length === 0) return new Float64Array(0);
  const scaled = logits.map(l => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sample(logits, temperature = 1.0) {
  const probs = softmax(logits, temperature);
  return sampleFromProbs(probs);
}

function sampleFromProbs(probs) {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}
