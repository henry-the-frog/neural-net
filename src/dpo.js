// dpo.js — Direct Preference Optimization (DPO)
// Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
// (Rafailov et al., 2023)
//
// DPO is a simpler alternative to RLHF (Reinforcement Learning from Human Feedback).
// Instead of training a separate reward model and using PPO, DPO directly optimizes
// the policy to prefer "chosen" responses over "rejected" responses.
//
// Loss = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
//
// where:
//   y_w = chosen (winning) response
//   y_l = rejected (losing) response
//   π_θ = current policy (model being trained)
//   π_ref = reference policy (frozen copy of initial model)
//   β = temperature parameter (controls deviation from reference)

import { softmax } from './sampling.js';

/**
 * Compute log probability of a sequence under a model.
 *
 * @param {object} model - model with forward() method
 * @param {number[]} prompt - prompt token IDs
 * @param {number[]} response - response token IDs
 * @param {number} vocabSize
 * @returns {number} sum of log probabilities
 */
export function computeLogProb(model, prompt, response, vocabSize) {
  const fullSeq = [...prompt, ...response];
  const logits = model.forward([fullSeq]);
  
  let logProb = 0;
  // Sum log P(token_t | token_0..token_{t-1}) for response tokens
  for (let i = 0; i < response.length; i++) {
    const pos = prompt.length + i - 1; // position of the context for this prediction
    if (pos < 0) continue; // skip if response starts at position 0
    
    const posLogits = new Float64Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      posLogits[v] = logits.get(0, pos * vocabSize + v);
    }
    const probs = softmax(posLogits);
    const targetToken = response[i];
    logProb += Math.log(Math.max(probs[targetToken], 1e-15));
  }
  
  return logProb;
}

/**
 * Compute DPO loss for a batch of preference pairs.
 *
 * @param {object} policy - current model being trained
 * @param {object} reference - frozen reference model
 * @param {Array<{prompt: number[], chosen: number[], rejected: number[]}>} batch
 * @param {number} vocabSize
 * @param {number} beta - temperature (default: 0.1)
 * @returns {{ loss: number, stats: object }}
 */
export function dpoLoss(policy, reference, batch, vocabSize, beta = 0.1) {
  let totalLoss = 0;
  let chosenWins = 0;
  const margins = [];

  for (const { prompt, chosen, rejected } of batch) {
    // Log probs under policy
    const policyChosenLP = computeLogProb(policy, prompt, chosen, vocabSize);
    const policyRejectedLP = computeLogProb(policy, prompt, rejected, vocabSize);
    
    // Log probs under reference
    const refChosenLP = computeLogProb(reference, prompt, chosen, vocabSize);
    const refRejectedLP = computeLogProb(reference, prompt, rejected, vocabSize);

    // Log-ratio differences
    const chosenLogRatio = policyChosenLP - refChosenLP;
    const rejectedLogRatio = policyRejectedLP - refRejectedLP;

    // DPO loss: -log σ(β * (chosen_ratio - rejected_ratio))
    const margin = beta * (chosenLogRatio - rejectedLogRatio);
    margins.push(margin);

    const loss = -Math.log(sigmoid(margin));
    totalLoss += loss;

    if (margin > 0) chosenWins++;
  }

  return {
    loss: totalLoss / batch.length,
    stats: {
      batchSize: batch.length,
      chosenWinRate: (chosenWins / batch.length * 100).toFixed(1) + '%',
      avgMargin: (margins.reduce((a, b) => a + b, 0) / margins.length).toFixed(4),
      beta,
    }
  };
}

/**
 * Compute reward from DPO-trained model.
 * DPO implicitly defines: r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
 */
export function implicitReward(policy, reference, prompt, response, vocabSize, beta = 0.1) {
  const policyLP = computeLogProb(policy, prompt, response, vocabSize);
  const refLP = computeLogProb(reference, prompt, response, vocabSize);
  return beta * (policyLP - refLP);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
