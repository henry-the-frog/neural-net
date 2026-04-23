// dpo.js — Direct Preference Optimization (Rafailov et al., 2023)
// Trains a language model on preference pairs without needing a reward model.
//
// Given preference pairs (x, y_w, y_l) where y_w is preferred over y_l:
// L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
//
// This implicitly fits a reward model within the policy optimization.

/**
 * Compute DPO loss for a batch of preference pairs.
 * @param {Array} batch - Array of { prompt, chosen, rejected } token sequences
 * @param {function} policyLogProbs - Current model: tokens → log probabilities
 * @param {function} refLogProbs - Reference model: tokens → log probabilities
 * @param {number} beta - Temperature parameter (default 0.1)
 * @returns {{ loss: number, chosenRewards: number[], rejectedRewards: number[], accuracy: number }}
 */
export function dpoLoss(batch, policyLogProbs, refLogProbs, beta = 0.1) {
  let totalLoss = 0;
  const chosenRewards = [];
  const rejectedRewards = [];
  let correct = 0;
  
  for (const { prompt, chosen, rejected } of batch) {
    // Log probabilities under policy
    const chosenTokens = [...prompt, ...chosen];
    const rejectedTokens = [...prompt, ...rejected];
    
    const policyChosenLP = policyLogProbs(chosenTokens);
    const policyRejectedLP = policyLogProbs(rejectedTokens);
    
    // Log probabilities under reference
    const refChosenLP = refLogProbs(chosenTokens);
    const refRejectedLP = refLogProbs(rejectedTokens);
    
    // Sum log probs over response tokens only (not prompt)
    let chosenLogRatio = 0, rejectedLogRatio = 0;
    for (let i = prompt.length; i < chosenTokens.length; i++) {
      chosenLogRatio += policyChosenLP[i] - refChosenLP[i];
    }
    for (let i = prompt.length; i < rejectedTokens.length; i++) {
      rejectedLogRatio += policyRejectedLP[i] - refRejectedLP[i];
    }
    
    // Implicit reward: β * log(π/π_ref)
    chosenRewards.push(beta * chosenLogRatio);
    rejectedRewards.push(beta * rejectedLogRatio);
    
    // DPO loss: -log σ(reward_w - reward_l)
    const diff = beta * (chosenLogRatio - rejectedLogRatio);
    const logSigmoid = diff >= 0 ? -Math.log(1 + Math.exp(-diff)) : diff - Math.log(1 + Math.exp(diff));
    totalLoss -= logSigmoid;
    
    // Accuracy: does the model prefer chosen over rejected?
    if (chosenLogRatio > rejectedLogRatio) correct++;
  }
  
  return {
    loss: totalLoss / batch.length,
    chosenRewards,
    rejectedRewards,
    accuracy: correct / batch.length,
    margin: chosenRewards.reduce((a, b) => a + b, 0) / batch.length - rejectedRewards.reduce((a, b) => a + b, 0) / batch.length,
  };
}

/**
 * Simplified DPO gradient for a single preference pair.
 * Returns the gradient multiplier for the policy model.
 */
export function dpoGradientMultiplier(chosenLogRatio, rejectedLogRatio, beta = 0.1) {
  const diff = beta * (chosenLogRatio - rejectedLogRatio);
  const sigmoid = 1 / (1 + Math.exp(diff)); // σ(-diff) = gradient multiplier
  return {
    chosenMultiplier: beta * sigmoid,    // Increase probability of chosen
    rejectedMultiplier: -beta * sigmoid,  // Decrease probability of rejected
  };
}
