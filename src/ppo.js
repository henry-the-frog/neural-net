// ppo.js — Proximal Policy Optimization (Schulman et al., 2017)
// The core algorithm behind RLHF for language models.
//
// PPO clips the policy ratio to prevent too-large updates:
// L_CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
//
// For RLHF:
// reward = reward_model(response) - β * KL(π || π_ref)
// Advantage computed via GAE (Generalized Advantage Estimation)

/**
 * Compute clipped PPO surrogate objective.
 * @param {Float64Array} ratios - π(a|s)/π_old(a|s) probability ratios
 * @param {Float64Array} advantages - Advantage estimates A_t
 * @param {number} clipEpsilon - Clipping parameter (default 0.2)
 * @returns {{ loss: number, clipFraction: number }}
 */
export function ppoClipLoss(ratios, advantages, clipEpsilon = 0.2) {
  const n = ratios.length;
  let totalLoss = 0;
  let clipped = 0;
  
  for (let i = 0; i < n; i++) {
    const r = ratios[i];
    const a = advantages[i];
    
    const unclipped = r * a;
    const clippedR = Math.max(1 - clipEpsilon, Math.min(1 + clipEpsilon, r));
    const clippedObj = clippedR * a;
    
    // PPO uses min for maximization (we negate for loss minimization)
    totalLoss -= Math.min(unclipped, clippedObj);
    
    if (Math.abs(r - clippedR) > 1e-8) clipped++;
  }
  
  return {
    loss: totalLoss / n,
    clipFraction: clipped / n,
  };
}

/**
 * Generalized Advantage Estimation (GAE, Schulman et al. 2016).
 * Computes advantages from rewards and value estimates.
 * @param {Float64Array} rewards - Per-step rewards
 * @param {Float64Array} values - Per-step value estimates V(s)
 * @param {number} gamma - Discount factor (default 0.99)
 * @param {number} lambda - GAE lambda (default 0.95)
 * @returns {Float64Array} Advantages
 */
export function computeGAE(rewards, values, gamma = 0.99, lambda = 0.95) {
  const n = rewards.length;
  const advantages = new Float64Array(n);
  let lastGAE = 0;
  
  for (let t = n - 1; t >= 0; t--) {
    const nextValue = t < n - 1 ? values[t + 1] : 0;
    const delta = rewards[t] + gamma * nextValue - values[t];
    lastGAE = delta + gamma * lambda * lastGAE;
    advantages[t] = lastGAE;
  }
  
  return advantages;
}

/**
 * Compute KL divergence penalty between policy and reference.
 * @param {Float64Array} policyLogProbs - Log probs under current policy
 * @param {Float64Array} refLogProbs - Log probs under reference policy
 * @returns {number} Mean KL divergence
 */
export function klPenalty(policyLogProbs, refLogProbs) {
  let kl = 0;
  for (let i = 0; i < policyLogProbs.length; i++) {
    // KL(π || π_ref) ≈ log(π/π_ref) = log_π - log_π_ref
    kl += policyLogProbs[i] - refLogProbs[i];
  }
  return kl / policyLogProbs.length;
}

/**
 * Compute RLHF reward with KL penalty.
 * reward_rlhf = reward_model(response) - β * KL(π || π_ref)
 */
export function rlhfReward(rewardScore, policyLogProbs, refLogProbs, beta = 0.1) {
  const kl = klPenalty(policyLogProbs, refLogProbs);
  return rewardScore - beta * kl;
}

/**
 * Normalize advantages (zero mean, unit variance).
 */
export function normalizeAdvantages(advantages) {
  const n = advantages.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += advantages[i];
  mean /= n;
  
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (advantages[i] - mean) ** 2;
  variance /= n;
  const std = Math.sqrt(variance + 1e-8);
  
  const normalized = new Float64Array(n);
  for (let i = 0; i < n; i++) normalized[i] = (advantages[i] - mean) / std;
  return normalized;
}

/**
 * Value function loss (MSE between predicted and actual returns).
 */
export function valueLoss(values, returns) {
  let mse = 0;
  for (let i = 0; i < values.length; i++) {
    mse += (values[i] - returns[i]) ** 2;
  }
  return mse / values.length;
}

/**
 * Full PPO training step metrics.
 */
export function ppoStep(ratios, advantages, values, returns, clipEpsilon = 0.2, vfCoef = 0.5) {
  const { loss: policyLoss, clipFraction } = ppoClipLoss(ratios, advantages, clipEpsilon);
  const vLoss = valueLoss(values, returns);
  const totalLoss = policyLoss + vfCoef * vLoss;
  
  return {
    policyLoss,
    valueLoss: vLoss,
    totalLoss,
    clipFraction,
    approxKL: ratios.reduce((s, r) => s + (r - 1 - Math.log(r)), 0) / ratios.length,
  };
}
