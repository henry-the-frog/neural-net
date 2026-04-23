// training-utils.js — Gradient Clipping + Learning Rate Warmup Schedules
// Essential training utilities for stable LLM training.

/**
 * Gradient clipping by global norm (Pascanu et al., 2013).
 * Rescales all gradients so their global L2 norm ≤ maxNorm.
 * @param {Array<Float64Array>} gradients - List of gradient arrays
 * @param {number} maxNorm - Maximum allowed gradient norm
 * @returns {{ clipped: Array<Float64Array>, gradNorm: number, wasClipped: boolean }}
 */
export function clipGradNorm(gradients, maxNorm = 1.0) {
  // Compute global norm
  let totalNormSq = 0;
  for (const grad of gradients) {
    for (let i = 0; i < grad.length; i++) totalNormSq += grad[i] * grad[i];
  }
  const gradNorm = Math.sqrt(totalNormSq);
  
  const wasClipped = gradNorm > maxNorm;
  const scale = wasClipped ? maxNorm / (gradNorm + 1e-8) : 1.0;
  
  const clipped = gradients.map(grad => {
    const result = new Float64Array(grad.length);
    for (let i = 0; i < grad.length; i++) result[i] = grad[i] * scale;
    return result;
  });
  
  return { clipped, gradNorm, wasClipped };
}

/**
 * Gradient clipping by value.
 * Clips each gradient element to [-maxVal, maxVal].
 */
export function clipGradValue(gradients, maxVal = 1.0) {
  return gradients.map(grad => {
    const result = new Float64Array(grad.length);
    for (let i = 0; i < grad.length; i++) {
      result[i] = Math.max(-maxVal, Math.min(maxVal, grad[i]));
    }
    return result;
  });
}

/**
 * Linear warmup schedule.
 * LR linearly increases from 0 to baseLR over warmupSteps.
 * @param {number} step - Current step
 * @param {number} baseLR - Target learning rate
 * @param {number} warmupSteps - Number of warmup steps
 * @returns {number} Current learning rate
 */
export function linearWarmup(step, baseLR, warmupSteps) {
  if (step < warmupSteps) return baseLR * step / warmupSteps;
  return baseLR;
}

/**
 * Warmup + cosine decay schedule (used in LLaMA, GPT training).
 * Linear warmup → cosine decay to minLR.
 */
export function warmupCosineDecay(step, baseLR, warmupSteps, totalSteps, minLR = 0) {
  if (step < warmupSteps) {
    return baseLR * step / warmupSteps;
  }
  
  const progress = (step - warmupSteps) / (totalSteps - warmupSteps);
  const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)));
  return minLR + (baseLR - minLR) * cosineDecay;
}

/**
 * Warmup + inverse square root decay (Vaswani et al., 2017 — original Transformer).
 * lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
 */
export function warmupInvSqrt(step, dModel, warmupSteps) {
  const step1 = Math.max(step, 1); // Avoid division by 0
  return Math.pow(dModel, -0.5) * Math.min(Math.pow(step1, -0.5), step1 * Math.pow(warmupSteps, -1.5));
}

/**
 * Polynomial warmup + decay.
 */
export function warmupPolynomial(step, baseLR, warmupSteps, totalSteps, power = 1.0, endLR = 0) {
  if (step < warmupSteps) return baseLR * step / warmupSteps;
  const progress = Math.min((step - warmupSteps) / (totalSteps - warmupSteps), 1);
  return endLR + (baseLR - endLR) * Math.pow(1 - progress, power);
}
