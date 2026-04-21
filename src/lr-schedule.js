// lr-schedule.js — Learning Rate Schedules for LLM Training
// Standard practice: linear warmup → cosine decay → min_lr

/**
 * Cosine learning rate schedule with linear warmup.
 * Used by GPT-3, Llama, Chinchilla, etc.
 *
 * @param {number} step - current training step
 * @param {number} maxLr - peak learning rate
 * @param {number} minLr - minimum learning rate (typically 0.1 * maxLr)
 * @param {number} warmupSteps - steps for linear warmup
 * @param {number} totalSteps - total training steps
 * @returns {number} learning rate for this step
 */
export function cosineWithWarmup(step, maxLr, minLr, warmupSteps, totalSteps) {
  if (step < warmupSteps) {
    // Linear warmup
    return maxLr * (step + 1) / warmupSteps;
  }

  if (step >= totalSteps) return minLr;

  // Cosine decay
  const progress = (step - warmupSteps) / (totalSteps - warmupSteps);
  return minLr + 0.5 * (maxLr - minLr) * (1 + Math.cos(Math.PI * progress));
}

/**
 * Generate the full learning rate schedule.
 */
export function generateSchedule(maxLr, minLr, warmupSteps, totalSteps) {
  const schedule = [];
  for (let step = 0; step < totalSteps; step++) {
    schedule.push(cosineWithWarmup(step, maxLr, minLr, warmupSteps, totalSteps));
  }
  return schedule;
}

/**
 * Warmup-Stable-Decay (WSD) schedule.
 * Used by some newer models as an alternative to cosine.
 */
export function warmupStableDecay(step, maxLr, minLr, warmupSteps, stableSteps, totalSteps) {
  if (step < warmupSteps) {
    return maxLr * (step + 1) / warmupSteps;
  }
  if (step < warmupSteps + stableSteps) {
    return maxLr;
  }
  const decaySteps = totalSteps - warmupSteps - stableSteps;
  const progress = (step - warmupSteps - stableSteps) / decaySteps;
  return maxLr - (maxLr - minLr) * progress;
}

/**
 * Common LLM training configurations.
 */
export const presets = {
  gpt3: { maxLr: 6e-4, minLr: 6e-5, warmupSteps: 375, totalSteps: 300000 },
  llama2_7b: { maxLr: 3e-4, minLr: 3e-5, warmupSteps: 2000, totalSteps: 250000 },
  llama2_70b: { maxLr: 1.5e-4, minLr: 1.5e-5, warmupSteps: 2000, totalSteps: 250000 },
};
