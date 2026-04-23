// cfg.js — Classifier-Free Guidance (Ho & Salimans, 2022)
// The key technique behind DALL-E, Stable Diffusion, Imagen.
//
// During training: randomly drop conditioning (set to null) with probability p_uncond
// During inference: interpolate between conditional and unconditional predictions:
//   ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
//   where w = guidance scale (typically 3-15)
//
// Higher w → more faithful to conditioning but less diversity.

/**
 * Apply classifier-free guidance to noise predictions.
 * @param {Float64Array} condPrediction - ε_θ(x_t, t, c) — conditional prediction
 * @param {Float64Array} uncondPrediction - ε_θ(x_t, t, ∅) — unconditional prediction
 * @param {number} guidanceScale - w, guidance strength (1.0 = no guidance)
 * @returns {Float64Array} Guided prediction
 */
export function classifierFreeGuidance(condPrediction, uncondPrediction, guidanceScale = 7.5) {
  const n = condPrediction.length;
  const guided = new Float64Array(n);
  
  for (let i = 0; i < n; i++) {
    guided[i] = uncondPrediction[i] + guidanceScale * (condPrediction[i] - uncondPrediction[i]);
  }
  
  return guided;
}

/**
 * Training helper: randomly drop conditioning.
 * @param {any} condition - The conditioning signal
 * @param {number} dropRate - Probability of dropping (e.g., 0.1)
 * @returns {{ condition: any, isDropped: boolean }}
 */
export function conditionalDropout(condition, dropRate = 0.1) {
  if (Math.random() < dropRate) {
    return { condition: null, isDropped: true };
  }
  return { condition, isDropped: false };
}

/**
 * Dynamic guidance scale (used in some implementations).
 * Reduces guidance at high noise levels (early steps) where the model is less certain.
 * @param {number} baseScale - Base guidance scale
 * @param {number} t - Current timestep
 * @param {number} T - Total timesteps
 * @param {string} mode - 'constant', 'linear', 'cosine'
 */
export function dynamicGuidanceScale(baseScale, t, T, mode = 'constant') {
  if (mode === 'constant') return baseScale;
  
  const progress = t / T; // 0 at start (noisy), 1 at end (clean)
  
  if (mode === 'linear') {
    // Scale increases as we get cleaner
    return 1 + (baseScale - 1) * (1 - progress);
  }
  
  if (mode === 'cosine') {
    // Smooth ramp-up
    return 1 + (baseScale - 1) * (1 - Math.cos(Math.PI * (1 - progress))) / 2;
  }
  
  return baseScale;
}

/**
 * Rescaled CFG (Lin et al., 2024).
 * Prevents the guided output from being too large in magnitude.
 * Rescales to match the std of the conditional prediction.
 */
export function rescaledCFG(condPrediction, uncondPrediction, guidanceScale = 7.5, rescaleStrength = 0.7) {
  const n = condPrediction.length;
  
  // Standard CFG
  const guided = classifierFreeGuidance(condPrediction, uncondPrediction, guidanceScale);
  
  // Compute std of conditional and guided
  let condMean = 0, guidedMean = 0;
  for (let i = 0; i < n; i++) {
    condMean += condPrediction[i];
    guidedMean += guided[i];
  }
  condMean /= n;
  guidedMean /= n;
  
  let condVar = 0, guidedVar = 0;
  for (let i = 0; i < n; i++) {
    condVar += (condPrediction[i] - condMean) ** 2;
    guidedVar += (guided[i] - guidedMean) ** 2;
  }
  const condStd = Math.sqrt(condVar / n + 1e-8);
  const guidedStd = Math.sqrt(guidedVar / n + 1e-8);
  
  // Rescale guided to match conditional std
  const rescaled = new Float64Array(n);
  const rescaleFactor = condStd / guidedStd;
  for (let i = 0; i < n; i++) {
    const raw = guided[i];
    const rescaledVal = guidedMean + (raw - guidedMean) * rescaleFactor;
    // Interpolate between rescaled and raw based on rescale strength
    rescaled[i] = rescaleStrength * rescaledVal + (1 - rescaleStrength) * raw;
  }
  
  return rescaled;
}
