// ema.js — Exponential Moving Average of Model Weights
// Used in diffusion models (DDPM uses EMA of denoiser weights for sampling),
// self-supervised learning (BYOL, DINO), and RL (target networks in DQN).
//
// θ_ema = β * θ_ema + (1-β) * θ_model
// Typical β: 0.999 for diffusion, 0.996 for BYOL

/**
 * EMA tracker for model parameters.
 */
export class EMA {
  /**
   * @param {Float64Array|Array<number>} params - Initial parameter values
   * @param {number} decay - EMA decay factor (β), typically 0.999
   */
  constructor(params, decay = 0.999) {
    this.decay = decay;
    this.shadow = new Float64Array(params); // EMA parameters
    this.steps = 0;
  }

  /**
   * Update EMA parameters with new model parameters.
   * @param {Float64Array|Array<number>} params - Current model parameters
   */
  update(params) {
    this.steps++;
    
    // Use warmup: actual_decay = min(decay, (1 + steps) / (10 + steps))
    const actualDecay = Math.min(this.decay, (1 + this.steps) / (10 + this.steps));
    
    for (let i = 0; i < this.shadow.length; i++) {
      this.shadow[i] = actualDecay * this.shadow[i] + (1 - actualDecay) * params[i];
    }
  }

  /**
   * Get EMA parameters.
   */
  get() {
    return new Float64Array(this.shadow);
  }

  /**
   * Copy EMA parameters back to model (for sampling/evaluation).
   */
  apply(target) {
    for (let i = 0; i < this.shadow.length; i++) {
      target[i] = this.shadow[i];
    }
  }

  /**
   * Reset to current model parameters.
   */
  reset(params) {
    this.shadow = new Float64Array(params);
    this.steps = 0;
  }
}

/**
 * Polyak averaging: average of all past parameter values.
 * Used in convex optimization theory.
 */
export class PolyakAveraging {
  constructor(params) {
    this.sum = new Float64Array(params);
    this.count = 1;
  }

  update(params) {
    this.count++;
    for (let i = 0; i < this.sum.length; i++) {
      this.sum[i] += params[i];
    }
  }

  get() {
    const avg = new Float64Array(this.sum.length);
    for (let i = 0; i < avg.length; i++) avg[i] = this.sum[i] / this.count;
    return avg;
  }
}
