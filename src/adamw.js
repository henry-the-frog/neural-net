// adamw.js — AdamW Optimizer with Weight Decay Decoupling
// Paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
// THE standard optimizer for training LLMs.
//
// Key difference from Adam+L2: weight decay is applied directly to weights,
// not through the gradient. This gives better generalization.
//
// Update rule:
//   m_t = β1 * m_{t-1} + (1-β1) * g_t
//   v_t = β2 * v_{t-1} + (1-β2) * g_t²
//   m̂_t = m_t / (1-β1^t)    (bias correction)
//   v̂_t = v_t / (1-β2^t)
//   θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

/**
 * AdamW optimizer state for a parameter.
 */
class ParamState {
  constructor(size) {
    this.m = new Float64Array(size); // first moment
    this.v = new Float64Array(size); // second moment
  }
}

/**
 * AdamW optimizer.
 *
 * @param {object} opts
 * @param {number} opts.lr - learning rate (default 1e-3)
 * @param {number} opts.beta1 - first moment decay (default 0.9)
 * @param {number} opts.beta2 - second moment decay (default 0.999)
 * @param {number} opts.epsilon - numerical stability (default 1e-8)
 * @param {number} opts.weightDecay - weight decay (default 0.01)
 */
export class AdamW {
  constructor({ lr = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0.01 } = {}) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.weightDecay = weightDecay;
    this.step = 0;
    this.states = new Map(); // param id → ParamState
  }

  /**
   * Update a parameter array given its gradient.
   *
   * @param {string} paramId - unique identifier for this parameter
   * @param {Float64Array} param - parameter values (modified in-place)
   * @param {Float64Array} grad - gradient values
   * @param {number} [lr] - optional per-step learning rate override
   */
  update(paramId, param, grad, lr = null) {
    const currentLr = lr ?? this.lr;

    if (!this.states.has(paramId)) {
      this.states.set(paramId, new ParamState(param.length));
    }
    const state = this.states.get(paramId);

    this.step++;
    const { beta1, beta2, epsilon, weightDecay } = this;

    // Bias correction factors
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);

    for (let i = 0; i < param.length; i++) {
      // Update moments
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i];
      state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i] * grad[i];

      // Bias-corrected moments
      const mHat = state.m[i] / bc1;
      const vHat = state.v[i] / bc2;

      // AdamW update: decoupled weight decay
      param[i] -= currentLr * (mHat / (Math.sqrt(vHat) + epsilon) + weightDecay * param[i]);
    }
  }

  /**
   * Reset optimizer state.
   */
  reset() {
    this.step = 0;
    this.states.clear();
  }
}

/**
 * Simple SGD with momentum (for comparison).
 */
export class SGDMomentum {
  constructor({ lr = 0.01, momentum = 0.9 } = {}) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocities = new Map();
  }

  update(paramId, param, grad) {
    if (!this.velocities.has(paramId)) {
      this.velocities.set(paramId, new Float64Array(param.length));
    }
    const v = this.velocities.get(paramId);

    for (let i = 0; i < param.length; i++) {
      v[i] = this.momentum * v[i] + grad[i];
      param[i] -= this.lr * v[i];
    }
  }

  reset() { this.velocities.clear(); }
}
