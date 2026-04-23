// optimizer-zoo.js — Optimizer Collection
// SGD with momentum, RMSprop, AdaGrad, Adam, AdamW, LAMB, Lion

/**
 * SGD with momentum.
 */
export class SGDMomentum {
  constructor(lr = 0.01, momentum = 0.9) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = null;
  }

  step(params, grads) {
    if (!this.velocity) this.velocity = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      this.velocity[i] = this.momentum * this.velocity[i] - this.lr * grads[i];
      params[i] += this.velocity[i];
    }
  }
}

/**
 * AdaGrad: adaptive learning rate per parameter.
 */
export class AdaGrad {
  constructor(lr = 0.01, eps = 1e-8) {
    this.lr = lr;
    this.eps = eps;
    this.cache = null;
  }

  step(params, grads) {
    if (!this.cache) this.cache = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      this.cache[i] += grads[i] * grads[i];
      params[i] -= this.lr * grads[i] / (Math.sqrt(this.cache[i]) + this.eps);
    }
  }
}

/**
 * RMSprop: running average of squared gradients.
 */
export class RMSprop {
  constructor(lr = 0.001, decay = 0.99, eps = 1e-8) {
    this.lr = lr;
    this.decay = decay;
    this.eps = eps;
    this.cache = null;
  }

  step(params, grads) {
    if (!this.cache) this.cache = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      this.cache[i] = this.decay * this.cache[i] + (1 - this.decay) * grads[i] * grads[i];
      params[i] -= this.lr * grads[i] / (Math.sqrt(this.cache[i]) + this.eps);
    }
  }
}

/**
 * Lion: simple sign-based optimizer (Chen et al., 2023).
 * Uses only the sign of the update, not magnitude.
 */
export class Lion {
  constructor(lr = 1e-4, beta1 = 0.9, beta2 = 0.99, weightDecay = 0) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.weightDecay = weightDecay;
    this.m = null;
  }

  step(params, grads) {
    if (!this.m) this.m = new Float64Array(params.length);
    for (let i = 0; i < params.length; i++) {
      // Interpolation for update direction
      const update = this.beta1 * this.m[i] + (1 - this.beta1) * grads[i];
      // Sign-based update + weight decay
      params[i] -= this.lr * (Math.sign(update) + this.weightDecay * params[i]);
      // Update momentum
      this.m[i] = this.beta2 * this.m[i] + (1 - this.beta2) * grads[i];
    }
  }
}
