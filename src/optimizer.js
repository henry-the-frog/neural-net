// optimizer.js — Optimizer + LR Scheduler integration
// Wraps an optimizer (e.g., AdamW) with an LR scheduler for clean training loops.

/**
 * Combines an optimizer with an LR scheduler.
 * Usage:
 *   const opt = new ScheduledOptimizer(
 *     new AdamW({ lr: 0.001 }),
 *     new WarmupScheduler(new CosineDecay(0.001, 10000), 1000)
 *   );
 *   // Each update uses the scheduler's current LR
 *   opt.update('layer1.W', params, grads);
 *   opt.schedulerStep(); // advance scheduler (call once per training step)
 */
export class ScheduledOptimizer {
  /**
   * @param {Object} optimizer — must have update(id, param, grad, lr?) method
   * @param {Object} scheduler — must have step() and getLR() methods
   */
  constructor(optimizer, scheduler) {
    this.optimizer = optimizer;
    this.scheduler = scheduler;
    this._currentLR = scheduler.getLR();
  }
  
  /**
   * Update parameters using the current scheduled learning rate.
   */
  update(paramId, param, grad) {
    this.optimizer.update(paramId, param, grad, this._currentLR);
  }
  
  /**
   * Advance the scheduler by one step (call once per training step, not per parameter).
   */
  schedulerStep() {
    this._currentLR = this.scheduler.step();
  }
  
  /**
   * Get current learning rate.
   */
  getLR() {
    return this._currentLR;
  }
  
  /**
   * Get current training step from the scheduler.
   */
  getStep() {
    return this.scheduler.getStep();
  }
  
  /**
   * Reset both optimizer and scheduler.
   */
  reset() {
    this.scheduler.reset();
    this._currentLR = this.scheduler.getLR();
    if (this.optimizer.states) this.optimizer.states.clear();
    if (this.optimizer.step !== undefined) this.optimizer.step = 0;
  }
}

/**
 * Simple SGD optimizer for use with ScheduledOptimizer.
 */
export class SGD {
  constructor({ lr = 0.01, momentum = 0 } = {}) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocities = new Map();
  }
  
  update(paramId, param, grad, lr = null) {
    const currentLr = lr ?? this.lr;
    
    if (this.momentum > 0) {
      if (!this.velocities.has(paramId)) {
        this.velocities.set(paramId, new Float64Array(param.length));
      }
      const v = this.velocities.get(paramId);
      for (let i = 0; i < param.length; i++) {
        v[i] = this.momentum * v[i] + grad[i];
        param[i] -= currentLr * v[i];
      }
    } else {
      for (let i = 0; i < param.length; i++) {
        param[i] -= currentLr * grad[i];
      }
    }
  }
}
