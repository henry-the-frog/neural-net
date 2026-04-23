// lr-scheduler.js — Learning Rate Schedulers for neural network training
//
// Provides composable LR scheduling policies:
// - ConstantLR: Fixed learning rate
// - LinearWarmup: Ramp from 0 to lr_max over warmup steps
// - CosineDecay: Cosine annealing from lr_max to lr_min
// - StepDecay: Multiply by gamma at milestone steps
// - OneCycle: Ramp up then cosine/linear decay (Smith 2018)
// - WarmupScheduler: Compose warmup with any base scheduler
//
// Usage:
//   const scheduler = new WarmupScheduler(
//     new CosineDecay(0.001, 10000, 1e-6),
//     warmupSteps: 1000
//   );
//   for (let step = 0; step < totalSteps; step++) {
//     const lr = scheduler.step();
//     optimizer.updateWeights(lr);
//   }

/**
 * Constant learning rate.
 */
export class ConstantLR {
  constructor(lr) {
    this.lr = lr;
    this._step = 0;
  }
  
  step() {
    this._step++;
    return this.lr;
  }
  
  getLR() { return this.lr; }
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

/**
 * Cosine annealing learning rate decay.
 * LR = lrMin + 0.5 * (lrMax - lrMin) * (1 + cos(π * t / T))
 * where t = current step, T = total steps
 */
export class CosineDecay {
  constructor(lrMax, totalSteps, lrMin = 0) {
    this.lrMax = lrMax;
    this.lrMin = lrMin;
    this.totalSteps = totalSteps;
    this._step = 0;
  }
  
  step() {
    const t = Math.min(this._step, this.totalSteps);
    const lr = this.lrMin + 0.5 * (this.lrMax - this.lrMin) * (1 + Math.cos(Math.PI * t / this.totalSteps));
    this._step++;
    return lr;
  }
  
  getLR() {
    const t = Math.min(this._step, this.totalSteps);
    return this.lrMin + 0.5 * (this.lrMax - this.lrMin) * (1 + Math.cos(Math.PI * t / this.totalSteps));
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

/**
 * Step decay: multiply LR by gamma at each milestone.
 * milestones: array of step numbers where decay occurs
 */
export class StepDecay {
  constructor(lrInit, milestones, gamma = 0.1) {
    this.lrInit = lrInit;
    this.milestones = [...milestones].sort((a, b) => a - b);
    this.gamma = gamma;
    this._step = 0;
  }
  
  step() {
    const lr = this.getLR();
    this._step++;
    return lr;
  }
  
  getLR() {
    let lr = this.lrInit;
    for (const m of this.milestones) {
      if (this._step >= m) lr *= this.gamma;
      else break;
    }
    return lr;
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

/**
 * Linear warmup: ramp from 0 to lrMax over warmupSteps.
 * After warmup, stays at lrMax.
 */
export class LinearWarmup {
  constructor(lrMax, warmupSteps) {
    this.lrMax = lrMax;
    this.warmupSteps = warmupSteps;
    this._step = 0;
  }
  
  step() {
    const lr = this.getLR();
    this._step++;
    return lr;
  }
  
  getLR() {
    if (this._step >= this.warmupSteps) return this.lrMax;
    return this.lrMax * (this._step / this.warmupSteps);
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

/**
 * Compose linear warmup with any base scheduler.
 * During warmup: LR ramps from 0 to base.getLR().
 * After warmup: delegates to base scheduler.
 */
export class WarmupScheduler {
  constructor(baseScheduler, warmupSteps) {
    this.base = baseScheduler;
    this.warmupSteps = warmupSteps;
    this._step = 0;
  }
  
  step() {
    const baseLR = this.base.step();
    const lr = this._step < this.warmupSteps
      ? baseLR * (this._step / this.warmupSteps)
      : baseLR;
    this._step++;
    return lr;
  }
  
  getLR() {
    const baseLR = this.base.getLR();
    return this._step < this.warmupSteps
      ? baseLR * (this._step / this.warmupSteps)
      : baseLR;
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; this.base.reset(); }
}

/**
 * One-cycle learning rate policy (Smith 2018).
 * Phase 1 (0 → pctStart): linear ramp from lrInit/divFactor to lrMax
 * Phase 2 (pctStart → 1): cosine decay from lrMax to lrInit/finalDivFactor
 */
export class OneCycle {
  constructor(lrMax, totalSteps, {
    pctStart = 0.3,
    divFactor = 25,
    finalDivFactor = 10000,
  } = {}) {
    this.lrMax = lrMax;
    this.totalSteps = totalSteps;
    this.pctStart = pctStart;
    this.divFactor = divFactor;
    this.finalDivFactor = finalDivFactor;
    this.lrInit = lrMax / divFactor;
    this.lrFinal = lrMax / finalDivFactor;
    this.warmupSteps = Math.floor(totalSteps * pctStart);
    this._step = 0;
  }
  
  step() {
    const lr = this.getLR();
    this._step++;
    return lr;
  }
  
  getLR() {
    const t = Math.min(this._step, this.totalSteps);
    
    if (t < this.warmupSteps) {
      // Phase 1: linear ramp up
      const progress = t / this.warmupSteps;
      return this.lrInit + (this.lrMax - this.lrInit) * progress;
    } else {
      // Phase 2: cosine decay
      const progress = (t - this.warmupSteps) / (this.totalSteps - this.warmupSteps);
      return this.lrFinal + 0.5 * (this.lrMax - this.lrFinal) * (1 + Math.cos(Math.PI * progress));
    }
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

/**
 * Exponential decay: LR = lrInit * decay^step
 */
export class ExponentialDecay {
  constructor(lrInit, decayRate, decaySteps = 1) {
    this.lrInit = lrInit;
    this.decayRate = decayRate;
    this.decaySteps = decaySteps;
    this._step = 0;
  }
  
  step() {
    const lr = this.getLR();
    this._step++;
    return lr;
  }
  
  getLR() {
    return this.lrInit * Math.pow(this.decayRate, this._step / this.decaySteps);
  }
  
  getStep() { return this._step; }
  reset() { this._step = 0; }
}

// ========================
// PyTorch-style aliases (used by test/lr-scheduler-stress.test.js)
// These accept step as parameter to getLR() for stateless use.
// ========================

/**
 * StepLR: multiply by gamma every stepSize steps.
 * getLR(step) returns lr at given step number.
 */
export class StepLR {
  constructor(lrInit, stepSize, gamma = 0.1) {
    this.lrInit = lrInit;
    this.stepSize = stepSize;
    this.gamma = gamma;
  }
  getLR(step) {
    const decays = Math.floor(step / this.stepSize);
    return this.lrInit * Math.pow(this.gamma, decays);
  }
}

/**
 * ExponentialLR: lr = lrInit * gamma^step
 */
export class ExponentialLR {
  constructor(lrInit, gamma) {
    this.lrInit = lrInit;
    this.gamma = gamma;
  }
  getLR(step) {
    return this.lrInit * Math.pow(this.gamma, step);
  }
}

/**
 * CosineAnnealingLR: cosine decay from lrMax to lrMin over T_max steps.
 */
export class CosineAnnealingLR {
  constructor(lrMax, T_max, lrMin = 0) {
    this.lrMax = lrMax;
    this.T_max = T_max;
    this.lrMin = lrMin;
  }
  getLR(step) {
    const t = Math.min(step, this.T_max);
    return this.lrMin + 0.5 * (this.lrMax - this.lrMin) * (1 + Math.cos(Math.PI * t / this.T_max));
  }
}

/**
 * WarmupLR: linear ramp from 0 to lrMax over warmupSteps, then constant.
 */
export class WarmupLR {
  constructor(lrMax, warmupSteps) {
    this.lrMax = lrMax;
    this.warmupSteps = warmupSteps;
  }
  getLR(step) {
    if (step >= this.warmupSteps) return this.lrMax;
    return this.lrMax * (step / this.warmupSteps);
  }
}

/**
 * CyclicLR: triangular cycling between baseLR and maxLR.
 */
export class CyclicLR {
  constructor(baseLR, maxLR, stepSize) {
    this.baseLR = baseLR;
    this.maxLR = maxLR;
    this.stepSize = stepSize;
  }
  getLR(step) {
    const cycle = Math.floor(1 + step / (2 * this.stepSize));
    const x = Math.abs(step / this.stepSize - 2 * cycle + 1);
    return this.baseLR + (this.maxLR - this.baseLR) * Math.max(0, 1 - x);
  }
}

/**
 * OneCycleLR: one-cycle policy with step-based getLR.
 */
export class OneCycleLR extends OneCycle {
  constructor(lrMax, totalSteps, opts) {
    super(lrMax, totalSteps, opts);
  }
  getLR(step) {
    if (step !== undefined) {
      this._step = step;
    }
    return super.getLR();
  }
}
