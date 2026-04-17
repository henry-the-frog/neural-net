// LR Scheduler wrappers with standard API: constructor(baseLR, ...), getLR(step)
import { StepDecay, ExponentialDecay, CosineAnnealing, LinearWarmup, CyclicLR as CyclicLRBase, WarmupCosine, ConstantLR, LinearDecay, createScheduler } from './scheduler.js';

export class StepLR {
  constructor(baseLR = 0.1, stepSize = 10, gamma = 0.5) {
    this.baseLR = baseLR;
    this.stepSize = stepSize;
    this.gamma = gamma;
  }
  getLR(step) {
    const decays = Math.floor(step / this.stepSize);
    return this.baseLR * Math.pow(this.gamma, decays);
  }
}

export class ExponentialLR {
  constructor(baseLR = 0.1, gamma = 0.99) {
    this.baseLR = baseLR;
    this.gamma = gamma;
  }
  getLR(step) {
    return this.baseLR * Math.pow(this.gamma, step);
  }
}

export class CosineAnnealingLR {
  constructor(baseLR = 0.1, totalSteps = 100, minLR = 0) {
    this.baseLR = baseLR;
    this.totalSteps = totalSteps;
    this.minLR = minLR;
  }
  getLR(step) {
    return this.minLR + (this.baseLR - this.minLR) * 0.5 * (1 + Math.cos(Math.PI * step / this.totalSteps));
  }
}

export class WarmupLR {
  constructor(baseLR = 0.1, warmupSteps = 10) {
    this.baseLR = baseLR;
    this.warmupSteps = warmupSteps;
  }
  getLR(step) {
    if (step >= this.warmupSteps) return this.baseLR;
    return this.baseLR * (step + 1) / this.warmupSteps;
  }
}

export class CyclicLR {
  constructor(baseLR = 0.001, maxLR = 0.1, stepSize = 10) {
    this.baseLR = baseLR;
    this.maxLR = maxLR;
    this.stepSize = stepSize;
  }
  getLR(step) {
    const cycle = Math.floor(step / (2 * this.stepSize));
    const x = Math.abs(step / this.stepSize - 2 * cycle - 1);
    return this.baseLR + (this.maxLR - this.baseLR) * Math.max(0, 1 - x);
  }
}

export class OneCycleLR {
  constructor(maxLR = 0.1, totalSteps = 100, divFactor = 25, finalDivFactor = 1e4) {
    this.maxLR = maxLR;
    this.totalSteps = totalSteps;
    this.divFactor = divFactor;
    this.finalDivFactor = finalDivFactor;
    this.initialLR = maxLR / divFactor;
    this.minLR = maxLR / finalDivFactor;
  }
  getLR(step) {
    const pct = step / this.totalSteps;
    if (pct <= 0.3) {
      // Warmup phase
      return this.initialLR + (this.maxLR - this.initialLR) * (pct / 0.3);
    } else {
      // Cosine annealing phase
      const annealPct = (pct - 0.3) / 0.7;
      return this.minLR + (this.maxLR - this.minLR) * 0.5 * (1 + Math.cos(Math.PI * annealPct));
    }
  }
}

export { ConstantLR, LinearDecay, createScheduler };
