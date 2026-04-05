// scheduler.js — Learning rate schedulers for neural network training
//
// Schedulers adjust the learning rate during training:
//   - Warmup: linearly increase lr from 0 to target over warmup steps
//   - CosineAnnealing: cosine decay from max_lr to min_lr
//   - StepDecay: multiply lr by factor every N epochs
//   - WarmupCosine: combine warmup + cosine annealing

export class ConstantLR {
  constructor(lr = 0.001) {
    this.baseLR = lr;
  }
  getLR(step, epoch) { return this.baseLR; }
}

export class LinearWarmup {
  constructor(baseLR = 0.001, warmupSteps = 100) {
    this.baseLR = baseLR;
    this.warmupSteps = warmupSteps;
  }

  getLR(step) {
    if (step >= this.warmupSteps) return this.baseLR;
    return this.baseLR * (step / this.warmupSteps);
  }
}

export class CosineAnnealing {
  constructor(maxLR = 0.001, minLR = 0, totalEpochs = 100) {
    this.maxLR = maxLR;
    this.minLR = minLR;
    this.totalEpochs = totalEpochs;
  }

  getLR(step, epoch) {
    const progress = epoch / this.totalEpochs;
    return this.minLR + 0.5 * (this.maxLR - this.minLR) * (1 + Math.cos(Math.PI * progress));
  }
}

export class StepDecay {
  constructor(baseLR = 0.001, factor = 0.1, stepSize = 30) {
    this.baseLR = baseLR;
    this.factor = factor;
    this.stepSize = stepSize;
  }

  getLR(step, epoch) {
    const decays = Math.floor(epoch / this.stepSize);
    return this.baseLR * Math.pow(this.factor, decays);
  }
}

export class WarmupCosine {
  constructor(maxLR = 0.001, minLR = 0, warmupSteps = 100, totalEpochs = 100) {
    this.warmup = new LinearWarmup(maxLR, warmupSteps);
    this.cosine = new CosineAnnealing(maxLR, minLR, totalEpochs);
    this.warmupSteps = warmupSteps;
    this.maxLR = maxLR;
  }

  getLR(step, epoch) {
    if (step < this.warmupSteps) {
      return this.warmup.getLR(step);
    }
    return this.cosine.getLR(step, epoch);
  }
}

export class ExponentialDecay {
  constructor(baseLR = 0.001, decayRate = 0.96, decaySteps = 1000) {
    this.baseLR = baseLR;
    this.decayRate = decayRate;
    this.decaySteps = decaySteps;
  }

  getLR(step) {
    return this.baseLR * Math.pow(this.decayRate, step / this.decaySteps);
  }
}

export class CyclicLR {
  constructor(baseLR = 0.0001, maxLR = 0.001, stepSize = 200) {
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

export class LinearDecay {
  constructor(startLR = 0.01, endLR = 0.0001, totalEpochs = 100) {
    this.startLR = startLR || 0.01;
    this.endLR = endLR || 0.0001;
    this.totalEpochs = totalEpochs || 100;
  }

  getLR(step) {
    const t = Math.min(step / Math.max(1, this.totalEpochs - 1), 1);
    return this.startLR + (this.endLR - this.startLR) * t;
  }
}

export function createScheduler(name, options = {}) {
  switch (name) {
    case 'constant': return new ConstantLR(options.lr);
    case 'warmup': return new LinearWarmup(options.lr, options.warmupSteps);
    case 'cosine': return new CosineAnnealing(options.maxLR || options.lr, options.minLR, options.totalEpochs);
    case 'step': return new StepDecay(options.lr, options.factor, options.stepSize);
    case 'warmup_cosine': return new WarmupCosine(options.maxLR || options.lr, options.minLR, options.warmupSteps, options.totalEpochs);
    case 'exponential': return new ExponentialDecay(options.lr, options.decayRate, options.decaySteps);
    case 'cyclic': return new CyclicLR(options.baseLR, options.maxLR || options.lr, options.stepSize);
    case 'linear': return new LinearDecay(options.lr || options.baseLR, options.endLR, options.totalEpochs);
    default: throw new Error(`Unknown scheduler: ${name}`);
  }
}
