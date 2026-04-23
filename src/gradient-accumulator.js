// gradient-accumulator.js — Gradient accumulation over multiple micro-batches
export class GradientAccumulator {
  constructor(paramCount, accumSteps = 4) {
    this.accumSteps = accumSteps;
    this.buffer = new Float64Array(paramCount);
    this.currentStep = 0;
  }

  accumulate(gradients) {
    for (let i = 0; i < gradients.length; i++) this.buffer[i] += gradients[i];
    this.currentStep++;
    if (this.currentStep >= this.accumSteps) {
      const averaged = new Float64Array(this.buffer.length);
      for (let i = 0; i < averaged.length; i++) averaged[i] = this.buffer[i] / this.accumSteps;
      this.buffer.fill(0);
      this.currentStep = 0;
      return { ready: true, gradients: averaged };
    }
    return { ready: false, gradients: null };
  }

  reset() {
    this.buffer.fill(0);
    this.currentStep = 0;
  }
}
