// ebm.js — Energy-Based Models
// Learn an energy function E(x) where low energy = likely data
// Train via contrastive divergence or score matching

// ===== Simple Energy Network =====
// E(x) = -f(x) where f is a neural network
export class EnergyNetwork {
  constructor(inputDim, hiddenDim = 32) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;

    // Two-layer MLP: input → hidden (tanh) → scalar energy
    this.w1 = Array.from({ length: hiddenDim }, () =>
      Array.from({ length: inputDim }, () => (Math.random() - 0.5) * Math.sqrt(2 / inputDim))
    );
    this.b1 = new Array(hiddenDim).fill(0);
    this.w2 = Array.from({ length: hiddenDim }, () => (Math.random() - 0.5) * Math.sqrt(2 / hiddenDim));
    this.b2 = 0;

    // Cache
    this.lastInput = null;
    this.lastHidden = null;
  }

  // Compute energy for input x
  energy(x) {
    this.lastInput = x;
    // Hidden layer
    this.lastHidden = new Array(this.hiddenDim);
    for (let h = 0; h < this.hiddenDim; h++) {
      let sum = this.b1[h];
      for (let i = 0; i < this.inputDim; i++) {
        sum += this.w1[h][i] * x[i];
      }
      this.lastHidden[h] = Math.tanh(sum);
    }

    // Output energy (scalar)
    let energy = this.b2;
    for (let h = 0; h < this.hiddenDim; h++) {
      energy += this.w2[h] * this.lastHidden[h];
    }

    return energy;
  }

  // Compute gradient of energy w.r.t. input x (for Langevin dynamics)
  energyGradient(x) {
    this.energy(x); // Forward to populate cache

    const dEdx = new Array(this.inputDim).fill(0);

    for (let h = 0; h < this.hiddenDim; h++) {
      const tanhDeriv = 1 - this.lastHidden[h] ** 2;
      for (let i = 0; i < this.inputDim; i++) {
        dEdx[i] += this.w2[h] * tanhDeriv * this.w1[h][i];
      }
    }

    return dEdx;
  }

  // Compute gradient of energy w.r.t. parameters (for training)
  paramGradient(x) {
    this.energy(x);

    const dw1 = this.w1.map(row => new Array(this.inputDim).fill(0));
    const db1 = new Array(this.hiddenDim).fill(0);
    const dw2 = new Array(this.hiddenDim).fill(0);
    let db2 = 0;

    // dE/dw2 = hidden
    for (let h = 0; h < this.hiddenDim; h++) {
      dw2[h] = this.lastHidden[h];
    }
    db2 = 1;

    // dE/dw1 = w2[h] * (1 - tanh²) * x[i]
    for (let h = 0; h < this.hiddenDim; h++) {
      const tanhDeriv = 1 - this.lastHidden[h] ** 2;
      db1[h] = this.w2[h] * tanhDeriv;
      for (let i = 0; i < this.inputDim; i++) {
        dw1[h][i] = this.w2[h] * tanhDeriv * x[i];
      }
    }

    return { dw1, db1, dw2, db2 };
  }

  // Update parameters: reduce energy for positive samples, increase for negative
  update(posGrad, negGrad, learningRate) {
    for (let h = 0; h < this.hiddenDim; h++) {
      for (let i = 0; i < this.inputDim; i++) {
        this.w1[h][i] -= learningRate * (posGrad.dw1[h][i] - negGrad.dw1[h][i]);
      }
      this.b1[h] -= learningRate * (posGrad.db1[h] - negGrad.db1[h]);
      this.w2[h] -= learningRate * (posGrad.dw2[h] - negGrad.dw2[h]);
    }
    this.b2 -= learningRate * (posGrad.db2 - negGrad.db2);
  }

  paramCount() {
    return this.hiddenDim * this.inputDim + this.hiddenDim + this.hiddenDim + 1;
  }
}

// ===== Langevin Dynamics Sampling =====
// Sample from p(x) ∝ exp(-E(x)) by running:
// x_{t+1} = x_t - (stepSize/2) * ∇E(x_t) + sqrt(stepSize) * noise
export function langevinSample(energyFn, gradFn, initialX, {
  steps = 100,
  stepSize = 0.01,
  noise = true,
} = {}) {
  let x = [...initialX];
  const trajectory = [x.map(v => v)];

  for (let t = 0; t < steps; t++) {
    const grad = gradFn(x);
    x = x.map((xi, i) => {
      const noiseVal = noise ? Math.sqrt(stepSize) * gaussianRandom() : 0;
      return xi - (stepSize / 2) * grad[i] + noiseVal;
    });
    trajectory.push(x.map(v => v));
  }

  return { sample: x, trajectory };
}

// ===== Contrastive Divergence Training =====
export function trainCD(model, data, {
  epochs = 100,
  learningRate = 0.01,
  cdSteps = 10,
  cdStepSize = 0.01,
} = {}) {
  const losses = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;

    for (const x of data) {
      // Positive phase: compute gradient at data point
      const posGrad = model.paramGradient(x);
      const posEnergy = model.energy(x);

      // Negative phase: sample via Langevin from current model
      const initNeg = Array.from({ length: model.inputDim }, () => Math.random() * 2 - 1);
      const { sample: negSample } = langevinSample(
        z => model.energy(z),
        z => model.energyGradient(z),
        initNeg,
        { steps: cdSteps, stepSize: cdStepSize }
      );
      const negGrad = model.paramGradient(negSample);
      const negEnergy = model.energy(negSample);

      // Update: push down energy of data, push up energy of samples
      model.update(posGrad, negGrad, learningRate);

      epochLoss += posEnergy - negEnergy;
    }

    losses.push(epochLoss / data.length);
  }

  return losses;
}

// ===== Score Matching (Denoising) =====
// Alternative training: match ∇_x log p(x) ≈ ∇_x E(x)
export function trainScoreMatching(model, data, {
  epochs = 100,
  learningRate = 0.01,
  noiseLevel = 0.1,
} = {}) {
  const losses = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;

    for (const x of data) {
      // Add noise
      const noisy = x.map(v => v + gaussianRandom() * noiseLevel);

      // Target score: (x - noisy) / noiseLevel²
      const targetScore = x.map((v, i) => (v - noisy[i]) / (noiseLevel ** 2));

      // Model score: -∇E(noisy)
      const modelScore = model.energyGradient(noisy).map(v => -v);

      // MSE loss between scores
      let scoreLoss = 0;
      for (let i = 0; i < model.inputDim; i++) {
        scoreLoss += (modelScore[i] - targetScore[i]) ** 2;
      }
      epochLoss += scoreLoss / model.inputDim;

      // Simple gradient update (approximate)
      const posGrad = model.paramGradient(x);
      const negGrad = model.paramGradient(noisy);
      model.update(negGrad, posGrad, learningRate * 0.1); // Smaller LR for stability
    }

    losses.push(epochLoss / data.length);
  }

  return losses;
}

// ===== Utility =====
function gaussianRandom() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}
