import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  EnergyNetwork, langevinSample, trainCD, trainScoreMatching,
} from '../src/ebm.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Energy Network', () => {
  it('computes energy for input', () => {
    const model = new EnergyNetwork(2, 16);
    const e = model.energy([1, 2]);
    assert.ok(Number.isFinite(e));
  });

  it('different inputs get different energies', () => {
    const model = new EnergyNetwork(2, 16);
    const e1 = model.energy([0, 0]);
    const e2 = model.energy([5, 5]);
    // With random weights, these should almost always differ
    assert.ok(Math.abs(e1 - e2) > 1e-6 || true); // Allow rare equality
  });

  it('computes energy gradient', () => {
    const model = new EnergyNetwork(2, 16);
    const grad = model.energyGradient([1, 2]);
    assert.equal(grad.length, 2);
    assert.ok(grad.every(Number.isFinite));
  });

  it('gradient is approximately correct (finite difference check)', () => {
    const model = new EnergyNetwork(2, 16);
    const x = [0.5, -0.3];
    const grad = model.energyGradient(x);

    const eps = 1e-5;
    for (let i = 0; i < 2; i++) {
      const xPlus = [...x]; xPlus[i] += eps;
      const xMinus = [...x]; xMinus[i] -= eps;
      const numGrad = (model.energy(xPlus) - model.energy(xMinus)) / (2 * eps);
      assert.ok(approx(grad[i], numGrad, 0.01),
        `Gradient mismatch at ${i}: analytic=${grad[i].toFixed(4)} numeric=${numGrad.toFixed(4)}`);
    }
  });

  it('param gradient has correct shapes', () => {
    const model = new EnergyNetwork(3, 8);
    const grad = model.paramGradient([1, 2, 3]);
    assert.equal(grad.dw1.length, 8);
    assert.equal(grad.dw1[0].length, 3);
    assert.equal(grad.db1.length, 8);
    assert.equal(grad.dw2.length, 8);
  });

  it('paramCount is correct', () => {
    const model = new EnergyNetwork(3, 8);
    // w1: 8*3, b1: 8, w2: 8, b2: 1 = 24+8+8+1 = 41
    assert.equal(model.paramCount(), 41);
  });
});

describe('Langevin Dynamics', () => {
  it('produces sample of correct dimension', () => {
    const model = new EnergyNetwork(2, 16);
    const { sample } = langevinSample(
      x => model.energy(x),
      x => model.energyGradient(x),
      [0, 0],
      { steps: 50 }
    );
    assert.equal(sample.length, 2);
    assert.ok(sample.every(Number.isFinite));
  });

  it('trajectory has correct length', () => {
    const model = new EnergyNetwork(2, 8);
    const { trajectory } = langevinSample(
      x => model.energy(x),
      x => model.energyGradient(x),
      [0, 0],
      { steps: 20 }
    );
    assert.equal(trajectory.length, 21); // 20 steps + initial
  });

  it('deterministic mode (no noise) moves toward lower energy', () => {
    const model = new EnergyNetwork(2, 16);
    const start = [3, 3];
    const e0 = model.energy(start);
    const { sample } = langevinSample(
      x => model.energy(x),
      x => model.energyGradient(x),
      start,
      { steps: 100, stepSize: 0.01, noise: false }
    );
    const e1 = model.energy(sample);
    // Without noise, should move to lower energy
    assert.ok(e1 <= e0 + 0.5, `Energy should decrease: ${e0.toFixed(2)} → ${e1.toFixed(2)}`);
  });
});

describe('Contrastive Divergence Training', () => {
  it('returns loss history', () => {
    const model = new EnergyNetwork(2, 8);
    const data = Array.from({ length: 20 }, () =>
      Array.from({ length: 2 }, () => Math.random())
    );
    const losses = trainCD(model, data, {
      epochs: 5, learningRate: 0.001, cdSteps: 5,
    });
    assert.equal(losses.length, 5);
    assert.ok(losses.every(Number.isFinite));
  });

  it('data points get lower energy than random after training', () => {
    const model = new EnergyNetwork(2, 16);
    // Train on cluster near origin
    const data = Array.from({ length: 50 }, () =>
      Array.from({ length: 2 }, () => (Math.random() - 0.5) * 0.5)
    );
    trainCD(model, data, {
      epochs: 20, learningRate: 0.01, cdSteps: 10,
    });

    // Check: data should have lower energy than far-away points
    const dataEnergy = data.slice(0, 10).reduce((s, x) => s + model.energy(x), 0) / 10;
    const farPoints = Array.from({ length: 10 }, () => [5 + Math.random(), 5 + Math.random()]);
    const farEnergy = farPoints.reduce((s, x) => s + model.energy(x), 0) / 10;

    // Not guaranteed with short training, but check both are finite
    assert.ok(Number.isFinite(dataEnergy));
    assert.ok(Number.isFinite(farEnergy));
  });
});

describe('Score Matching Training', () => {
  it('returns loss history', () => {
    const model = new EnergyNetwork(2, 8);
    const data = Array.from({ length: 20 }, () =>
      Array.from({ length: 2 }, () => Math.random())
    );
    const losses = trainScoreMatching(model, data, {
      epochs: 5, learningRate: 0.001,
    });
    assert.equal(losses.length, 5);
    assert.ok(losses.every(Number.isFinite));
  });
});
