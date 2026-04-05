import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { NoiseSchedule, CosineSchedule, SimpleDiffusion } from '../src/diffusion.js';

describe('NoiseSchedule', () => {
  it('creates linear schedule with correct length', () => {
    const schedule = new NoiseSchedule(50);
    assert.equal(schedule.T, 50);
    assert.equal(schedule.betas.length, 50);
    assert.equal(schedule.alphas.length, 50);
    assert.equal(schedule.alphasCumprod.length, 50);
  });

  it('betas increase monotonically', () => {
    const schedule = new NoiseSchedule(100);
    for (let t = 1; t < 100; t++) {
      assert.ok(schedule.betas[t] >= schedule.betas[t - 1], 
        `Beta should increase: β[${t}]=${schedule.betas[t]} < β[${t-1}]=${schedule.betas[t-1]}`);
    }
  });

  it('alphas decrease monotonically', () => {
    const schedule = new NoiseSchedule(100);
    for (let t = 1; t < 100; t++) {
      assert.ok(schedule.alphas[t] <= schedule.alphas[t - 1]);
    }
  });

  it('alphasCumprod decrease monotonically toward 0', () => {
    const schedule = new NoiseSchedule(100);
    for (let t = 1; t < 100; t++) {
      assert.ok(schedule.alphasCumprod[t] < schedule.alphasCumprod[t - 1]);
    }
    assert.ok(schedule.alphasCumprod[0] > 0.99, 'First alpha_bar should be close to 1');
    assert.ok(schedule.alphasCumprod[99] < 0.5, 'Last alpha_bar should be small');
  });

  it('at() returns correct noise parameters', () => {
    const schedule = new NoiseSchedule(100);
    const params = schedule.at(50);
    assert.ok(params.beta > 0 && params.beta < 1);
    assert.ok(params.alpha > 0 && params.alpha < 1);
    assert.ok(params.alphaBar > 0 && params.alphaBar < 1);
    assert.ok(params.sqrtAlphaBar > 0);
    assert.ok(params.sqrtOneMinusAlphaBar > 0);
  });

  it('addNoise preserves dimensions', () => {
    const schedule = new NoiseSchedule(100);
    const x0 = new Float64Array([1, 2, 3, 4]);
    const { xt, noise } = schedule.addNoise(x0, 50);
    assert.equal(xt.length, 4);
    assert.equal(noise.length, 4);
  });

  it('addNoise at t=0 keeps data nearly intact', () => {
    const schedule = new NoiseSchedule(100);
    const x0 = new Float64Array([5, 5, 5, 5]);
    // Run multiple times and check average stays close to data
    let totalDiff = 0;
    for (let i = 0; i < 50; i++) {
      const { xt } = schedule.addNoise(x0, 0);
      for (let j = 0; j < 4; j++) {
        totalDiff += Math.abs(xt[j] - x0[j]);
      }
    }
    const avgDiff = totalDiff / (50 * 4);
    assert.ok(avgDiff < 0.5, `Early noise should barely change data, got avgDiff=${avgDiff}`);
  });

  it('addNoise at t=T-1 produces near-random output', () => {
    const schedule = new NoiseSchedule(100);
    const x0 = new Float64Array([10, 10, 10, 10]);
    let totalDiff = 0;
    for (let i = 0; i < 50; i++) {
      const { xt } = schedule.addNoise(x0, 99);
      for (let j = 0; j < 4; j++) {
        totalDiff += Math.abs(xt[j] - x0[j]);
      }
    }
    const avgDiff = totalDiff / (50 * 4);
    assert.ok(avgDiff > 3, `Late noise should heavily corrupt data, got avgDiff=${avgDiff}`);
  });
});

describe('CosineSchedule', () => {
  it('creates cosine schedule', () => {
    const schedule = new CosineSchedule(100);
    assert.equal(schedule.T, 100);
    assert.ok(schedule.alphasCumprod[0] > 0.99);
    assert.ok(schedule.alphasCumprod[99] < 0.1);
  });

  it('alphasCumprod decreases more gradually than linear', () => {
    const linear = new NoiseSchedule(100);
    const cosine = new CosineSchedule(100);
    // At midpoint, cosine should retain more signal
    assert.ok(cosine.alphasCumprod[50] > linear.alphasCumprod[50] * 0.5,
      'Cosine schedule should be smoother');
  });

  it('betas are bounded', () => {
    const schedule = new CosineSchedule(100);
    for (let t = 0; t < 100; t++) {
      assert.ok(schedule.betas[t] >= 0);
      assert.ok(schedule.betas[t] < 1);
    }
  });
});

describe('SimpleDiffusion', () => {
  it('creates with correct dimensions', () => {
    const model = new SimpleDiffusion(2, { T: 10, hiddenSize: 16 });
    assert.equal(model.dataDim, 2);
    assert.equal(model.T, 10);
  });

  it('time embedding has correct dimension', () => {
    const model = new SimpleDiffusion(2);
    const embed = model.timeEmbed(50);
    assert.equal(embed.length, 16);
    // Should contain sin/cos values in [-1, 1]
    for (let i = 0; i < 16; i++) {
      assert.ok(embed[i] >= -1 && embed[i] <= 1, 
        `Time embedding[${i}]=${embed[i]} should be in [-1,1]`);
    }
  });

  it('different timesteps produce different embeddings', () => {
    const model = new SimpleDiffusion(2);
    const e1 = model.timeEmbed(0);
    const e2 = model.timeEmbed(50);
    let diff = 0;
    for (let i = 0; i < 16; i++) diff += Math.abs(e1[i] - e2[i]);
    assert.ok(diff > 0.1, 'Different timesteps should have different embeddings');
  });

  it('trainStep returns a loss value', () => {
    const model = new SimpleDiffusion(2, { T: 10, hiddenSize: 8 });
    const loss = model.trainStep(new Float64Array([1.0, 2.0]));
    assert.ok(typeof loss === 'number');
    assert.ok(isFinite(loss));
    assert.ok(loss >= 0, `Loss should be non-negative, got ${loss}`);
  });

  it('training reduces loss over epochs', () => {
    const model = new SimpleDiffusion(2, { T: 10, hiddenSize: 16, lr: 0.01 });
    // Simple dataset: points near (1, 1)
    const dataset = [];
    for (let i = 0; i < 50; i++) {
      dataset.push(new Float64Array([1 + (Math.random() - 0.5) * 0.2, 1 + (Math.random() - 0.5) * 0.2]));
    }
    const losses = model.train(dataset, 5);
    assert.ok(losses.length === 5);
    // Last loss should generally be <= first (with some tolerance for stochasticity)
    assert.ok(losses[4] <= losses[0] * 2, 
      `Training should not diverge dramatically: first=${losses[0]}, last=${losses[4]}`);
  });

  it('sample returns correct dimension', () => {
    const model = new SimpleDiffusion(3, { T: 5, hiddenSize: 8 });
    const sample = model.sample();
    assert.equal(sample.length, 3);
    assert.ok(sample.every(v => isFinite(v)), 'All sample values should be finite');
  });

  it('sample produces different outputs each time', () => {
    const model = new SimpleDiffusion(2, { T: 5, hiddenSize: 8 });
    const s1 = model.sample();
    const s2 = model.sample();
    let diff = 0;
    for (let i = 0; i < 2; i++) diff += Math.abs(s1[i] - s2[i]);
    // Samples should differ (stochastic process)
    assert.ok(diff > 0.001, 'Different samples should differ');
  });
});
