// ppo.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { ppoClipLoss, computeGAE, klPenalty, rlhfReward, normalizeAdvantages, valueLoss, ppoStep } from './ppo.js';

describe('PPO', () => {
  test('clip loss with no clipping (ratio ≈ 1)', () => {
    const ratios = new Float64Array([1.0, 1.0, 1.0]);
    const advantages = new Float64Array([1.0, -0.5, 0.3]);
    const { loss, clipFraction } = ppoClipLoss(ratios, advantages);
    assert.ok(isFinite(loss));
    assert.equal(clipFraction, 0);
  });

  test('clip loss clips extreme ratios', () => {
    const ratios = new Float64Array([5.0, 0.01, 1.0]); // Very extreme
    const advantages = new Float64Array([1.0, 1.0, 1.0]);
    const { clipFraction } = ppoClipLoss(ratios, advantages, 0.2);
    assert.ok(clipFraction > 0, 'Should clip extreme ratios');
  });

  test('GAE computes advantages', () => {
    const rewards = new Float64Array([1, 0, 0, 0, 10]); // Reward at start and end
    const values = new Float64Array([5, 4, 3, 2, 1]);
    const advantages = computeGAE(rewards, values);
    
    assert.equal(advantages.length, 5);
    // Last advantage should be positive (reward=10, value=1)
    assert.ok(advantages[4] > 0, `Last advantage should be positive: ${advantages[4]}`);
  });

  test('GAE with gamma=0 gives TD(0) errors', () => {
    const rewards = new Float64Array([1, 2, 3]);
    const values = new Float64Array([0, 0, 0]);
    const advantages = computeGAE(rewards, values, 0, 0.95);
    // With gamma=0: delta_t = r_t + 0*V(t+1) - V(t) = r_t
    assert.ok(Math.abs(advantages[0] - 1) < 0.01);
    assert.ok(Math.abs(advantages[1] - 2) < 0.01);
    assert.ok(Math.abs(advantages[2] - 3) < 0.01);
  });

  test('KL penalty is 0 for identical policies', () => {
    const logProbs = new Float64Array([-1, -2, -0.5]);
    assert.ok(Math.abs(klPenalty(logProbs, logProbs)) < 1e-10);
  });

  test('RLHF reward penalizes KL divergence', () => {
    const policyLP = new Float64Array([-0.5, -0.5]); // More confident than ref
    const refLP = new Float64Array([-1.0, -1.0]);
    const reward = rlhfReward(5.0, policyLP, refLP, 0.1);
    assert.ok(reward < 5.0, 'KL penalty should reduce reward');
  });

  test('normalize advantages gives zero mean, unit std', () => {
    const advantages = new Float64Array([1, 5, 3, 2, 4]);
    const norm = normalizeAdvantages(advantages);
    
    let mean = 0;
    for (let i = 0; i < norm.length; i++) mean += norm[i];
    mean /= norm.length;
    assert.ok(Math.abs(mean) < 0.001, `Mean should be ~0, got ${mean}`);
    
    let variance = 0;
    for (let i = 0; i < norm.length; i++) variance += (norm[i] - mean) ** 2;
    variance /= norm.length;
    assert.ok(Math.abs(variance - 1) < 0.01, `Variance should be ~1, got ${variance}`);
  });

  test('ppoStep combines policy and value losses', () => {
    const ratios = new Float64Array([1.1, 0.9, 1.0]);
    const advantages = new Float64Array([1.0, -0.5, 0.3]);
    const values = new Float64Array([1, 2, 3]);
    const returns = new Float64Array([1.5, 1.8, 3.2]);
    
    const step = ppoStep(ratios, advantages, values, returns);
    assert.ok(isFinite(step.totalLoss));
    assert.ok(isFinite(step.approxKL));
    assert.ok(step.clipFraction >= 0 && step.clipFraction <= 1);
  });
});
