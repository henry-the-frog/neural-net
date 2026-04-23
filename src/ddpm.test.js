// ddpm.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { DDPMScheduler, randn } from './ddpm.js';

describe('DDPM', () => {
  test('linear schedule has increasing betas', () => {
    const scheduler = new DDPMScheduler(100, 0.0001, 0.02, 'linear');
    assert.ok(scheduler.betas[0] < scheduler.betas[99]);
  });

  test('alphas_cumprod is decreasing', () => {
    const scheduler = new DDPMScheduler(100);
    for (let t = 1; t < 100; t++) {
      assert.ok(scheduler.alphasCumprod[t] < scheduler.alphasCumprod[t - 1],
        `ᾱ should decrease: ᾱ[${t}]=${scheduler.alphasCumprod[t]} >= ᾱ[${t-1}]`);
    }
  });

  test('alphas_cumprod starts near 1 and ends near 0', () => {
    const scheduler = new DDPMScheduler(1000);
    assert.ok(scheduler.alphasCumprod[0] > 0.99);
    assert.ok(scheduler.alphasCumprod[999] < 0.01);
  });

  test('addNoise at t=0 keeps data mostly intact', () => {
    const scheduler = new DDPMScheduler(1000);
    const x0 = new Float64Array([1, 2, 3, 4]);
    const noise = new Float64Array([0, 0, 0, 0]); // No noise
    const { xt } = scheduler.addNoise(x0, 0, noise);
    
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(xt[i] - x0[i]) < 0.01, `Should be close at t=0: ${xt[i]} vs ${x0[i]}`);
    }
  });

  test('addNoise at t=T-1 produces mostly noise', () => {
    const scheduler = new DDPMScheduler(1000);
    const x0 = new Float64Array([100, 100, 100, 100]);
    const noise = new Float64Array([0, 0, 0, 0]);
    const { xt } = scheduler.addNoise(x0, 999, noise);
    
    // At t=999, √ᾱ ≈ 0, so xt ≈ 0
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(xt[i]) < 10, `Should be mostly noise at t=999: ${xt[i]}`);
    }
  });

  test('denoise with perfect prediction recovers signal', () => {
    const scheduler = new DDPMScheduler(100);
    const x0 = new Float64Array([5, -3, 2, 0]);
    const noise = new Float64Array([0.1, -0.2, 0.3, -0.1]);
    const t = 10;
    
    const { xt } = scheduler.addNoise(x0, t, noise);
    // If we predict the exact noise, denoise should recover approximately x0
    // (Not exact due to posterior variance sampling at t>0)
    // Test with t=0 for exact recovery
    const { xt: xt0 } = scheduler.addNoise(x0, 0, noise);
    const recovered = scheduler.denoise(xt0, noise, 0);
    
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(recovered[i] - x0[i]) < 1, `Recovery should be close: ${recovered[i]} vs ${x0[i]}`);
    }
  });

  test('loss is 0 for perfect prediction', () => {
    const scheduler = new DDPMScheduler(100);
    const noise = new Float64Array([1, 2, 3]);
    assert.equal(scheduler.loss(noise, noise), 0);
  });

  test('cosine schedule has smoother decay', () => {
    const cosine = new DDPMScheduler(100, 0, 0, 'cosine');
    assert.ok(cosine.alphasCumprod[0] > 0.9);
    assert.ok(cosine.alphasCumprod[99] < 0.1);
  });

  test('randn produces approximately standard normal', () => {
    const samples = Array.from({ length: 10000 }, () => randn());
    const mean = samples.reduce((a, b) => a + b) / samples.length;
    const variance = samples.reduce((a, b) => a + b * b, 0) / samples.length - mean * mean;
    
    assert.ok(Math.abs(mean) < 0.1, `Mean should be ~0, got ${mean}`);
    assert.ok(Math.abs(variance - 1) < 0.15, `Variance should be ~1, got ${variance}`);
  });
});
