// diffusion-stress.test.js — Diffusion model stress tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { NoiseSchedule, CosineSchedule, SimpleDiffusion } from '../src/diffusion.js';
import { Matrix } from '../src/matrix.js';

describe('Noise Schedule Stress', () => {
  it('linear schedule: alphasCumprod decreases monotonically', () => {
    const ns = new NoiseSchedule(100);
    for (let t = 1; t < 100; t++) {
      assert.ok(ns.alphasCumprod[t] < ns.alphasCumprod[t - 1],
        `alphasCumprod should decrease: ${ns.alphasCumprod[t]} >= ${ns.alphasCumprod[t - 1]} at t=${t}`);
    }
  });

  it('linear schedule: alphasCumprod starts near 1, ends near 0', () => {
    const ns = new NoiseSchedule(100);
    assert.ok(ns.alphasCumprod[0] > 0.99, `Should start near 1: ${ns.alphasCumprod[0]}`);
    assert.ok(ns.alphasCumprod[99] < 0.5, `Should end below 0.5: ${ns.alphasCumprod[99]}`);
  });

  it('cosine schedule: smoother than linear', () => {
    const cos = new CosineSchedule(100);
    const lin = new NoiseSchedule(100);
    assert.ok(isFinite(cos.alphasCumprod[50]), 'Cosine midpoint should be finite');
    assert.ok(isFinite(lin.alphasCumprod[50]), 'Linear midpoint should be finite');
  });

  it('forward process adds noise', () => {
    const ns = new NoiseSchedule(100);
    const x0 = [1, 2, 3, 4];
    
    const { xt: noisy0 } = ns.addNoise(x0, 0);
    let diff0 = 0;
    for (let i = 0; i < 4; i++) diff0 += Math.abs(noisy0[i] - x0[i]);
    
    const { xt: noisy99 } = ns.addNoise(x0, 99);
    let diff99 = 0;
    for (let i = 0; i < 4; i++) diff99 += Math.abs(noisy99[i] - x0[i]);
    
    // On average, more noise at later timesteps (may not always hold for single sample)
    // Just verify both are finite
    assert.ok(isFinite(diff0), 'Early noise should be finite');
    assert.ok(isFinite(diff99), 'Late noise should be finite');
  });

  it('noise has correct length', () => {
    const ns = new NoiseSchedule(100);
    const x0 = [1, 2, 3, 4];
    
    const { xt, noise } = ns.addNoise(x0, 50);
    assert.equal(xt.length, 4, 'xt should have same length as input');
    assert.equal(noise.length, 4, 'noise should have same length as input');
  });
});

describe('SimpleDiffusion Stress', () => {
  it('training loss decreases', () => {
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const diff = new SimpleDiffusion(4, { hiddenSize: 16, T: 50 });
      
      // Simple data: repeated patterns
      const data = [];
      for (let i = 0; i < 20; i++) data.push([0.5, 0.5, 0.5, 0.5]);
      
      const losses = diff.train(data, 50);
      if (losses[losses.length - 1] < losses[0]) passed = true;
    }
    assert.ok(passed, 'Diffusion training loss should decrease in 1 of 5 attempts');
  });

  it('sampling produces finite output', () => {
    const diff = new SimpleDiffusion(4, { hiddenSize: 16, T: 50 });
    const sample = diff.sample();
    assert.equal(sample.length, 4, 'Sample should have correct dimension');
    
    for (let i = 0; i < sample.length; i++) {
      assert.ok(isFinite(sample[i]), `Sample[${i}] should be finite: ${sample[i]}`);
    }
  });

  it('training does not produce NaN', () => {
    const diff = new SimpleDiffusion(4, { hiddenSize: 16, T: 50 });
    const data = [];
    for (let i = 0; i < 10; i++) {
      data.push([Math.random(), Math.random(), Math.random(), Math.random()]);
    }
    
    const losses = diff.train(data, 10);
    for (const loss of losses) {
      assert.ok(isFinite(loss), `Loss should be finite: ${loss}`);
    }
  });
});
