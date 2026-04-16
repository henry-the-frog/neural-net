// gan-stress.test.js — GAN training stress tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { GAN } from '../src/gan.js';
import { Matrix } from '../src/matrix.js';

describe('GAN Training Stress', () => {
  const ganOpts = {
    latentDim: 4,
    dataSize: 2,
    generatorLayers: [8, 8],
    discriminatorLayers: [8, 8],
  };

  it('discriminator loss decreases on real data', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const gan = new GAN(ganOpts);
      
      // Real data: cluster around [1, 1]
      const realData = new Matrix(20, 2);
      for (let i = 0; i < 20; i++) {
        realData.set(i, 0, 0.8 + Math.random() * 0.4);
        realData.set(i, 1, 0.8 + Math.random() * 0.4);
      }
      
      const result1 = gan.trainDiscriminator(realData, 0.01);
      const firstLoss = result1.dLoss !== undefined ? result1.dLoss : result1;
      let lastLoss = firstLoss;
      for (let i = 0; i < 50; i++) {
        const r = gan.trainDiscriminator(realData, 0.01);
        lastLoss = r.dLoss !== undefined ? r.dLoss : r;
      }
      
      if (lastLoss < firstLoss) passed = true;
    }
    assert.ok(passed, 'Discriminator loss should decrease on real data');
  });

  it('adversarial training produces finite output', () => {
    const gan = new GAN(ganOpts);
    
    const realData = new Matrix(20, 2);
    for (let i = 0; i < 20; i++) {
      realData.set(i, 0, 0.8 + Math.random() * 0.4);
      realData.set(i, 1, 0.8 + Math.random() * 0.4);
    }
    
    // Run adversarial training
    for (let step = 0; step < 50; step++) {
      gan.trainDiscriminator(realData, 0.01);
      gan.trainGenerator(20, 0.01);
    }
    
    // After training, generator should still produce finite output
    const samples = gan.generate(10);
    let allFinite = true;
    for (let i = 0; i < samples.data.length; i++) {
      if (!isFinite(samples.data[i])) allFinite = false;
    }
    assert.ok(allFinite, 'Generator should produce finite values after adversarial training');
    
    // Discriminator should still produce scores in [0,1]
    const scores = gan.discriminate(samples);
    for (let i = 0; i < 10; i++) {
      const s = scores.get(i, 0);
      assert.ok(s >= 0 && s <= 1, `Discriminator score should be in [0,1]: ${s}`);
    }
  });

  it('generator produces finite output', () => {
    const gan = new GAN(ganOpts);
    const samples = gan.generate(10);
    assert.equal(samples.rows, 10);
    assert.equal(samples.cols, 2);
    
    for (let i = 0; i < samples.data.length; i++) {
      assert.ok(isFinite(samples.data[i]), `Sample should be finite: ${samples.data[i]}`);
    }
  });

  it('discriminator output is in [0, 1]', () => {
    const gan = new GAN(ganOpts);
    const input = Matrix.random(10, 2);
    const scores = gan.discriminate(input);
    
    for (let i = 0; i < 10; i++) {
      const score = scores.get(i, 0);
      assert.ok(score >= 0 && score <= 1, `Score should be in [0,1]: ${score}`);
    }
  });
});
