// vae-stress.test.js — VAE training stress tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { VAE } from '../src/vae.js';
import { Matrix } from '../src/matrix.js';

describe('VAE Stress Tests', () => {
  it('reconstruction loss decreases', () => {
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const vae = new VAE(4, 8, 2);
      
      // Training data: vectors
      const data = [];
      for (let i = 0; i < 20; i++) {
        const v = new Matrix(4, 1);
        for (let j = 0; j < 4; j++) v.set(0, j, i % 2 === 0 ? 0.8 : 0.2);
        data.push(v);
      }
      
      const h1 = vae.train(data, { epochs: 5 });
      const firstLoss = h1.history[0].loss;
      const h2 = vae.train(data, { epochs: 50 });
      const lastLoss = h2.history[h2.history.length - 1].loss;
      
      if (lastLoss < firstLoss) passed = true;
    }
    assert.ok(passed, 'VAE reconstruction loss should decrease');
  });

  it('KL divergence is non-negative after training', () => {
    const vae = new VAE(4, 8, 2);
    const data = [];
    for (let i = 0; i < 10; i++) {
      const v = new Matrix(4, 1);
      for (let j = 0; j < 4; j++) v.set(0, j, Math.random());
      data.push(v);
    }
    
    const history = vae.train(data, { epochs: 10 });
    const lastKL = history.history[history.history.length - 1].kl;
    assert.ok(lastKL >= -0.01, `KL divergence should be non-negative: ${lastKL}`);
  });

  it('encode produces latent representations', () => {
    const vae = new VAE(4, 8, 2);
    const input = new Matrix(4, 1);
    for (let j = 0; j < 4; j++) input.set(0, j, Math.random());
    
    const { mu, logVar } = vae.encode(input);
    assert.ok(mu, 'mu should exist');
    assert.ok(logVar, 'logVar should exist');
    
    for (let i = 0; i < mu.data.length; i++) {
      assert.ok(isFinite(mu.data[i]), `mu should be finite: ${mu.data[i]}`);
      assert.ok(isFinite(logVar.data[i]), `logVar should be finite: ${logVar.data[i]}`);
    }
  });

  it('decode produces output of correct size', () => {
    const vae = new VAE(4, 8, 2);
    const z = new Matrix(2, 1);
    z.set(0, 0, 0.5);
    z.set(0, 1, -0.3);
    
    const decoded = vae.decode(z);
    assert.equal(decoded.rows, 4, 'Decoded output should match inputSize (column vector)');
    
    for (let i = 0; i < decoded.data.length; i++) {
      assert.ok(isFinite(decoded.data[i]), `Decoded should be finite: ${decoded.data[i]}`);
    }
  });

  it('training does not produce NaN', () => {
    const vae = new VAE(4, 8, 2);
    const data = [];
    for (let i = 0; i < 10; i++) {
      const v = new Matrix(4, 1);
      for (let j = 0; j < 4; j++) v.set(0, j, Math.random());
      data.push(v);
    }
    
    const history = vae.train(data, { epochs: 20 });
    for (const epoch of history.history) {
      assert.ok(isFinite(epoch.loss), `Loss should be finite: ${epoch.loss}`);
      assert.ok(isFinite(epoch.recon), `Recon loss should be finite: ${epoch.recon}`);
      assert.ok(isFinite(epoch.kl), `KL loss should be finite: ${epoch.kl}`);
    }
  });
});
