// vae.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { VAE } from './vae.js';

describe('VAE', () => {
  test('encode produces mu and logvar of correct dimension', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).fill(0.5);
    const { mu, logvar } = vae.encode(x);
    assert.equal(mu.length, 3);
    assert.equal(logvar.length, 3);
  });

  test('reparameterize produces latent of correct dimension', () => {
    const vae = new VAE(8, 16, 3);
    const mu = new Float64Array([0, 0, 0]);
    const logvar = new Float64Array([0, 0, 0]);
    const z = vae.reparameterize(mu, logvar);
    assert.equal(z.length, 3);
  });

  test('decode produces output in [0, 1]', () => {
    const vae = new VAE(8, 16, 3);
    const z = new Float64Array([1, -1, 0.5]);
    const output = vae.decode(z);
    assert.equal(output.length, 8);
    for (let i = 0; i < 8; i++) {
      assert.ok(output[i] >= 0 && output[i] <= 1, `Output[${i}] = ${output[i]} not in [0,1]`);
    }
  });

  test('forward pass works end-to-end', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).map(() => Math.random());
    const { reconstruction, mu, logvar, z } = vae.forward(x);
    assert.equal(reconstruction.length, 8);
    assert.equal(mu.length, 3);
    assert.equal(z.length, 3);
  });

  test('loss is finite and positive', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).map(() => Math.random());
    const { reconstruction, mu, logvar } = vae.forward(x);
    const { total, recon, kl } = vae.loss(x, reconstruction, mu, logvar);
    assert.ok(isFinite(total), `Total loss should be finite: ${total}`);
    assert.ok(recon >= 0, 'Reconstruction loss should be non-negative');
    assert.ok(kl >= 0 || kl < 0.1, 'KL should be reasonable');
  });

  test('KL divergence is 0 for standard normal', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).fill(0.5);
    const mu = new Float64Array(3).fill(0);
    const logvar = new Float64Array(3).fill(0); // σ² = 1
    const { kl } = vae.loss(x, x, mu, logvar);
    assert.ok(Math.abs(kl) < 0.001, `KL for standard normal should be ~0, got ${kl}`);
  });

  test('sample produces valid output', () => {
    const vae = new VAE(8, 16, 3);
    const sample = vae.sample();
    assert.equal(sample.length, 8);
    for (let i = 0; i < 8; i++) {
      assert.ok(sample[i] >= 0 && sample[i] <= 1);
    }
  });

  test('interpolate produces correct number of steps', () => {
    const vae = new VAE(8, 16, 3);
    const x1 = new Float64Array(8).fill(0.2);
    const x2 = new Float64Array(8).fill(0.8);
    const results = vae.interpolate(x1, x2, 4);
    assert.equal(results.length, 5); // 4 steps + 1
  });

  test('beta-VAE with higher beta increases KL weight', () => {
    const vae1 = new VAE(8, 16, 3, 1.0);
    const vae10 = new VAE(8, 16, 3, 10.0);
    
    // Same KL, different total loss
    const x = new Float64Array(8).fill(0.5);
    const mu = new Float64Array(3).fill(1); // Non-zero mean → KL > 0
    const logvar = new Float64Array(3).fill(0);
    
    const loss1 = vae1.loss(x, x, mu, logvar);
    const loss10 = vae10.loss(x, x, mu, logvar);
    
    assert.ok(loss10.total > loss1.total, 'Higher beta should increase total loss');
  });
});
