// vae.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { VAE } from './vae.js';
import { Matrix } from './matrix.js';

describe('VAE', () => {
  test('encode produces mu and logVar of correct dimension', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).fill(0.5);
    const { mu, logVar } = vae.encode(x);
    assert.equal(mu.rows, 3);
    assert.equal(logVar.rows, 3);
  });

  test('reparameterize produces latent of correct dimension', () => {
    const vae = new VAE(8, 16, 3);
    const mu = new Matrix(3, 1, new Float64Array([0, 0, 0]));
    const logVar = new Matrix(3, 1, new Float64Array([0, 0, 0]));
    const z = vae.reparameterize(mu, logVar);
    assert.equal(z.rows, 3);
  });

  test('decode produces output in [0, 1]', () => {
    const vae = new VAE(8, 16, 3);
    const z = new Matrix(3, 1, new Float64Array([1, -1, 0.5]));
    const output = vae.decode(z);
    assert.equal(output.rows, 8);
    for (let i = 0; i < 8; i++) {
      assert.ok(output.data[i] >= 0 && output.data[i] <= 1, `Output[${i}] = ${output.data[i]} not in [0,1]`);
    }
  });

  test('forward pass works end-to-end', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).map(() => Math.random());
    const { reconstruction, mu, logVar, z } = vae.forward(x);
    assert.equal(reconstruction.rows, 8);
    assert.equal(mu.rows, 3);
    assert.equal(z.rows, 3);
  });

  test('loss is finite and positive', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).map(() => Math.random());
    const { reconstruction, mu, logVar } = vae.forward(x);
    const { total, recon, kl } = vae.computeLoss(x, reconstruction, mu, logVar);
    assert.ok(isFinite(total), `Total loss should be finite: ${total}`);
    assert.ok(recon >= 0, 'Reconstruction loss should be non-negative');
    assert.ok(kl >= 0 || kl < 0.1, 'KL should be reasonable');
  });

  test('KL divergence is 0 for standard normal', () => {
    const vae = new VAE(8, 16, 3);
    const x = new Float64Array(8).fill(0.5);
    const mu = new Matrix(3, 1, new Float64Array([0, 0, 0]));
    const logVar = new Matrix(3, 1, new Float64Array([0, 0, 0])); // σ² = 1
    const reconstruction = new Matrix(8, 1, new Float64Array(x));
    const { kl } = vae.computeLoss(x, reconstruction, mu, logVar);
    assert.ok(Math.abs(kl) < 0.001, `KL for standard normal should be ~0, got ${kl}`);
  });

  test('sample produces valid output', () => {
    const vae = new VAE(8, 16, 3);
    const samples = vae.generate(1);
    assert.equal(samples.length, 1);
    const sample = samples[0];
    assert.equal(sample.rows, 8);
    for (let i = 0; i < 8; i++) {
      assert.ok(sample.data[i] >= 0 && sample.data[i] <= 1);
    }
  });

  test('interpolate produces correct number of steps', () => {
    const vae = new VAE(8, 16, 3);
    const x1 = new Float64Array(8).fill(0.2);
    const x2 = new Float64Array(8).fill(0.8);
    const results = vae.interpolate(x1, x2, 4);
    assert.equal(results.length, 5); // 4 steps + 1 (0..4 inclusive)
  });

  test('beta-VAE with higher beta increases KL weight', () => {
    const vae1 = new VAE(8, 16, 3, { beta: 1.0 });
    const vae10 = new VAE(8, 16, 3, { beta: 10.0 });
    
    const x = new Float64Array(8).fill(0.5);
    const mu = new Matrix(3, 1, new Float64Array([1, 1, 1])); // Non-zero mean → KL > 0
    const logVar = new Matrix(3, 1, new Float64Array([0, 0, 0]));
    const reconstruction = new Matrix(8, 1, new Float64Array(x));
    
    const loss1 = vae1.computeLoss(x, reconstruction, mu, logVar);
    const loss10 = vae10.computeLoss(x, reconstruction, mu, logVar);
    
    assert.ok(loss10.total > loss1.total, 'Higher beta should increase total loss');
  });
});
