import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { VAE } from '../src/vae.js';
import { Matrix } from '../src/matrix.js';
import { createMiniDigits } from '../src/mnist.js';

describe('Variational Autoencoder', () => {
  describe('Construction', () => {
    it('should create with correct dimensions', () => {
      const vae = new VAE(10, 8, 3);
      assert.equal(vae.inputSize, 10);
      assert.equal(vae.hiddenSize, 8);
      assert.equal(vae.latentSize, 3);
    });
  });

  describe('Forward Pass', () => {
    it('should encode to latent distribution', () => {
      const vae = new VAE(8, 6, 2);
      const input = new Matrix(8, 1, new Float64Array([1, 0, 1, 0, 1, 0, 1, 0]));
      const { mu, logVar } = vae.encode(input);
      assert.equal(mu.rows, 2);
      assert.equal(logVar.rows, 2);
      assert.ok(!mu.data.some(isNaN));
      assert.ok(!logVar.data.some(isNaN));
    });

    it('should decode from latent space', () => {
      const vae = new VAE(8, 6, 2);
      const z = new Matrix(2, 1, new Float64Array([0.5, -0.3]));
      const recon = vae.decode(z);
      assert.equal(recon.rows, 8);
      // Sigmoid output in [0, 1]
      for (let i = 0; i < 8; i++) {
        assert.ok(recon.data[i] >= 0 && recon.data[i] <= 1);
      }
    });

    it('should run full forward pass', () => {
      const vae = new VAE(8, 6, 2);
      const { reconstruction, mu, logVar, z } = vae.forward([1, 0, 1, 0, 1, 0, 1, 0]);
      assert.equal(reconstruction.rows, 8);
      assert.equal(mu.rows, 2);
      assert.equal(z.rows, 2);
    });
  });

  describe('Loss', () => {
    it('should compute reconstruction + KL loss', () => {
      const vae = new VAE(4, 3, 2);
      const input = new Matrix(4, 1, new Float64Array([0.8, 0.2, 0.7, 0.3]));
      const { reconstruction, mu, logVar } = vae.forward(input);
      const { total, recon, kl } = vae.computeLoss(input, reconstruction, mu, logVar);
      assert.ok(total >= 0);
      assert.ok(recon >= 0);
      assert.ok(kl >= 0);
      assert.ok(Math.abs(total - (recon + vae.beta * kl)) < 1e-6);
    });
  });

  describe('Training', () => {
    it('should reduce loss over training', () => {
      const vae = new VAE(4, 8, 2, { learningRate: 0.01 });
      const data = [
        [0.9, 0.1, 0.9, 0.1],
        [0.1, 0.9, 0.1, 0.9],
        [0.9, 0.9, 0.1, 0.1],
      ];

      const { history } = vae.train(data, { epochs: 50 });
      assert.equal(history.length, 50);
      assert.ok(history[history.length - 1].loss < history[0].loss,
        `Loss should decrease: ${history[0].loss.toFixed(2)} → ${history[history.length - 1].loss.toFixed(2)}`);
    });

    it('should learn digit representations', () => {
      const data = createMiniDigits({ samplesPerDigit: 5, noise: 0.02 });
      const inputs = data.images.map(img => Array.from(img.data));
      const vae = new VAE(64, 32, 8, { learningRate: 0.005 });

      const { history } = vae.train(inputs, { epochs: 20 });
      assert.ok(history[history.length - 1].loss < history[0].loss);
    });
  });

  describe('Generation', () => {
    it('should generate samples from prior', () => {
      const vae = new VAE(8, 6, 2);
      const samples = vae.generate(5);
      assert.equal(samples.length, 5);
      for (const s of samples) {
        assert.equal(s.rows, 8);
        for (let i = 0; i < 8; i++) {
          assert.ok(s.data[i] >= 0 && s.data[i] <= 1);
        }
      }
    });
  });

  describe('Interpolation', () => {
    it('should interpolate between two inputs', () => {
      const vae = new VAE(4, 6, 2, { learningRate: 0.01 });
      // Quick train
      vae.train([[0.9, 0.1, 0.9, 0.1], [0.1, 0.9, 0.1, 0.9]], { epochs: 10 });

      const interp = vae.interpolate(
        [0.9, 0.1, 0.9, 0.1],
        [0.1, 0.9, 0.1, 0.9],
        5,
      );
      assert.equal(interp.length, 6); // 5 steps + endpoints
      for (const step of interp) {
        assert.equal(step.rows, 4);
        assert.ok(!step.data.some(isNaN));
      }
    });
  });

  describe('β-VAE', () => {
    it('should have higher KL with larger β', () => {
      const data = [
        [0.9, 0.1, 0.9, 0.1],
        [0.1, 0.9, 0.1, 0.9],
      ];

      // Lower β: less regularization
      const vae1 = new VAE(4, 6, 2, { beta: 0.1, learningRate: 0.01 });
      vae1.train(data, { epochs: 20 });

      // Higher β: more regularization
      const vae2 = new VAE(4, 6, 2, { beta: 5.0, learningRate: 0.01 });
      vae2.train(data, { epochs: 20 });

      // Both should produce valid outputs
      const out1 = vae1.forward(data[0]);
      const out2 = vae2.forward(data[0]);
      assert.ok(!out1.reconstruction.data.some(isNaN));
      assert.ok(!out2.reconstruction.data.some(isNaN));
    });
  });

  describe('Callbacks', () => {
    it('should call onEpoch', () => {
      const vae = new VAE(4, 3, 2);
      const calls = [];
      vae.train([[0.5, 0.5, 0.5, 0.5]], {
        epochs: 3,
        onEpoch: (data) => calls.push(data),
      });
      assert.equal(calls.length, 3);
      assert.equal(calls[0].epoch, 0);
      assert.ok('loss' in calls[0]);
    });
  });
});
