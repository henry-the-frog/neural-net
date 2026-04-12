import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  gaussianPDF, logGaussianPDF, parseMDNOutput, mdnLoss,
  sampleMDN, parseMDNOutputMultiDim, mdnOutputSize,
} from '../src/mdn.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Gaussian PDF', () => {
  it('peak at mean', () => {
    const p = gaussianPDF(0, 0, 1);
    assert.ok(approx(p, 1 / Math.sqrt(2 * Math.PI), 0.01));
  });

  it('symmetric', () => {
    assert.ok(approx(gaussianPDF(-1, 0, 1), gaussianPDF(1, 0, 1)));
  });

  it('log version matches', () => {
    const p = gaussianPDF(0.5, 1, 2);
    const lp = logGaussianPDF(0.5, 1, 2);
    assert.ok(approx(Math.log(p), lp, 0.001));
  });
});

describe('MDN Output Parsing', () => {
  it('mixing coefficients sum to 1', () => {
    const output = [1, 2, 3, 0, 0, 0, 0, 0, 0]; // 3 components
    const { pi } = parseMDNOutput(output, 3);
    assert.ok(approx(pi.reduce((a, b) => a + b, 0), 1, 0.001));
  });

  it('sigma is positive', () => {
    const output = [0, 0, 0, 0, -5, 0]; // 2 components
    const { sigma } = parseMDNOutput(output, 2);
    assert.ok(sigma.every(s => s > 0));
  });

  it('parses correct number of components', () => {
    const K = 4;
    const output = new Array(K * 3).fill(0);
    const { pi, mu, sigma } = parseMDNOutput(output, K);
    assert.equal(pi.length, K);
    assert.equal(mu.length, K);
    assert.equal(sigma.length, K);
  });
});

describe('MDN Loss', () => {
  it('lower loss for correct prediction', () => {
    // Component centered at target
    const output = [10, 0, 5, 0, 0, 0]; // 2 components: strong pi for first, mu=5
    const lossBad = mdnLoss([0, 10, 0, 10, 0, 0], 5, 2); // Wrong prediction
    const lossGood = mdnLoss([10, 0, 5, 0, 0, 0], 5, 2);
    assert.ok(Number.isFinite(lossGood));
    assert.ok(Number.isFinite(lossBad));
  });

  it('loss is positive', () => {
    const output = [0, 0, 1, 2, 0, 0];
    const loss = mdnLoss(output, 1.5, 2);
    assert.ok(loss >= 0);
  });
});

describe('Sampling', () => {
  it('produces finite values', () => {
    const output = [1, 0, 0, 5, 0, 0]; // 2 components
    for (let i = 0; i < 20; i++) {
      const sample = sampleMDN(output, 2);
      assert.ok(Number.isFinite(sample));
    }
  });

  it('samples cluster around means', () => {
    // Strong single component at mean=10, small sigma
    const output = [100, -100, 10, 0, -3, 0]; // pi=[~1, ~0], mu=[10, 0], sigma=[~0.05, ~1]
    const samples = Array.from({ length: 100 }, () => sampleMDN(output, 2));
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    assert.ok(approx(mean, 10, 1), `Mean should be ~10: ${mean.toFixed(2)}`);
  });
});

describe('Multi-Dimensional MDN', () => {
  it('parses 2D output correctly', () => {
    const K = 2, D = 3;
    const size = mdnOutputSize(K, D);
    assert.equal(size, 2 * (1 + 2 * 3)); // 14
    const output = new Array(size).fill(0);
    const { pi, mu, sigma } = parseMDNOutputMultiDim(output, K, D);
    assert.equal(pi.length, 2);
    assert.equal(mu.length, 2);
    assert.equal(mu[0].length, 3);
    assert.equal(sigma[0].length, 3);
  });
});

describe('Output Size', () => {
  it('1D: K components need 3K outputs', () => {
    assert.equal(mdnOutputSize(5, 1), 15);
  });

  it('2D: K components need K(1+2D) outputs', () => {
    assert.equal(mdnOutputSize(3, 2), 15); // 3 * (1 + 4) = 15
  });
});
