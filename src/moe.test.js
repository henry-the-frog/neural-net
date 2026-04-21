// moe.test.js — Tests for Mixture of Experts
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MixtureOfExperts } from './moe.js';
import { Matrix } from './matrix.js';

describe('Mixture of Experts', () => {
  describe('constructor', () => {
    it('creates with valid config', () => {
      const moe = new MixtureOfExperts(8, 4, 2);
      assert.equal(moe.numExperts, 4);
      assert.equal(moe.topK, 2);
    });

    it('rejects topK > numExperts', () => {
      assert.throws(() => new MixtureOfExperts(8, 4, 5));
    });

    it('rejects topK < 1', () => {
      assert.throws(() => new MixtureOfExperts(8, 4, 0));
    });
  });

  describe('forward', () => {
    it('produces correct output shape', () => {
      const moe = new MixtureOfExperts(8, 4, 2, 16);
      const input = Matrix.random(5, 8); // 5 tokens
      const output = moe.forward(input);
      assert.equal(output.rows, 5);
      assert.equal(output.cols, 8);
    });

    it('output values are finite', () => {
      const moe = new MixtureOfExperts(4, 3, 2, 8);
      const input = Matrix.random(3, 4);
      const output = moe.forward(input);
      for (let i = 0; i < output.rows; i++)
        for (let j = 0; j < output.cols; j++)
          assert.ok(isFinite(output.get(i, j)));
    });

    it('different inputs produce different outputs', () => {
      const moe = new MixtureOfExperts(4, 4, 2, 8);
      const in1 = Matrix.random(1, 4);
      const in2 = Matrix.random(1, 4);
      const out1 = moe.forward(in1);
      const out2 = moe.forward(in2);
      let diff = 0;
      for (let d = 0; d < 4; d++) diff += Math.abs(out1.get(0, d) - out2.get(0, d));
      assert.ok(diff > 1e-6, 'Different inputs should give different outputs');
    });

    it('topK=1 (Switch Transformer style)', () => {
      const moe = new MixtureOfExperts(4, 8, 1, 8);
      const input = Matrix.random(10, 4);
      const output = moe.forward(input);
      assert.equal(output.rows, 10);
      assert.equal(output.cols, 4);
    });
  });

  describe('routing stats', () => {
    it('tracks expert utilization', () => {
      const moe = new MixtureOfExperts(4, 4, 2);
      const input = Matrix.random(20, 4);
      moe.forward(input);

      const stats = moe.routingStats();
      assert.equal(stats.length, 4);
      
      const totalRouted = stats.reduce((sum, s) => sum + s.count, 0);
      assert.equal(totalRouted, 20 * 2, 'Total should be tokens × topK');
      
      console.log('  Routing distribution:');
      for (const s of stats) console.log(`    Expert ${s.expert}: ${s.count} (${s.pct})`);
    });

    it('resetStats clears counts', () => {
      const moe = new MixtureOfExperts(4, 4, 2);
      moe.forward(Matrix.random(5, 4));
      moe.resetStats();
      const stats = moe.routingStats();
      assert.ok(stats.every(s => s.count === 0));
    });
  });

  describe('load balancing', () => {
    it('perfect balance gives loss = 1', () => {
      const moe = new MixtureOfExperts(4, 4, 2);
      // Simulate perfect balance: each expert selected equal times
      moe._routingStats = [10, 10, 10, 10];
      const loss = moe.loadBalanceLoss();
      // 4 * Σ(0.25²) = 4 * 4 * 0.0625 = 1.0
      assert.ok(Math.abs(loss - 1.0) < 0.01, `Perfect balance loss should be ~1.0, got ${loss}`);
    });

    it('perfect imbalance gives high loss', () => {
      const moe = new MixtureOfExperts(4, 4, 2);
      moe._routingStats = [40, 0, 0, 0]; // all tokens → expert 0
      const loss = moe.loadBalanceLoss();
      // 4 * (1.0² + 0 + 0 + 0) = 4.0
      assert.equal(loss, 4.0, 'Perfect imbalance should give max loss');
    });
  });

  describe('parameter efficiency', () => {
    it('MoE has more total params but fewer active params', () => {
      const moe = new MixtureOfExperts(16, 8, 2, 32);
      const totalParams = moe.paramCount();
      const activeParams = moe.activeParamsPerToken();
      
      console.log(`  Total params: ${totalParams}`);
      console.log(`  Active per token: ${activeParams} (${(activeParams/totalParams*100).toFixed(1)}%)`);
      
      assert.ok(activeParams < totalParams, 'Active params should be less than total');
      // With 8 experts and top-2: active ≈ 2/8 = 25% of expert params
      assert.ok(activeParams / totalParams < 0.5, 'Should use < 50% of total params');
    });
  });
});
