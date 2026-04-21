// gqa-attention.test.js — Tests for Grouped Query Attention + KV-cache
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { GroupedQueryAttention } from './gqa-attention.js';
import { Matrix } from './matrix.js';

describe('GroupedQueryAttention', () => {
  describe('constructor validation', () => {
    it('rejects dModel not divisible by numQHeads', () => {
      assert.throws(() => new GroupedQueryAttention(10, 3, 1), /divisible/);
    });

    it('rejects numQHeads not divisible by numKVHeads', () => {
      assert.throws(() => new GroupedQueryAttention(12, 4, 3), /divisible/);
    });

    it('accepts standard MHA config (numQHeads === numKVHeads)', () => {
      const gqa = new GroupedQueryAttention(8, 4, 4);
      assert.equal(gqa.groupSize, 1);
    });

    it('accepts MQA config (numKVHeads === 1)', () => {
      const gqa = new GroupedQueryAttention(8, 4, 1);
      assert.equal(gqa.groupSize, 4);
    });

    it('accepts GQA config', () => {
      const gqa = new GroupedQueryAttention(8, 4, 2);
      assert.equal(gqa.groupSize, 2);
    });
  });

  describe('forward pass', () => {
    it('produces correct output shape for single token', () => {
      const gqa = new GroupedQueryAttention(8, 4, 2);
      const input = Matrix.random(1, 8); // batch=1, seqLen=1
      const output = gqa.forward(input);
      assert.equal(output.rows, 1);
      assert.equal(output.cols, 8);
    });

    it('produces correct output shape for sequence', () => {
      const gqa = new GroupedQueryAttention(8, 4, 2);
      const input = Matrix.random(2, 24); // batch=2, seqLen=3, dModel=8
      const output = gqa.forward(input);
      assert.equal(output.rows, 2);
      assert.equal(output.cols, 24);
    });

    it('output values are finite', () => {
      const gqa = new GroupedQueryAttention(16, 4, 2);
      const input = Matrix.random(1, 48); // seqLen=3
      const output = gqa.forward(input);
      for (let i = 0; i < output.rows; i++)
        for (let j = 0; j < output.cols; j++)
          assert.ok(isFinite(output.get(i, j)), `NaN/Inf at (${i},${j})`);
    });

    it('standard MHA (groupSize=1) produces same shape as MQA (groupSize=numQHeads)', () => {
      const mha = new GroupedQueryAttention(8, 4, 4);
      const mqa = new GroupedQueryAttention(8, 4, 1);
      const input = Matrix.random(1, 16); // seqLen=2
      const out1 = mha.forward(input);
      const out2 = mqa.forward(input);
      assert.equal(out1.rows, out2.rows);
      assert.equal(out1.cols, out2.cols);
    });
  });

  describe('causal masking', () => {
    it('causal attention: first token only attends to itself', () => {
      // With causal=true, first position can only see position 0
      const gqa = new GroupedQueryAttention(4, 2, 1, { causal: true });
      const input1 = Matrix.random(1, 4); // single token
      const out1 = gqa.forward(input1);

      // Adding a second token shouldn't change the first token's output
      // (if using KV-cache correctly)
      gqa.clearCache();
      const input2 = Matrix.random(1, 8); // two tokens
      // Copy first token from input1
      for (let d = 0; d < 4; d++) input2.set(0, d, input1.get(0, d));
      const out2 = gqa.forward(input2);

      // First token output should be identical
      for (let d = 0; d < 4; d++) {
        assert.ok(
          Math.abs(out1.get(0, d) - out2.get(0, d)) < 1e-6,
          `Causal violation at d=${d}: ${out1.get(0, d)} vs ${out2.get(0, d)}`
        );
      }
    });

    it('non-causal attention differs from causal', () => {
      const causal = new GroupedQueryAttention(4, 2, 1, { causal: true });
      const noncausal = new GroupedQueryAttention(4, 2, 1, { causal: false });
      // Copy weights
      noncausal.Wq = causal.Wq;
      noncausal.Wk = causal.Wk;
      noncausal.Wv = causal.Wv;
      noncausal.Wo = causal.Wo;
      noncausal.bq = causal.bq;
      noncausal.bk = causal.bk;
      noncausal.bv = causal.bv;
      noncausal.bo = causal.bo;

      const input = Matrix.random(1, 16); // 4 tokens (more tokens = more likely difference)
      const out1 = causal.forward(input);
      const out2 = noncausal.forward(input);

      // First token differs because causal sees only self, non-causal sees all
      let diff = 0;
      for (let d = 0; d < 4; d++) diff += Math.abs(out1.get(0, d) - out2.get(0, d));
      assert.ok(diff > 1e-6, 'First token should differ: causal sees only self, non-causal sees all');

      // Last token should be identical (sees all tokens in both modes)
      let lastDiff = 0;
      for (let d = 12; d < 16; d++) lastDiff += Math.abs(out1.get(0, d) - out2.get(0, d));
      assert.ok(lastDiff < 1e-6, 'Last token should be identical in both modes');
    });
  });

  describe('KV-cache', () => {
    it('incremental generation matches full-sequence computation', () => {
      const gqa = new GroupedQueryAttention(4, 2, 1, { causal: true });

      // Full sequence: 3 tokens at once
      const fullInput = Matrix.random(1, 12); // 3 tokens
      const fullOut = gqa.forward(fullInput);

      // Incremental: token by token with cache
      gqa.clearCache();
      const t0 = new Matrix(1, 4);
      const t1 = new Matrix(1, 4);
      const t2 = new Matrix(1, 4);
      for (let d = 0; d < 4; d++) {
        t0.set(0, d, fullInput.get(0, d));
        t1.set(0, d, fullInput.get(0, 4 + d));
        t2.set(0, d, fullInput.get(0, 8 + d));
      }

      const out0 = gqa.forward(t0, true);
      const out1 = gqa.forward(t1, true);
      const out2 = gqa.forward(t2, true);

      // Token 0 output should match
      for (let d = 0; d < 4; d++) {
        assert.ok(
          Math.abs(out0.get(0, d) - fullOut.get(0, d)) < 1e-5,
          `Token 0 mismatch at d=${d}`
        );
      }
      // Token 2 output should match
      for (let d = 0; d < 4; d++) {
        assert.ok(
          Math.abs(out2.get(0, d) - fullOut.get(0, 8 + d)) < 1e-5,
          `Token 2 mismatch at d=${d}`
        );
      }
    });

    it('cache grows with each token', () => {
      const gqa = new GroupedQueryAttention(4, 2, 1);
      const token = Matrix.random(1, 4);

      gqa.clearCache();
      gqa.forward(token, true);
      assert.equal(gqa.cacheStats().totalTokens, 1);

      gqa.forward(token, true);
      assert.equal(gqa.cacheStats().totalTokens, 2);

      gqa.forward(token, true);
      assert.equal(gqa.cacheStats().totalTokens, 3);
    });

    it('clearCache resets stats', () => {
      const gqa = new GroupedQueryAttention(4, 2, 1);
      gqa.forward(Matrix.random(1, 4), true);
      gqa.clearCache();
      assert.equal(gqa.cacheStats().totalTokens, 0);
    });

    it('cacheStats reports memory usage', () => {
      const gqa = new GroupedQueryAttention(8, 4, 2); // kvDim = 2 * (8/4) = 4
      gqa.forward(Matrix.random(1, 8), true); // 1 token
      const stats = gqa.cacheStats();
      assert.equal(stats.totalTokens, 1);
      // 1 token × kvDim(4) × 2(K+V) × 8 bytes = 64
      assert.equal(stats.memoryBytes, 64);
    });
  });

  describe('memory efficiency', () => {
    it('GQA uses less KV memory than MHA', () => {
      const mha = new GroupedQueryAttention(16, 8, 8); // standard: 8 KV heads
      const gqa = new GroupedQueryAttention(16, 8, 2); // GQA: 2 KV heads

      // KV weights for MHA: dModel × (2 * dModel) = 16 × 32 = 512 params
      // KV weights for GQA: dModel × (2 * kvDim) = 16 × (2*4) = 128 params
      assert.ok(gqa.Wk.cols < mha.Wk.cols, 'GQA Wk should have fewer columns');
      assert.ok(gqa.Wv.cols < mha.Wv.cols, 'GQA Wv should have fewer columns');

      // Cache token: MHA = 8 heads × 2 × 2 = 32 floats, GQA = 2 heads × 2 × 2 = 8 floats
      const token = Matrix.random(1, 16);
      mha.forward(token, true);
      gqa.forward(token, true);
      assert.ok(gqa.cacheStats().memoryBytes < mha.cacheStats().memoryBytes);
    });

    it('MQA is most memory efficient', () => {
      const gqa = new GroupedQueryAttention(16, 8, 2);
      const mqa = new GroupedQueryAttention(16, 8, 1); // single KV head

      const token = Matrix.random(1, 16);
      gqa.forward(token, true);
      mqa.forward(token, true);
      assert.ok(mqa.cacheStats().memoryBytes < gqa.cacheStats().memoryBytes);
    });
  });
});
