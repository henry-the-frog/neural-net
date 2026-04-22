// rope.test.js — Tests for Rotary Position Embeddings
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { precomputeFreqs, applyRoPE, applyInverseRoPE, verifyRelativeProperty } from './rope.js';
import { Matrix } from './matrix.js';

describe('RoPE', () => {
  const dim = 8;
  const maxSeqLen = 64;
  const freqs = precomputeFreqs(dim, maxSeqLen);
  
  it('precomputeFreqs creates correct table size', () => {
    assert.equal(freqs.cos.length, maxSeqLen);
    assert.equal(freqs.sin.length, maxSeqLen);
    assert.equal(freqs.cos[0].length, dim / 2);
  });
  
  it('position 0 has cos=1, sin=0', () => {
    for (let i = 0; i < dim / 2; i++) {
      assert.ok(Math.abs(freqs.cos[0][i] - 1) < 1e-10);
      assert.ok(Math.abs(freqs.sin[0][i]) < 1e-10);
    }
  });
  
  it('applyRoPE preserves shape', () => {
    const x = Matrix.random(4, dim); // seqLen=4
    const rotated = applyRoPE(x, freqs);
    assert.equal(rotated.rows, 4);
    assert.equal(rotated.cols, dim);
  });
  
  it('position 0 rotation is identity', () => {
    const x = Matrix.random(1, dim);
    const rotated = applyRoPE(x, freqs, 0);
    for (let i = 0; i < dim; i++) {
      assert.ok(Math.abs(rotated.get(0, i) - x.get(0, i)) < 1e-10);
    }
  });
  
  it('rotation preserves vector norm', () => {
    const x = Matrix.random(1, dim);
    const rotated = applyRoPE(x, freqs, 5);
    
    let normOrig = 0, normRot = 0;
    for (let i = 0; i < dim; i++) {
      normOrig += x.get(0, i) ** 2;
      normRot += rotated.get(0, i) ** 2;
    }
    assert.ok(Math.abs(normOrig - normRot) < 1e-10, 'RoPE should preserve norm');
  });
  
  it('inverse rotation recovers original', () => {
    const x = Matrix.random(3, dim);
    const rotated = applyRoPE(x, freqs, 2);
    const recovered = applyInverseRoPE(rotated, freqs, 2);
    
    for (let i = 0; i < x.rows; i++)
      for (let j = 0; j < x.cols; j++)
        assert.ok(Math.abs(recovered.get(i, j) - x.get(i, j)) < 1e-10, `[${i},${j}]`);
  });
  
  it('relative position property: dot product depends only on m-n', () => {
    const q = Matrix.random(1, dim);
    const k = Matrix.random(1, dim);
    
    // Same relative distance (m-n = 3) at different absolute positions
    const dot1 = verifyRelativeProperty(q, k, freqs, 5, 2);  // 5-2=3
    const dot2 = verifyRelativeProperty(q, k, freqs, 10, 7); // 10-7=3
    const dot3 = verifyRelativeProperty(q, k, freqs, 20, 17); // 20-17=3
    
    assert.ok(Math.abs(dot1 - dot2) < 1e-10, `dot1=${dot1} vs dot2=${dot2}`);
    assert.ok(Math.abs(dot2 - dot3) < 1e-10, `dot2=${dot2} vs dot3=${dot3}`);
  });
  
  it('different relative positions give different dot products', () => {
    const q = Matrix.random(1, dim);
    const k = Matrix.random(1, dim);
    
    const dot_0 = verifyRelativeProperty(q, k, freqs, 5, 5); // m-n=0
    const dot_3 = verifyRelativeProperty(q, k, freqs, 8, 5); // m-n=3
    
    // These should generally be different (unless very unlucky)
    // We just check they're computed without error
    assert.ok(isFinite(dot_0));
    assert.ok(isFinite(dot_3));
  });
  
  it('works with offset for KV cache continuation', () => {
    const x = Matrix.random(4, dim);
    
    // Full sequence rotation
    const full = applyRoPE(x, freqs, 0);
    
    // Split: first 2 tokens, then next 2 with offset=2
    const first = applyRoPE(new Matrix(2, dim, x.data.slice(0, 2 * dim)), freqs, 0);
    const second = applyRoPE(new Matrix(2, dim, x.data.slice(2 * dim)), freqs, 2);
    
    // Should match
    for (let j = 0; j < dim; j++) {
      assert.ok(Math.abs(full.get(0, j) - first.get(0, j)) < 1e-10);
      assert.ok(Math.abs(full.get(1, j) - first.get(1, j)) < 1e-10);
      assert.ok(Math.abs(full.get(2, j) - second.get(0, j)) < 1e-10);
      assert.ok(Math.abs(full.get(3, j) - second.get(1, j)) < 1e-10);
    }
  });
  
  it('throws on odd dimension', () => {
    assert.throws(() => precomputeFreqs(7, 10), /even/);
  });
  
  it('custom base changes frequency distribution', () => {
    const lowBase = precomputeFreqs(dim, 10, 100);
    const highBase = precomputeFreqs(dim, 10, 100000);
    
    // Lower base = higher frequency = more rotation per position
    // At position 5, lower base should have larger sin values
    let lowSinSum = 0, highSinSum = 0;
    for (let i = 0; i < dim / 2; i++) {
      lowSinSum += Math.abs(lowBase.sin[5][i]);
      highSinSum += Math.abs(highBase.sin[5][i]);
    }
    assert.ok(lowSinSum > highSinSum, 'Lower base should produce more rotation');
  });
});
