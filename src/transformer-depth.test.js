// transformer-depth.test.js — Transformer attention + components depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { SelfAttention, MultiHeadAttention } from './attention.js';
import { PositionalEncoding, LayerNorm } from './transformer.js';
import { Matrix } from './matrix.js';

describe('SelfAttention Shapes', () => {
  it('output shape matches input shape', () => {
    const attn = new SelfAttention(8);
    attn.training = false;
    // batch=2, seqLen=3, dModel=8 → input cols = 3*8 = 24
    const input = Matrix.random(2, 3 * 8);
    const output = attn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('single token sequence', () => {
    const attn = new SelfAttention(4);
    attn.training = false;
    const input = Matrix.random(1, 4); // 1 token
    const output = attn.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 4);
  });
});

describe('MultiHeadAttention Shapes', () => {
  it('multi-head preserves shape', () => {
    const mha = new MultiHeadAttention(8, 2); // dModel=8, heads=2
    mha.training = false;
    const input = Matrix.random(2, 3 * 8);
    const output = mha.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('4 heads', () => {
    const mha = new MultiHeadAttention(16, 4);
    mha.training = false;
    const input = Matrix.random(1, 5 * 16);
    const output = mha.forward(input);
    assert.equal(output.cols, 5 * 16);
  });
});

describe('PositionalEncoding', () => {
  it('adds position information', () => {
    const pe = new PositionalEncoding(8, 100);
    const input = Matrix.zeros(1, 3 * 8); // 3 tokens, 8 dims
    const output = pe.forward(input);
    
    // Zero input + PE should give PE values
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 24);
    
    // First position PE should not be all zeros
    let hasNonZero = false;
    for (let d = 0; d < 8; d++) {
      if (Math.abs(output.get(0, d)) > 1e-6) hasNonZero = true;
    }
    assert.ok(hasNonZero, 'PE should add non-zero values');
  });

  it('different positions get different encodings', () => {
    const pe = new PositionalEncoding(8, 100);
    const input = Matrix.zeros(1, 2 * 8); // 2 tokens
    const output = pe.forward(input);
    
    let same = true;
    for (let d = 0; d < 8; d++) {
      if (Math.abs(output.get(0, d) - output.get(0, 8 + d)) > 1e-6) {
        same = false;
        break;
      }
    }
    assert.ok(!same, 'Different positions should have different encodings');
  });
});

describe('LayerNorm', () => {
  it('output has approximately zero mean', () => {
    const ln = new LayerNorm(8);
    const input = Matrix.random(2, 8); // 1 token per sample
    const output = ln.forward(input);
    
    // Check mean is near zero for each sample
    for (let b = 0; b < 2; b++) {
      let sum = 0;
      for (let d = 0; d < 8; d++) sum += output.get(b, d);
      const mean = sum / 8;
      assert.ok(Math.abs(mean) < 0.5, `Mean should be near zero, got ${mean}`);
    }
  });

  it('preserves shape', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.random(3, 8); // 2 tokens, 4 dims
    const output = ln.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 8);
  });
});
