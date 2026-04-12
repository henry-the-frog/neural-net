// transformer-stress.test.js — Deep stress tests for Transformer components
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { PositionalEncoding, LayerNorm, TransformerEncoderBlock } from '../src/transformer.js';
import { MultiHeadAttention } from '../src/attention.js';
import { Matrix } from '../src/matrix.js';

describe('Positional Encoding', () => {
  it('output has same shape as input', () => {
    const pe = new PositionalEncoding(8, 100);
    const input = Matrix.random(2, 24); // batch=2, seqLen=3, dModel=8
    const output = pe.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('different positions have different encodings', () => {
    const pe = new PositionalEncoding(16, 100);
    // Compare PE at position 0 vs position 10
    let same = true;
    for (let d = 0; d < 16; d++) {
      if (Math.abs(pe.pe.get(0, d) - pe.pe.get(10, d)) > 1e-6) same = false;
    }
    assert.ok(!same, 'Different positions should have different encodings');
  });

  it('sin/cos pattern: even dims use sin, odd use cos', () => {
    const pe = new PositionalEncoding(8, 10);
    const pos = 5;
    // Check dim 0 (sin) and dim 1 (cos) use same frequency
    const angle = pos / Math.pow(10000, 0 / 8);
    assert.ok(Math.abs(pe.pe.get(pos, 0) - Math.sin(angle)) < 1e-6);
    assert.ok(Math.abs(pe.pe.get(pos, 1) - Math.cos(angle)) < 1e-6);
  });

  it('PE values are bounded in [-1, 1]', () => {
    const pe = new PositionalEncoding(32, 200);
    for (let pos = 0; pos < 200; pos++) {
      for (let d = 0; d < 32; d++) {
        const v = pe.pe.get(pos, d);
        assert.ok(v >= -1.01 && v <= 1.01, `PE(${pos},${d})=${v.toFixed(4)} out of bounds`);
      }
    }
  });

  it('backward passes gradient through (additive)', () => {
    const pe = new PositionalEncoding(4, 10);
    const grad = Matrix.random(1, 8);
    const result = pe.backward(grad);
    // Should pass through unchanged
    for (let i = 0; i < 8; i++) {
      assert.equal(result.get(0, i), grad.get(0, i));
    }
  });
});

describe('LayerNorm', () => {
  it('output has zero mean per position', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.fromArray([[1, 5, 3, 7]]); // seqLen=1, dModel=4
    const output = ln.forward(input);
    let sum = 0;
    for (let d = 0; d < 4; d++) sum += output.get(0, d);
    assert.ok(Math.abs(sum) < 0.01, `Mean should be ~0: ${(sum / 4).toFixed(4)}`);
  });

  it('output has unit variance per position', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.fromArray([[1, 5, 3, 7]]);
    const output = ln.forward(input);
    let mean = 0;
    for (let d = 0; d < 4; d++) mean += output.get(0, d);
    mean /= 4;
    let variance = 0;
    for (let d = 0; d < 4; d++) variance += (output.get(0, d) - mean) ** 2;
    variance /= 4;
    assert.ok(Math.abs(variance - 1) < 0.2, `Variance should be ~1: ${variance.toFixed(4)}`);
  });

  it('identical inputs produce identical normalized outputs', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.fromArray([[3, 3, 3, 3]]);
    const output = ln.forward(input);
    // All same → mean = 3, std ≈ 0 → output ≈ 0 (or NaN if not handled)
    assert.ok(output.data.every(Number.isFinite), 'Constant input should produce finite output');
  });
});

describe('MultiHeadAttention', () => {
  it('output has same shape as input', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(1, 24); // batch=1, seqLen=3, dModel=8
    const output = mha.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 24);
  });

  it('attention output is finite', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(2, 16); // batch=2, seqLen=2, dModel=8
    const output = mha.forward(input);
    assert.ok(output.data.every(Number.isFinite), 'MHA output should be finite');
  });

  it('different number of heads changes output', () => {
    const input = Matrix.random(1, 16); // seqLen=2, dModel=8
    
    const mha2 = new MultiHeadAttention(8, 2);
    const mha4 = new MultiHeadAttention(8, 4);
    
    const out2 = mha2.forward(input);
    const out4 = mha4.forward(input);
    
    // Different configs should produce different outputs (different random weights)
    let diff = 0;
    for (let i = 0; i < out2.data.length; i++) {
      diff += Math.abs(out2.data[i] - out4.data[i]);
    }
    assert.ok(diff > 0.01, 'Different head configs should differ');
  });

  it('handles single token sequence', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(1, 8); // seqLen=1
    const output = mha.forward(input);
    assert.equal(output.cols, 8);
    assert.ok(output.data.every(Number.isFinite));
  });
});

describe('TransformerEncoderBlock', () => {
  it('output has same shape as input', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const input = Matrix.random(1, 24); // seqLen=3
    const output = block.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 24);
  });

  it('output is finite for random input', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const input = Matrix.random(2, 16);
    const output = block.forward(input);
    assert.ok(output.data.every(Number.isFinite), 'Encoder output should be finite');
  });

  it('stacking 3 blocks produces finite output', () => {
    const blocks = [
      new TransformerEncoderBlock(8, 2),
      new TransformerEncoderBlock(8, 2),
      new TransformerEncoderBlock(8, 2),
    ];
    let x = Matrix.random(1, 24);
    for (const block of blocks) x = block.forward(x);
    assert.ok(x.data.every(Number.isFinite), 'Stacked encoder output should be finite');
  });

  it('different inputs produce different outputs', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const in1 = Matrix.random(1, 16);
    const in2 = Matrix.random(1, 16);
    const out1 = block.forward(in1);
    const out2 = block.forward(in2);
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01, 'Different inputs should produce different outputs');
  });

  it('backward produces finite gradients', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const input = Matrix.random(1, 16);
    block.forward(input);
    const dOutput = Matrix.random(1, 16);
    const dInput = block.backward(dOutput);
    assert.ok(dInput.data.every(Number.isFinite), 'Gradients should be finite');
  });
});
