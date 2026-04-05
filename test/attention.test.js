import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, SelfAttention, MultiHeadAttention, Network, Dense } from '../src/index.js';

describe('SelfAttention', () => {
  it('forward produces correct shape', () => {
    const attn = new SelfAttention(8);
    // batch=2, seqLen=3, dModel=8 → input [2, 24]
    const input = Matrix.random(2, 24);
    const output = attn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('different sequences produce different outputs', () => {
    const attn = new SelfAttention(4);
    const input1 = Matrix.random(1, 12); // seq_len=3, d=4
    const input2 = Matrix.random(1, 12);
    const out1 = attn.forward(input1);
    const out2 = attn.forward(input2);
    let diff = 0;
    for (let j = 0; j < 12; j++) diff += Math.abs(out1.get(0, j) - out2.get(0, j));
    assert.ok(diff > 0.01, 'Different inputs should produce different outputs');
  });

  it('backward produces correct gradient shape', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(2, 12);
    attn.forward(input);
    const dOutput = Matrix.random(2, 12);
    const dInput = attn.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 12);
  });

  it('param count', () => {
    const attn = new SelfAttention(8);
    // 4 × (8×8 + 8) = 4 × 72 = 288
    assert.equal(attn.paramCount(), 288);
  });

  it('single token sequence', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 4); // seq_len=1
    const output = attn.forward(input);
    assert.equal(output.cols, 4);
  });

  it('attention weights sum to 1 per row', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 12); // seq_len=3
    attn.forward(input);
    // Check cached attention weights
    const weights = attn._cache.allAttn[0]; // [3, 3]
    for (let i = 0; i < weights.rows; i++) {
      let sum = 0;
      for (let j = 0; j < weights.cols; j++) sum += weights.get(i, j);
      assert.ok(Math.abs(sum - 1) < 1e-6, `Row ${i} attention should sum to 1, got ${sum}`);
    }
  });

  it('gradients are non-zero', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(2, 8);
    attn.forward(input);
    attn.backward(Matrix.random(2, 8));
    
    let wGradSum = 0;
    for (let i = 0; i < attn._dWq.rows; i++)
      for (let j = 0; j < attn._dWq.cols; j++)
        wGradSum += Math.abs(attn._dWq.get(i, j));
    assert.ok(wGradSum > 0, 'Weight gradients should be non-zero');
  });
});

describe('MultiHeadAttention', () => {
  it('forward produces correct shape', () => {
    const mha = new MultiHeadAttention(8, 2); // 2 heads, headDim=4
    const input = Matrix.random(2, 24); // batch=2, seq=3, d=8
    const output = mha.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('rejects non-divisible dModel/numHeads', () => {
    assert.throws(() => new MultiHeadAttention(7, 3), /divisible/);
  });

  it('backward produces correct gradient shape', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(2, 24);
    mha.forward(input);
    const dInput = mha.backward(Matrix.random(2, 24));
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 24);
  });

  it('param count same as single attention', () => {
    const mha = new MultiHeadAttention(8, 4);
    // 4 × (8×8 + 8) = 288
    assert.equal(mha.paramCount(), 288);
  });

  it('4 heads vs 1 head produce different outputs', () => {
    const mha1 = new MultiHeadAttention(8, 1);
    const mha4 = new MultiHeadAttention(8, 4);
    const input = Matrix.random(1, 16); // seq=2, d=8
    const out1 = mha1.forward(input);
    const out4 = mha4.forward(input);
    // They have different random weights, so outputs differ
    let diff = 0;
    for (let j = 0; j < 16; j++) diff += Math.abs(out1.get(0, j) - out4.get(0, j));
    assert.ok(diff > 0.01);
  });

  it('works with Network.add()', () => {
    const net = new Network();
    net.add(new MultiHeadAttention(4, 2));
    net.add(new Dense(8, 2, 'softmax')); // seq_len=2, d_model=4 → flatten to 8
    net.loss('crossEntropy');
    
    const input = Matrix.random(3, 8); // batch=3, seq=2, d=4
    const output = net.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 2);
  });
});
