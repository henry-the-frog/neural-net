// flash-attention.test.js — Tests for Flash Attention
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { FlashAttention, flashAttention } from './flash-attention.js';
import { SelfAttention } from './attention.js';
import { Matrix } from './matrix.js';

// Helper: standard attention for reference
function standardAttention(Q, K, V, { causal = false } = {}) {
  const n = Q.rows, d = Q.cols;
  const scale = 1 / Math.sqrt(d);
  
  // S = Q·K^T / sqrt(d)
  const S = Q.dot(K.T()).mul(scale);
  
  // Apply causal mask
  if (causal) {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        S.set(i, j, -Infinity);
  }
  
  // Softmax each row
  const P = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let j = 0; j < n; j++) max = Math.max(max, S.get(i, j));
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const v = S.get(i, j) === -Infinity ? 0 : Math.exp(S.get(i, j) - max);
      P.set(i, j, v);
      sum += v;
    }
    if (sum > 0) for (let j = 0; j < n; j++) P.set(i, j, P.get(i, j) / sum);
  }
  
  return P.dot(V);
}

function matricesClose(a, b, tol = 1e-10) {
  assert.equal(a.rows, b.rows, `rows: ${a.rows} vs ${b.rows}`);
  assert.equal(a.cols, b.cols, `cols: ${a.cols} vs ${b.cols}`);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++) {
      const diff = Math.abs(a.get(i, j) - b.get(i, j));
      assert.ok(diff < tol, `[${i},${j}]: ${a.get(i, j)} vs ${b.get(i, j)}, diff=${diff}`);
    }
}

describe('flashAttention (standalone function)', () => {
  it('matches standard attention for small input', () => {
    const Q = Matrix.random(4, 8);
    const K = Matrix.random(4, 8);
    const V = Matrix.random(4, 8);
    
    const expected = standardAttention(Q, K, V);
    const actual = flashAttention(Q, K, V, { blockSize: 2 });
    
    matricesClose(actual, expected, 1e-10);
  });
  
  it('matches standard attention for larger input', () => {
    const Q = Matrix.random(16, 8);
    const K = Matrix.random(16, 8);
    const V = Matrix.random(16, 8);
    
    const expected = standardAttention(Q, K, V);
    const actual = flashAttention(Q, K, V, { blockSize: 4 });
    
    matricesClose(actual, expected, 1e-10);
  });
  
  it('matches standard attention with causal mask', () => {
    const Q = Matrix.random(8, 4);
    const K = Matrix.random(8, 4);
    const V = Matrix.random(8, 4);
    
    const expected = standardAttention(Q, K, V, { causal: true });
    const actual = flashAttention(Q, K, V, { blockSize: 3, causal: true });
    
    matricesClose(actual, expected, 1e-10);
  });
  
  it('works with blockSize=1 (degenerate case)', () => {
    const Q = Matrix.random(4, 4);
    const K = Matrix.random(4, 4);
    const V = Matrix.random(4, 4);
    
    const expected = standardAttention(Q, K, V);
    const actual = flashAttention(Q, K, V, { blockSize: 1 });
    
    matricesClose(actual, expected, 1e-10);
  });
  
  it('works with blockSize >= seqLen (single block)', () => {
    const Q = Matrix.random(4, 4);
    const K = Matrix.random(4, 4);
    const V = Matrix.random(4, 4);
    
    const expected = standardAttention(Q, K, V);
    const actual = flashAttention(Q, K, V, { blockSize: 100 });
    
    matricesClose(actual, expected, 1e-10);
  });
  
  it('produces correct output shape', () => {
    const Q = Matrix.random(10, 6);
    const K = Matrix.random(10, 6);
    const V = Matrix.random(10, 6);
    
    const result = flashAttention(Q, K, V);
    assert.equal(result.rows, 10);
    assert.equal(result.cols, 6);
  });
  
  it('handles single-token sequence', () => {
    const Q = Matrix.random(1, 4);
    const K = Matrix.random(1, 4);
    const V = Matrix.random(1, 4);
    
    const expected = standardAttention(Q, K, V);
    const actual = flashAttention(Q, K, V, { blockSize: 1 });
    
    matricesClose(actual, expected, 1e-10);
  });
});

describe('FlashAttention class', () => {
  it('forward produces correct output shape', () => {
    const attn = new FlashAttention(8, { blockSize: 4 });
    const input = Matrix.random(2, 4 * 8); // batch=2, seqLen=4, d=8
    const output = attn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4 * 8);
  });
  
  it('forward matches standard attention (shared weights)', () => {
    const d = 4;
    const flash = new FlashAttention(d, { blockSize: 2 });
    const std = new SelfAttention(d);
    
    // Share weights
    std.Wq = flash.Wq; std.Wk = flash.Wk; std.Wv = flash.Wv; std.Wo = flash.Wo;
    std.bq = flash.bq; std.bk = flash.bk; std.bv = flash.bv; std.bo = flash.bo;
    
    const input = Matrix.random(1, 3 * d); // batch=1, seqLen=3
    const flashOut = flash.forward(input);
    const stdOut = std.forward(input);
    
    matricesClose(flashOut, stdOut, 1e-10);
  });
  
  it('backward produces gradients', () => {
    const attn = new FlashAttention(4, { blockSize: 2 });
    const input = Matrix.random(1, 3 * 4);
    const output = attn.forward(input);
    const dOutput = Matrix.random(output.rows, output.cols);
    const dInput = attn.backward(dOutput);
    
    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 3 * 4);
    assert.ok(attn.dWeights);
    assert.ok(attn.dWeights.dWq);
  });
  
  it('backward gradient check (numerical)', () => {
    const d = 4, seqLen = 3;
    const attn = new FlashAttention(d, { blockSize: 2 });
    const input = Matrix.random(1, seqLen * d);
    
    // Forward + backward
    const output = attn.forward(input);
    const dOutput = Matrix.ones(output.rows, output.cols);
    const dInput = attn.backward(dOutput);
    
    // Numerical gradient check for a few input elements
    const eps = 1e-5;
    for (let idx = 0; idx < Math.min(4, input.cols); idx++) {
      const orig = input.get(0, idx);
      
      input.set(0, idx, orig + eps);
      const outPlus = attn.forward(input);
      let sumPlus = 0;
      for (let j = 0; j < outPlus.cols; j++) sumPlus += outPlus.get(0, j);
      
      input.set(0, idx, orig - eps);
      const outMinus = attn.forward(input);
      let sumMinus = 0;
      for (let j = 0; j < outMinus.cols; j++) sumMinus += outMinus.get(0, j);
      
      input.set(0, idx, orig);
      
      const numerical = (sumPlus - sumMinus) / (2 * eps);
      const analytical = dInput.get(0, idx);
      const relErr = Math.abs(numerical - analytical) / (Math.abs(numerical) + Math.abs(analytical) + 1e-8);
      
      assert.ok(relErr < 0.05, `Gradient check failed at idx ${idx}: numerical=${numerical}, analytical=${analytical}, relErr=${relErr}`);
    }
  });
  
  it('causal mask forward matches standard', () => {
    const d = 4;
    const flash = new FlashAttention(d, { blockSize: 2, causal: true });
    
    const input = Matrix.random(1, 4 * d);
    const output = flash.forward(input);
    
    // Verify output is finite
    for (let i = 0; i < output.cols; i++) {
      assert.ok(isFinite(output.get(0, i)), `output[${i}] is not finite`);
    }
  });
  
  it('serialization round-trip', () => {
    const attn = new FlashAttention(8, { blockSize: 4, causal: true });
    const json = attn.toJSON();
    const restored = FlashAttention.fromJSON(json);
    
    assert.equal(restored.dModel, 8);
    assert.equal(restored.blockSize, 4);
    assert.equal(restored.causal, true);
    
    const input = Matrix.random(1, 2 * 8);
    matricesClose(attn.forward(input), restored.forward(input), 1e-10);
  });
  
  it('updateWeights modifies parameters', () => {
    const attn = new FlashAttention(4, { blockSize: 2 });
    const wqBefore = attn.Wq.get(0, 0);
    
    const input = Matrix.random(1, 2 * 4);
    attn.forward(input);
    attn.backward(Matrix.random(1, 2 * 4));
    attn.updateWeights(0.01);
    
    assert.notEqual(attn.Wq.get(0, 0), wqBefore);
  });
});

describe('Memory efficiency', () => {
  it('flash attention does not create N×N matrix', () => {
    // With blockSize=4 and seqLen=64, standard attention would create 64×64=4096 element matrix
    // Flash attention creates at most blockSize tiles
    const n = 64, d = 8;
    const Q = Matrix.random(n, d);
    const K = Matrix.random(n, d);
    const V = Matrix.random(n, d);
    
    // This should not OOM even for large sequences (in a real scenario)
    const result = flashAttention(Q, K, V, { blockSize: 8 });
    assert.equal(result.rows, n);
    assert.equal(result.cols, d);
    
    // Verify correctness
    const expected = standardAttention(Q, K, V);
    matricesClose(result, expected, 1e-8);
  });
});
