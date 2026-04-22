// multi-head-flash-attention.test.js — Tests for MultiHeadFlashAttention
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MultiHeadFlashAttention } from './multi-head-flash-attention.js';
import { MultiHeadAttention } from './attention.js';
import { TransformerEncoderBlock } from './transformer.js';
import { Matrix } from './matrix.js';

function matricesClose(a, b, tol = 1e-8) {
  assert.equal(a.rows, b.rows);
  assert.equal(a.cols, b.cols);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++) {
      const diff = Math.abs(a.get(i, j) - b.get(i, j));
      assert.ok(diff < tol, `[${i},${j}]: ${a.get(i, j)} vs ${b.get(i, j)}, diff=${diff}`);
    }
}

describe('MultiHeadFlashAttention', () => {
  it('produces correct output shape', () => {
    const attn = new MultiHeadFlashAttention(8, 2, { blockSize: 4 });
    const input = Matrix.random(2, 4 * 8); // batch=2, seqLen=4, d=8
    const output = attn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4 * 8);
  });
  
  it('matches MultiHeadAttention with shared weights', () => {
    const d = 8, heads = 2;
    const flash = new MultiHeadFlashAttention(d, heads, { blockSize: 2 });
    const std = new MultiHeadAttention(d, heads);
    
    // Share weights
    std.Wq = flash.Wq; std.Wk = flash.Wk; std.Wv = flash.Wv; std.Wo = flash.Wo;
    std.bq = flash.bq; std.bk = flash.bk; std.bv = flash.bv; std.bo = flash.bo;
    
    const input = Matrix.random(1, 4 * d); // batch=1, seqLen=4
    const flashOut = flash.forward(input);
    const stdOut = std.forward(input);
    
    matricesClose(flashOut, stdOut, 1e-10);
  });
  
  it('matches with multiple batch items', () => {
    const d = 8, heads = 2;
    const flash = new MultiHeadFlashAttention(d, heads, { blockSize: 2 });
    const std = new MultiHeadAttention(d, heads);
    
    std.Wq = flash.Wq; std.Wk = flash.Wk; std.Wv = flash.Wv; std.Wo = flash.Wo;
    std.bq = flash.bq; std.bk = flash.bk; std.bv = flash.bv; std.bo = flash.bo;
    
    const input = Matrix.random(3, 4 * d);
    matricesClose(flash.forward(input), std.forward(input), 1e-10);
  });
  
  it('backward produces gradients', () => {
    const attn = new MultiHeadFlashAttention(8, 2, { blockSize: 4 });
    const input = Matrix.random(1, 3 * 8);
    const output = attn.forward(input);
    const dOutput = Matrix.random(output.rows, output.cols);
    const dInput = attn.backward(dOutput);
    
    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 3 * 8);
    assert.ok(attn.dWeights);
  });
  
  it('updateWeights changes parameters', () => {
    const attn = new MultiHeadFlashAttention(8, 2);
    const before = attn.Wq.get(0, 0);
    
    const input = Matrix.random(1, 2 * 8);
    attn.forward(input);
    attn.backward(Matrix.random(1, 2 * 8));
    attn.updateWeights(0.01);
    
    assert.notEqual(attn.Wq.get(0, 0), before);
  });
  
  it('paramCount matches MultiHeadAttention', () => {
    const flash = new MultiHeadFlashAttention(16, 4);
    const std = new MultiHeadAttention(16, 4);
    assert.equal(flash.paramCount(), std.paramCount());
  });
});

describe('TransformerEncoderBlock with flash attention', () => {
  it('constructs with attention=flash', () => {
    const block = new TransformerEncoderBlock(8, 2, null, { attention: 'flash' });
    assert.ok(block.attention instanceof MultiHeadFlashAttention);
  });
  
  it('constructs with attention=standard (default)', () => {
    const block = new TransformerEncoderBlock(8, 2);
    assert.ok(block.attention instanceof MultiHeadAttention);
  });
  
  it('flash block produces correct output shape', () => {
    const block = new TransformerEncoderBlock(8, 2, null, { attention: 'flash', blockSize: 4 });
    const input = Matrix.random(2, 4 * 8);
    const output = block.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4 * 8);
  });
  
  it('flash and standard blocks produce same output with shared weights', () => {
    const d = 8, heads = 2;
    const flashBlock = new TransformerEncoderBlock(d, heads, null, { attention: 'flash', blockSize: 2 });
    const stdBlock = new TransformerEncoderBlock(d, heads);
    
    // Share all weights
    stdBlock.attention.Wq = flashBlock.attention.Wq;
    stdBlock.attention.Wk = flashBlock.attention.Wk;
    stdBlock.attention.Wv = flashBlock.attention.Wv;
    stdBlock.attention.Wo = flashBlock.attention.Wo;
    stdBlock.attention.bq = flashBlock.attention.bq;
    stdBlock.attention.bk = flashBlock.attention.bk;
    stdBlock.attention.bv = flashBlock.attention.bv;
    stdBlock.attention.bo = flashBlock.attention.bo;
    // Share norm and FF weights
    stdBlock.norm1 = flashBlock.norm1;
    stdBlock.norm2 = flashBlock.norm2;
    stdBlock.ff1 = flashBlock.ff1;
    stdBlock.ff2 = flashBlock.ff2;
    
    const input = Matrix.random(1, 3 * d);
    const flashOut = flashBlock.forward(input);
    const stdOut = stdBlock.forward(input);
    
    matricesClose(flashOut, stdOut, 1e-8);
  });
  
  it('flash block backward works', () => {
    const block = new TransformerEncoderBlock(8, 2, null, { attention: 'flash' });
    const input = Matrix.random(1, 3 * 8);
    const output = block.forward(input);
    const dOutput = Matrix.random(output.rows, output.cols);
    const dInput = block.backward(dOutput);
    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 3 * 8);
  });
});
