// grouped-query-attention.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { GroupedQueryAttention } from './grouped-query-attention.js';
import { MultiHeadFlashAttention } from './multi-head-flash-attention.js';
import { Matrix } from './matrix.js';

describe('GroupedQueryAttention', () => {
  it('produces correct output shape', () => {
    const gqa = new GroupedQueryAttention(8, 4, 2); // 4 Q heads, 2 KV heads
    const input = Matrix.random(2, 3 * 8); // batch=2, seqLen=3
    const output = gqa.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 3 * 8);
  });
  
  it('numKVHeads == numHeads is equivalent to MHA', () => {
    const d = 8, heads = 2;
    const gqa = new GroupedQueryAttention(d, heads, heads, { blockSize: 2 });
    const mha = new MultiHeadFlashAttention(d, heads, { blockSize: 2 });
    
    // Share weights
    mha.Wq = gqa.Wq; mha.Wk = gqa.Wk; mha.Wv = gqa.Wv; mha.Wo = gqa.Wo;
    mha.bq = gqa.bq; mha.bk = gqa.bk; mha.bv = gqa.bv; mha.bo = gqa.bo;
    
    const input = Matrix.random(1, 3 * d);
    const gqaOut = gqa.forward(input);
    const mhaOut = mha.forward(input);
    
    // Should be numerically identical
    for (let i = 0; i < gqaOut.cols; i++) {
      const diff = Math.abs(gqaOut.get(0, i) - mhaOut.get(0, i));
      assert.ok(diff < 1e-10, `col ${i}: ${gqaOut.get(0, i)} vs ${mhaOut.get(0, i)}`);
    }
  });
  
  it('numKVHeads == 1 is Multi-Query Attention', () => {
    const gqa = new GroupedQueryAttention(8, 4, 1); // 4 Q heads, 1 KV head
    const input = Matrix.random(1, 3 * 8);
    const output = gqa.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 3 * 8);
    // All values should be finite
    for (let i = 0; i < output.cols; i++) {
      assert.ok(isFinite(output.get(0, i)));
    }
  });
  
  it('reduces KV cache size', () => {
    const gqa = new GroupedQueryAttention(64, 8, 2); // 8 Q heads, 2 KV heads
    const seqLen = 1024;
    const kvSize = gqa.kvCacheSize(seqLen);
    const mhaSize = gqa.mhaKVCacheSize(seqLen);
    assert.equal(mhaSize / kvSize, 4); // 4x reduction (8/2 = 4)
  });
  
  it('reduces parameter count', () => {
    const d = 64;
    const gqa = new GroupedQueryAttention(d, 8, 2);
    const mha = new MultiHeadFlashAttention(d, 8);
    assert.ok(gqa.paramCount() < mha.paramCount(), 'GQA should have fewer params');
    // GQA saves on K and V projections: 64*16 + 16 + 64*16 + 16 = 2080 instead of 64*64 + 64 + 64*64 + 64 = 8320
    // That's a 6240 param reduction
    const savings = mha.paramCount() - gqa.paramCount();
    assert.ok(savings > 0);
  });
  
  it('works with causal mask', () => {
    const gqa = new GroupedQueryAttention(8, 4, 2, { causal: true });
    const input = Matrix.random(1, 4 * 8);
    const output = gqa.forward(input);
    for (let i = 0; i < output.cols; i++) {
      assert.ok(isFinite(output.get(0, i)));
    }
  });
  
  it('batch processing works correctly', () => {
    const gqa = new GroupedQueryAttention(8, 4, 2);
    const input = Matrix.random(4, 3 * 8); // batch=4
    const output = gqa.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 3 * 8);
  });
  
  it('throws on invalid numHeads/numKVHeads', () => {
    assert.throws(() => new GroupedQueryAttention(8, 4, 3), /must be divisible/);
    assert.throws(() => new GroupedQueryAttention(7, 4, 2), /must be divisible/);
  });
});
