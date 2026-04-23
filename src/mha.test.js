// mha.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { MultiHeadAttention } from './mha.js';
import { causalMask } from './attention-masks.js';
import { Matrix } from './matrix.js';

describe('Multi-Head Attention', () => {
  test('output shape matches input', () => {
    const mha = new MultiHeadAttention(8, 2);
    const x = Matrix.random(5, 8);
    const out = mha.forward(x);
    assert.equal(out.rows, 5);
    assert.equal(out.cols, 8);
  });

  test('works with causal mask', () => {
    const mha = new MultiHeadAttention(8, 2);
    const x = Matrix.random(4, 8);
    const mask = causalMask(4);
    const out = mha.forward(x, mask);
    assert.equal(out.rows, 4);
    assert.ok(isFinite(out.get(0, 0)));
  });

  test('attention weights are saved', () => {
    const mha = new MultiHeadAttention(8, 4);
    const x = Matrix.random(3, 8);
    mha.forward(x);
    const weights = mha.getAttentionWeights();
    assert.equal(weights.length, 4); // 4 heads
    assert.equal(weights[0].rows, 3);
    assert.equal(weights[0].cols, 3);
  });

  test('attention weights sum to 1 per row', () => {
    const mha = new MultiHeadAttention(8, 2);
    const x = Matrix.random(4, 8);
    mha.forward(x);
    const weights = mha.getAttentionWeights()[0];
    
    for (let i = 0; i < 4; i++) {
      let sum = 0;
      for (let j = 0; j < 4; j++) sum += weights.get(i, j);
      assert.ok(Math.abs(sum - 1) < 1e-6, `Row ${i} should sum to 1, got ${sum}`);
    }
  });

  test('different inputs produce different outputs', () => {
    const mha = new MultiHeadAttention(8, 2);
    const x1 = Matrix.random(3, 8);
    const x2 = Matrix.random(3, 8);
    const out1 = mha.forward(x1);
    const out2 = mha.forward(x2);
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01);
  });

  test('single token input works', () => {
    const mha = new MultiHeadAttention(4, 2);
    const x = Matrix.random(1, 4);
    const out = mha.forward(x);
    assert.equal(out.rows, 1);
    assert.equal(out.cols, 4);
  });
});
