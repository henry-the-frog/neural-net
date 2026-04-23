// cross-attention.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { crossAttention, attentionPooling } from './cross-attention.js';
import { Matrix } from './matrix.js';

describe('Cross-Attention', () => {
  test('output shape matches query sequence', () => {
    const Q = Matrix.random(3, 8); // 3 decoder tokens
    const K = Matrix.random(5, 8); // 5 encoder tokens
    const V = Matrix.random(5, 8);
    const out = crossAttention(Q, K, V, 2);
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 8);
  });

  test('single head single query attends to all keys', () => {
    const Q = new Matrix(1, 4);
    Q.set(0, 0, 1); Q.set(0, 1, 0); Q.set(0, 2, 0); Q.set(0, 3, 0);
    
    const K = Matrix.random(3, 4);
    const V = Matrix.random(3, 4);
    const out = crossAttention(Q, K, V, 1);
    assert.equal(out.rows, 1);
    assert.equal(out.cols, 4);
    // Output should be a weighted combination of V rows
    let nonZero = 0;
    for (let j = 0; j < 4; j++) if (Math.abs(out.get(0, j)) > 1e-6) nonZero++;
    assert.ok(nonZero > 0);
  });

  test('different Q/K sequences produce different outputs', () => {
    const Q1 = Matrix.random(2, 4);
    const Q2 = Matrix.random(2, 4);
    const K = Matrix.random(3, 4);
    const V = Matrix.random(3, 4);
    
    const out1 = crossAttention(Q1, K, V, 1);
    const out2 = crossAttention(Q2, K, V, 1);
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01);
  });

  test('attention pooling produces fixed-size output', () => {
    const tokens = Matrix.random(10, 8);
    const query = new Float64Array(8).fill(0.1);
    const pooled = attentionPooling(tokens, query);
    assert.equal(pooled.length, 8);
  });

  test('attention pooling is weighted combination', () => {
    const tokens = new Matrix(2, 2);
    tokens.set(0, 0, 1); tokens.set(0, 1, 0);
    tokens.set(1, 0, 0); tokens.set(1, 1, 1);
    
    const query = new Float64Array([1, 0]); // Should attend more to token 0
    const pooled = attentionPooling(tokens, query);
    
    // pooled[0] should be closer to 1 than pooled[1]
    assert.ok(pooled[0] > pooled[1], `Should attend to token 0: ${pooled[0]} vs ${pooled[1]}`);
  });

  test('multi-head cross attention', () => {
    const Q = Matrix.random(4, 8);
    const K = Matrix.random(6, 8);
    const V = Matrix.random(6, 8);
    const out = crossAttention(Q, K, V, 4); // 4 heads
    assert.equal(out.rows, 4);
    assert.equal(out.cols, 8);
  });
});
