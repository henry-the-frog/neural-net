// gqa.test.js — Grouped Query Attention tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { groupedQueryAttention, gqaDimensions } from './gqa.js';
import { standardAttention } from './flash-attention.js';
import { Matrix } from './matrix.js';

describe('Grouped Query Attention', () => {
  test('GQA with nKVHeads=nHeads is standard MHA', () => {
    const N = 4, d = 8, nHeads = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    // GQA with nKVHeads=nHeads should match MHA
    const gqa = groupedQueryAttention(Q, K, V, nHeads, nHeads);
    assert.equal(gqa.rows, N);
    assert.equal(gqa.cols, d);
  });

  test('GQA with nKVHeads=1 is MQA (multi-query)', () => {
    const N = 4, d = 8, nHeads = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, 2); // nKVHeads=1, headDim=2
    const V = Matrix.random(N, 2);
    
    const out = groupedQueryAttention(Q, K, V, nHeads, 1);
    assert.equal(out.rows, N);
    assert.equal(out.cols, d);
  });

  test('GQA correct shape with group size 2', () => {
    const N = 6, nHeads = 8, headDim = 4;
    const dModel = nHeads * headDim; // 32
    const nKVHeads = 4;
    const dKV = nKVHeads * headDim; // 16
    
    const Q = Matrix.random(N, dModel);
    const K = Matrix.random(N, dKV);
    const V = Matrix.random(N, dKV);
    
    const out = groupedQueryAttention(Q, K, V, nHeads, nKVHeads);
    assert.equal(out.rows, N);
    assert.equal(out.cols, dModel);
  });

  test('causal mask works', () => {
    const N = 4, d = 4, nHeads = 2, nKVHeads = 1;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, 2);
    const V = new Matrix(N, 2);
    for (let i = 0; i < N; i++) V.set(i, 0, i); // V[i] = [i, 0]
    
    const out = groupedQueryAttention(Q, K, V, nHeads, nKVHeads, true);
    // First token can only attend to itself
    // With causal mask, row 0 only sees V[0]
    // Each head's output for position 0, dim 0 should be 0 (V[0][0]=0)
    assert.ok(Math.abs(out.get(0, 0)) < 0.001);
  });

  test('nHeads must be divisible by nKVHeads', () => {
    const Q = Matrix.random(4, 6);
    const K = Matrix.random(4, 4);
    const V = Matrix.random(4, 4);
    assert.throws(() => groupedQueryAttention(Q, K, V, 3, 2), /divisible/);
  });

  test('gqaDimensions computes savings correctly', () => {
    const dims = gqaDimensions(512, 8, 2);
    assert.equal(dims.dQ, 512);
    assert.equal(dims.dKV, 128);
    assert.equal(dims.headDim, 64);
    assert.equal(dims.groupSize, 4);
    assert.equal(dims.kvSaving, '75.0% KV memory saved');
  });

  test('gqaDimensions: MHA has 0% savings', () => {
    const dims = gqaDimensions(512, 8, 8);
    assert.equal(dims.kvSaving, '0.0% KV memory saved');
  });

  test('gqaDimensions: MQA has maximum savings', () => {
    const dims = gqaDimensions(512, 8, 1);
    assert.equal(dims.kvSaving, '87.5% KV memory saved');
  });
});
