// flash-attention.test.js — Flash Attention tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { standardAttention, flashAttention, multiHeadFlashAttention } from './flash-attention.js';
import { Matrix } from './matrix.js';

function matrixClose(a, b, tol = 0.001) {
  if (a.rows !== b.rows || a.cols !== b.cols) return false;
  for (let i = 0; i < a.data.length; i++) {
    if (Math.abs(a.data[i] - b.data[i]) > tol) return false;
  }
  return true;
}

describe('Flash Attention', () => {
  test('standardAttention produces correct shape', () => {
    const Q = Matrix.random(4, 8);
    const K = Matrix.random(4, 8);
    const V = Matrix.random(4, 8);
    const out = standardAttention(Q, K, V);
    assert.equal(out.rows, 4);
    assert.equal(out.cols, 8);
  });

  test('flashAttention matches standardAttention (non-causal)', () => {
    const N = 8, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    const standard = standardAttention(Q, K, V);
    const flash = flashAttention(Q, K, V, 4);
    
    assert.ok(matrixClose(standard, flash), 
      'Flash attention should match standard attention');
  });

  test('flashAttention matches standardAttention (causal)', () => {
    const N = 8, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    const standard = standardAttention(Q, K, V, true);
    const flash = flashAttention(Q, K, V, 3, true);
    
    assert.ok(matrixClose(standard, flash, 0.01), 
      'Flash causal attention should match standard causal');
  });

  test('flashAttention with blockSize=1 matches standard', () => {
    const N = 6, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    const standard = standardAttention(Q, K, V);
    const flash = flashAttention(Q, K, V, 1); // Most granular tiling
    
    assert.ok(matrixClose(standard, flash), 
      'Block size 1 should match standard exactly');
  });

  test('flashAttention with blockSize=N matches standard', () => {
    const N = 8, d = 4;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = Matrix.random(N, d);
    
    const standard = standardAttention(Q, K, V);
    const flash = flashAttention(Q, K, V, N); // Single block = standard attention
    
    assert.ok(matrixClose(standard, flash), 
      'Single block should match standard exactly');
  });

  test('multiHeadFlashAttention correct shape', () => {
    const Q = Matrix.random(6, 8);
    const K = Matrix.random(6, 8);
    const V = Matrix.random(6, 8);
    
    const out = multiHeadFlashAttention(Q, K, V, 2);
    assert.equal(out.rows, 6);
    assert.equal(out.cols, 8);
  });

  test('multiHeadFlashAttention requires divisible dims', () => {
    const Q = Matrix.random(4, 7);
    const K = Matrix.random(4, 7);
    const V = Matrix.random(4, 7);
    assert.throws(() => multiHeadFlashAttention(Q, K, V, 3), /divisible/);
  });

  test('causal mask: first row only attends to itself', () => {
    const N = 4, d = 2;
    const Q = Matrix.random(N, d);
    const K = Matrix.random(N, d);
    const V = new Matrix(N, d);
    // Set V to identity-like: V[i] = [i, 0]
    for (let i = 0; i < N; i++) V.set(i, 0, i);
    
    const out = flashAttention(Q, K, V, 2, true);
    // First row should only attend to V[0] = [0, 0]
    assert.ok(Math.abs(out.get(0, 0) - 0) < 0.001, 
      `First row col 0 should be 0, got ${out.get(0, 0)}`);
  });

  test('identical Q=K gives identity-like attention', () => {
    const N = 4, d = 8;
    const Q = Matrix.random(N, d);
    // K = Q means each query attends most to its own position
    const V = new Matrix(N, d);
    for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) V.set(i, j, i === j % N ? 1 : 0);
    
    const standard = standardAttention(Q, Q, V);
    const flash = flashAttention(Q, Q, V, 2);
    assert.ok(matrixClose(standard, flash), 'Q=K: flash should match standard');
  });
});
