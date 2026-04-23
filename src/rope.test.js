// rope.test.js — RoPE (Rotary Position Embedding) tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { precomputeFreqs, applyRoPE, applyRoPEBackward, ropeAttention } from './rope.js';
import { Matrix } from './matrix.js';

describe('RoPE', () => {
  test('precomputeFreqs returns correct shape', () => {
    const { cos, sin } = precomputeFreqs(8, 16);
    assert.equal(cos.rows, 16);
    assert.equal(cos.cols, 4); // dim/2
    assert.equal(sin.rows, 16);
    assert.equal(sin.cols, 4);
  });

  test('precomputeFreqs: position 0 has cos=1, sin=0', () => {
    const { cos, sin } = precomputeFreqs(8, 16);
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(cos.get(0, i) - 1) < 0.001, `cos(0,${i}) should be 1`);
      assert.ok(Math.abs(sin.get(0, i)) < 0.001, `sin(0,${i}) should be 0`);
    }
  });

  test('applyRoPE preserves norm (rotation is orthogonal)', () => {
    const { cos, sin } = precomputeFreqs(8, 16);
    const x = new Matrix(4, 8);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 8; j++) x.set(i, j, Math.random());
    
    const rotated = applyRoPE(x, cos, sin);
    
    // Check that norm is preserved for each row
    for (let i = 0; i < 4; i++) {
      let normOrig = 0, normRot = 0;
      for (let j = 0; j < 8; j++) {
        normOrig += x.get(i, j) ** 2;
        normRot += rotated.get(i, j) ** 2;
      }
      assert.ok(Math.abs(normOrig - normRot) < 0.001, 
        `Norm should be preserved: orig=${normOrig.toFixed(4)} rot=${normRot.toFixed(4)}`);
    }
  });

  test('applyRoPE at position 0 is identity', () => {
    const { cos, sin } = precomputeFreqs(8, 16);
    const x = new Matrix(1, 8);
    for (let j = 0; j < 8; j++) x.set(0, j, j + 1);
    
    const rotated = applyRoPE(x, cos, sin);
    for (let j = 0; j < 8; j++) {
      assert.ok(Math.abs(rotated.get(0, j) - x.get(0, j)) < 0.001,
        `Position 0 should be identity: got ${rotated.get(0, j)} expected ${x.get(0, j)}`);
    }
  });

  test('backward is inverse of forward (rotation)', () => {
    const { cos, sin } = precomputeFreqs(8, 16);
    const x = new Matrix(3, 8);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 8; j++) x.set(i, j, Math.random() * 2 - 1);
    
    const rotated = applyRoPE(x, cos, sin);
    const recovered = applyRoPEBackward(rotated, cos, sin);
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 8; j++) {
        assert.ok(Math.abs(recovered.get(i, j) - x.get(i, j)) < 0.001,
          `Backward should recover original: got ${recovered.get(i, j)} expected ${x.get(i, j)}`);
      }
    }
  });

  test('relative position: dot product depends on position difference', () => {
    const { cos, sin } = precomputeFreqs(8, 32);
    
    // Create two vectors
    const q = new Matrix(1, 8);
    const k = new Matrix(1, 8);
    for (let j = 0; j < 8; j++) {
      q.set(0, j, 1.0);
      k.set(0, j, 1.0);
    }
    
    // Rotate q to position 5, k to position 3 (diff = 2)
    const q5 = applyRoPE(q, cos, sin, 5);
    const k3 = applyRoPE(k, cos, sin, 3);
    let dot53 = 0;
    for (let j = 0; j < 8; j++) dot53 += q5.get(0, j) * k3.get(0, j);
    
    // Rotate q to position 10, k to position 8 (diff = 2)
    const q10 = applyRoPE(q, cos, sin, 10);
    const k8 = applyRoPE(k, cos, sin, 8);
    let dot108 = 0;
    for (let j = 0; j < 8; j++) dot108 += q10.get(0, j) * k8.get(0, j);
    
    // Same relative position → same dot product
    assert.ok(Math.abs(dot53 - dot108) < 0.001,
      `Same relative position should give same dot: ${dot53.toFixed(4)} vs ${dot108.toFixed(4)}`);
  });

  test('ropeAttention returns rotated Q and K', () => {
    const freqs = precomputeFreqs(8, 16);
    const Q = Matrix.random(4, 8);
    const K = Matrix.random(4, 8);
    
    const { Q_rot, K_rot } = ropeAttention(Q, K, freqs);
    assert.equal(Q_rot.rows, 4);
    assert.equal(Q_rot.cols, 8);
    assert.equal(K_rot.rows, 4);
    assert.equal(K_rot.cols, 8);
  });

  test('offset shifts position correctly', () => {
    const { cos, sin } = precomputeFreqs(8, 32);
    const x = new Matrix(1, 8);
    for (let j = 0; j < 8; j++) x.set(0, j, 1.0);
    
    // Position 5 directly
    const directPos5 = new Matrix(6, 8);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 8; j++) directPos5.set(i, j, 1.0);
    const rotDirect = applyRoPE(directPos5, cos, sin);
    
    // Position 5 via offset
    const rotOffset = applyRoPE(x, cos, sin, 5);
    
    // Should match row 5 of the direct computation
    for (let j = 0; j < 8; j++) {
      assert.ok(Math.abs(rotDirect.get(5, j) - rotOffset.get(0, j)) < 0.001);
    }
  });

  test('even dimension required', () => {
    assert.throws(() => precomputeFreqs(7, 16), /even/);
  });
});
