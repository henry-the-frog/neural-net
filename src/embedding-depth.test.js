// embedding-depth.test.js — Embedding layer depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Embedding } from './embedding.js';
import { Matrix } from './matrix.js';

describe('Embedding Output Shape', () => {
  it('single token', () => {
    const emb = new Embedding(100, 8); // vocab=100, dim=8
    const input = new Matrix(1, 1, new Float64Array([5])); // 1 token
    const output = emb.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 8); // 1 * embedDim
  });

  it('sequence of 3 tokens', () => {
    const emb = new Embedding(100, 8);
    const input = new Matrix(1, 3, new Float64Array([5, 10, 15]));
    const output = emb.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 24); // 3 * 8
  });

  it('batch of 4 sequences', () => {
    const emb = new Embedding(100, 16);
    const input = new Matrix(4, 5, new Float64Array(20).fill(1)); // 4 batch, 5 tokens each
    const output = emb.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 80); // 5 * 16
  });
});

describe('Embedding Lookup', () => {
  it('same token gives same embedding', () => {
    const emb = new Embedding(50, 4);
    const input = new Matrix(1, 3, new Float64Array([7, 7, 7])); // Same token repeated
    const output = emb.forward(input);
    
    // All 3 embeddings should be identical
    for (let d = 0; d < 4; d++) {
      assert.equal(output.get(0, d), output.get(0, 4 + d));
      assert.equal(output.get(0, d), output.get(0, 8 + d));
    }
  });

  it('different tokens give different embeddings', () => {
    const emb = new Embedding(50, 4);
    const input = new Matrix(1, 2, new Float64Array([3, 7]));
    const output = emb.forward(input);
    
    let different = false;
    for (let d = 0; d < 4; d++) {
      if (Math.abs(output.get(0, d) - output.get(0, 4 + d)) > 1e-6) {
        different = true;
        break;
      }
    }
    assert.ok(different, 'Different tokens should have different embeddings');
  });

  it('out-of-range token IDs are clamped', () => {
    const emb = new Embedding(10, 4);
    const input = new Matrix(1, 1, new Float64Array([999])); // Beyond vocab
    // Should not throw, should clamp to valid range
    const output = emb.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 4);
  });
});

describe('Embedding Backward', () => {
  it('backward returns correct gradient shape', () => {
    const emb = new Embedding(50, 8);
    const input = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    emb.forward(input);
    const dOutput = Matrix.random(2, 24); // 3 * 8
    const dInput = emb.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 3);
  });

  it('weight gradient has correct shape', () => {
    const emb = new Embedding(20, 4);
    const input = new Matrix(1, 2, new Float64Array([5, 10]));
    emb.forward(input);
    emb.backward(Matrix.random(1, 8));
    assert.equal(emb.dWeights.rows, 20);
    assert.equal(emb.dWeights.cols, 4);
  });
});
