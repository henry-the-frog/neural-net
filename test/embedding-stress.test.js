// embedding-stress.test.js — Stress tests for embedding layers
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Embedding } from '../src/embedding.js';
import { Matrix } from '../src/matrix.js';

describe('Embedding Stress', () => {
  it('different indices produce different embeddings', () => {
    const emb = new Embedding(10, 4);
    const idx1 = new Matrix(1, 1, new Float64Array([0]));
    const idx2 = new Matrix(1, 1, new Float64Array([5]));
    const e1 = emb.forward(idx1);
    const e2 = emb.forward(idx2);
    let different = false;
    for (let i = 0; i < e1.data.length; i++) {
      if (Math.abs(e1.data[i] - e2.data[i]) > 1e-10) different = true;
    }
    assert.ok(different, 'Different indices should give different embeddings');
  });

  it('same index gives same embedding', () => {
    const emb = new Embedding(10, 4);
    const idx = new Matrix(1, 1, new Float64Array([3]));
    const e1 = emb.forward(idx);
    const e2 = emb.forward(idx);
    for (let i = 0; i < e1.data.length; i++) {
      assert.equal(e1.data[i], e2.data[i], `Embedding should be deterministic at dim ${i}`);
    }
  });

  it('batch embedding', () => {
    const emb = new Embedding(10, 4);
    const indices = new Matrix(3, 2, new Float64Array([0, 1, 2, 3, 4, 5]));
    const output = emb.forward(indices);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 8); // 2 indices * 4 dims
  });

  it('embedding weights are trainable', () => {
    const emb = new Embedding(5, 3);
    const idx = new Matrix(1, 1, new Float64Array([2]));
    
    // Forward
    const before = emb.forward(idx);
    const beforeVal = before.data[0];
    
    // Backward with gradient
    const dOutput = new Matrix(1, 3, new Float64Array([1, 0, 0]));
    emb.backward(dOutput);
    
    // Check that dWeights was set
    assert.ok(emb.dWeights, 'dWeights should be set after backward');
    
    // Apply update manually
    const lr = 0.1;
    for (let i = 0; i < emb.weights.data.length; i++) {
      if (emb.dWeights.data[i] !== 0) {
        emb.weights.data[i] -= lr * emb.dWeights.data[i];
      }
    }
    
    const after = emb.forward(idx);
    assert.notEqual(after.data[0], beforeVal, 'Embedding should change after update');
  });

  it('large vocabulary', () => {
    const emb = new Embedding(10000, 64);
    const idx = new Matrix(1, 1, new Float64Array([9999]));
    const output = emb.forward(idx);
    assert.equal(output.cols, 64);
    for (let i = 0; i < 64; i++) {
      assert.ok(isFinite(output.data[i]), `Embedding should be finite at dim ${i}`);
    }
  });
});
