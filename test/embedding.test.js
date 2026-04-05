import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, Embedding, Network, Dense, PositionalEncoding, TransformerEncoderBlock } from '../src/index.js';

describe('Embedding', () => {
  it('forward produces correct shape', () => {
    const emb = new Embedding(100, 8); // vocab=100, dim=8
    // batch=2, seq_len=5, token IDs
    const input = Matrix.fromArray([[3, 7, 12, 1, 0], [50, 99, 0, 42, 15]]);
    const output = emb.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 40); // 5 * 8
  });

  it('same token produces same embedding', () => {
    const emb = new Embedding(50, 4);
    const input = Matrix.fromArray([[5, 5, 5]]);
    const output = emb.forward(input);
    // All three positions should have identical embeddings
    for (let d = 0; d < 4; d++) {
      assert.equal(output.get(0, d), output.get(0, 4 + d));
      assert.equal(output.get(0, d), output.get(0, 8 + d));
    }
  });

  it('different tokens produce different embeddings', () => {
    const emb = new Embedding(50, 4);
    const input = Matrix.fromArray([[0, 1]]);
    const output = emb.forward(input);
    let diff = 0;
    for (let d = 0; d < 4; d++) diff += Math.abs(output.get(0, d) - output.get(0, 4 + d));
    assert.ok(diff > 0.01, 'Different tokens should have different embeddings');
  });

  it('param count is vocab * dim', () => {
    const emb = new Embedding(100, 16);
    assert.equal(emb.paramCount(), 1600);
  });

  it('backward updates only used token rows', () => {
    const emb = new Embedding(10, 4);
    const input = Matrix.fromArray([[2, 5]]);
    emb.forward(input);
    const dOutput = new Matrix(1, 8);
    for (let j = 0; j < 8; j++) dOutput.set(0, j, 1);
    emb.backward(dOutput);
    
    // Token 2 and 5 should have gradients
    let grad2 = 0, grad5 = 0, grad0 = 0;
    for (let d = 0; d < 4; d++) {
      grad2 += Math.abs(emb.dWeights.get(2, d));
      grad5 += Math.abs(emb.dWeights.get(5, d));
      grad0 += Math.abs(emb.dWeights.get(0, d));
    }
    assert.ok(grad2 > 0, 'Token 2 should have gradients');
    assert.ok(grad5 > 0, 'Token 5 should have gradients');
    assert.equal(grad0, 0, 'Token 0 (unused) should have zero gradient');
  });

  it('clamps out-of-range token IDs', () => {
    const emb = new Embedding(10, 4);
    const input = Matrix.fromArray([[999, -5]]); // Out of range
    const output = emb.forward(input);
    // Should not crash, values clamped to [0, 9]
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 8);
  });

  it('works with Network', () => {
    const net = new Network();
    net.add(new Embedding(20, 4));
    net.add(new Dense(12, 2, 'softmax')); // seq_len=3, dim=4 → flatten to 12
    net.loss('crossEntropy');
    
    const input = Matrix.fromArray([[1, 5, 10], [3, 7, 0]]);
    const output = net.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 2);
  });

  it('full pipeline: Embedding + PE + Transformer', () => {
    const net = new Network();
    net.add(new Embedding(50, 4));
    net.add(new PositionalEncoding(4));
    net.add(new TransformerEncoderBlock(4, 2, 8));
    net.add(new Dense(8, 3, 'softmax'));
    net.loss('crossEntropy');
    
    const input = Matrix.fromArray([[5, 10]]); // 2 tokens
    const output = net.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 3);
  });
});
