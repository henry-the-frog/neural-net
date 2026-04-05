import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, PositionalEncoding, LayerNorm, TransformerEncoderBlock, Network, Dense } from '../src/index.js';

describe('PositionalEncoding', () => {
  it('adds position information to input', () => {
    const pe = new PositionalEncoding(4, 100);
    const input = Matrix.zeros(1, 12); // seq_len=3, d=4, all zeros
    const output = pe.forward(input);
    // Output should be non-zero (positional encoding added)
    let sum = 0;
    for (let j = 0; j < 12; j++) sum += Math.abs(output.get(0, j));
    assert.ok(sum > 0, 'PE should add non-zero values');
  });

  it('different positions get different encodings', () => {
    const pe = new PositionalEncoding(4, 100);
    // Position 0 and position 1 should differ
    assert.notDeepStrictEqual(
      [pe.pe.get(0, 0), pe.pe.get(0, 1)],
      [pe.pe.get(1, 0), pe.pe.get(1, 1)]
    );
  });

  it('preserves input dimensions', () => {
    const pe = new PositionalEncoding(8);
    const input = Matrix.random(3, 24);
    const output = pe.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 24);
  });

  it('backward passes gradient through', () => {
    const pe = new PositionalEncoding(4);
    const input = Matrix.random(1, 8);
    pe.forward(input);
    const dOutput = Matrix.random(1, 8);
    const dInput = pe.backward(dOutput);
    // PE backward is identity (additive constant)
    for (let j = 0; j < 8; j++) {
      assert.equal(dInput.get(0, j), dOutput.get(0, j));
    }
  });

  it('no learnable parameters', () => {
    const pe = new PositionalEncoding(8);
    assert.equal(pe.paramCount(), 0);
  });
});

describe('LayerNorm', () => {
  it('normalizes to zero mean', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.fromArray([[10, 20, 30, 40]]);
    const output = ln.forward(input);
    let mean = 0;
    for (let j = 0; j < 4; j++) mean += output.get(0, j);
    mean /= 4;
    assert.ok(Math.abs(mean) < 0.01, `Mean should be ~0, got ${mean}`);
  });

  it('normalizes to unit variance', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.fromArray([[1, 5, 3, 7]]);
    const output = ln.forward(input);
    let mean = 0;
    for (let j = 0; j < 4; j++) mean += output.get(0, j);
    mean /= 4;
    let variance = 0;
    for (let j = 0; j < 4; j++) {
      const d = output.get(0, j) - mean;
      variance += d * d;
    }
    variance /= 4;
    assert.ok(Math.abs(variance - 1) < 0.1, `Variance should be ~1, got ${variance}`);
  });

  it('preserves shape for sequences', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.random(2, 12); // batch=2, seq=3, d=4
    const output = ln.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 12);
  });

  it('backward produces gradients', () => {
    const ln = new LayerNorm(4);
    const input = Matrix.random(2, 8);
    ln.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = ln.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 8);
    let gradSum = 0;
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 8; j++)
        gradSum += Math.abs(dInput.get(i, j));
    assert.ok(gradSum > 0, 'Gradients should be non-zero');
  });

  it('param count is 2 * dModel', () => {
    const ln = new LayerNorm(8);
    assert.equal(ln.paramCount(), 16);
  });
});

describe('TransformerEncoderBlock', () => {
  it('forward produces correct shape', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const input = Matrix.random(2, 24); // batch=2, seq=3, d=8
    const output = block.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('param count includes all components', () => {
    const block = new TransformerEncoderBlock(8, 2);
    const params = block.paramCount();
    // attention: 4*(8*8+8) = 288
    // norm1 + norm2: 2*2*8 = 32
    // ff1: 8*32+32 = 288, ff2: 32*8+8 = 264
    assert.ok(params > 500, `Should have many params, got ${params}`);
  });

  it('backward produces gradient', () => {
    const block = new TransformerEncoderBlock(4, 2, 8);
    const input = Matrix.random(1, 8); // seq=2, d=4
    block.forward(input);
    const dOutput = Matrix.random(1, 8);
    const dInput = block.backward(dOutput);
    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 8);
  });

  it('output differs from input (not identity)', () => {
    const block = new TransformerEncoderBlock(4, 2, 8);
    const input = Matrix.random(1, 8);
    const output = block.forward(input);
    let diff = 0;
    for (let j = 0; j < 8; j++) diff += Math.abs(output.get(0, j) - input.get(0, j));
    assert.ok(diff > 0.01, 'Transformer block should transform input');
  });

  it('single position sequence works', () => {
    const block = new TransformerEncoderBlock(4, 1, 8);
    const input = Matrix.random(1, 4); // seq=1
    const output = block.forward(input);
    assert.equal(output.cols, 4);
  });

  it('stacks multiple blocks', () => {
    const net = new Network();
    net.add(new PositionalEncoding(4));
    net.add(new TransformerEncoderBlock(4, 2, 8));
    net.add(new TransformerEncoderBlock(4, 2, 8));
    net.add(new Dense(8, 2, 'softmax')); // pool/flatten

    const input = Matrix.random(2, 8); // batch=2, seq=2, d=4
    const output = net.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 2);
  });
});
