// transformer.test.js — Tests for TransformerEncoderBlock

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TransformerEncoderBlock } from '../src/transformer.js';
import { Matrix } from '../src/matrix.js';
import { getLoss } from '../src/loss.js';
import { Network } from '../src/network.js';

describe('TransformerEncoderBlock', () => {
  it('should preserve input dimensions', () => {
    const t = new TransformerEncoderBlock(8, 2, 16);
    const input = Matrix.random(1, 24); // 3 tokens × 8 dims
    const output = t.forward(input);
    assert.equal(output.rows, input.rows);
    assert.equal(output.cols, input.cols);
  });

  it('should handle single token', () => {
    const t = new TransformerEncoderBlock(4, 1, 8);
    const input = Matrix.random(1, 4); // 1 token × 4 dims
    const output = t.forward(input);
    assert.equal(output.cols, 4);
  });

  it('should handle multiple tokens', () => {
    const t = new TransformerEncoderBlock(4, 2, 8);
    const input = Matrix.random(1, 20); // 5 tokens × 4 dims
    const output = t.forward(input);
    assert.equal(output.cols, 20);
  });

  it('should compute backward pass', () => {
    const t = new TransformerEncoderBlock(4, 1, 8);
    const input = Matrix.random(1, 12);
    const target = Matrix.random(1, 12);
    const loss = getLoss('mse');

    const output = t.forward(input);
    const dOutput = loss.gradient(output, target);
    const dInput = t.backward(dOutput);

    assert.equal(dInput.rows, 1);
    assert.equal(dInput.cols, 12);
    
    // Gradients should be finite
    for (let i = 0; i < dInput.data.length; i++) {
      assert.ok(isFinite(dInput.data[i]), `dInput[${i}] not finite: ${dInput.data[i]}`);
    }
  });

  it('should produce different outputs for different inputs', () => {
    const t = new TransformerEncoderBlock(4, 1, 8);
    const input1 = Matrix.fromArray([[1, 0, 0, 0, 0, 1, 0, 0]]);
    const input2 = Matrix.fromArray([[0, 0, 1, 0, 0, 0, 0, 1]]);

    const output1 = t.forward(input1);
    const output2 = t.forward(input2);

    let diff = 0;
    for (let i = 0; i < output1.data.length; i++) {
      diff += Math.abs(output1.data[i] - output2.data[i]);
    }
    assert.ok(diff > 0.01, 'Different inputs should produce different outputs');
  });

  it('should work in a Network', () => {
    const net = new Network();
    net.add(new TransformerEncoderBlock(4, 1, 8));
    net.dense(8, 2, 'softmax').loss('cross_entropy');
    
    const input = Matrix.random(1, 8);  // 2 tokens × 4 dims
    const target = Matrix.fromArray([[1, 0]]);
    
    const loss = net.trainBatch(input, target, 0.01);
    assert.ok(isFinite(loss), `Loss should be finite: ${loss}`);
  });
});

  it('should decrease loss with training', () => {
    const net = new Network();
    net.add(new TransformerEncoderBlock(4, 1, 8));
    net.dense(8, 2, 'softmax').loss('cross_entropy');
    
    const input = Matrix.fromArray([[0.5, 0.3, 0.8, 0.1, 0.2, 0.7, 0.4, 0.6]]);
    const target = Matrix.fromArray([[1, 0]]);
    
    const loss1 = net.trainBatch(input, target, 0.01);
    let lastLoss = loss1;
    for (let i = 0; i < 50; i++) {
      lastLoss = net.trainBatch(input, target, 0.01);
    }
    
    assert.ok(lastLoss < loss1, `Loss should decrease: ${loss1.toFixed(4)} → ${lastLoss.toFixed(4)}`);
  });
