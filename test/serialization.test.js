// serialization.test.js — Tests for model save/load

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';

describe('Model Serialization', () => {
  it('should save and load a simple Dense network', () => {
    const net = new Network();
    net.dense(3, 4, 'relu').dense(4, 2, 'sigmoid').loss('mse');
    
    const input = Matrix.fromArray([[0.5, -0.3, 0.8]]);
    const original = net.predict(input);
    
    const json = net.save();
    const loaded = Network.load(json);
    const restored = loaded.predict(input);
    
    for (let i = 0; i < original.data.length; i++) {
      assert.ok(Math.abs(original.data[i] - restored.data[i]) < 1e-10,
        `Output ${i} differs: ${original.data[i]} vs ${restored.data[i]}`);
    }
  });

  it('should preserve weights exactly', () => {
    const net = new Network();
    net.dense(2, 3, 'tanh').loss('mse');
    
    const json = net.toJSON();
    const loaded = Network.load(json);
    
    assert.deepEqual(
      Array.from(net.layers[0].weights.data),
      Array.from(loaded.layers[0].weights.data)
    );
    assert.deepEqual(
      Array.from(net.layers[0].biases.data),
      Array.from(loaded.layers[0].biases.data)
    );
  });

  it('should preserve loss function', () => {
    const net = new Network();
    net.dense(2, 1, 'sigmoid').loss('cross_entropy');
    
    const loaded = Network.load(net.save());
    assert.ok(loaded.lossFunction, 'Should have loss function');
  });

  it('should work after training (XOR)', () => {
    const net = new Network();
    net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');
    
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[0]]);
    
    for (let i = 0; i < 2000; i++) net.trainBatch(inputs, targets, 0.5);
    
    const json = net.save();
    const loaded = Network.load(json);
    
    // Loaded model should predict same as trained model
    const pred1 = net.predict(inputs);
    const pred2 = loaded.predict(inputs);
    
    for (let i = 0; i < pred1.data.length; i++) {
      assert.ok(Math.abs(pred1.data[i] - pred2.data[i]) < 1e-10,
        `Prediction ${i} differs after load`);
    }
  });

  it('should produce valid JSON', () => {
    const net = new Network();
    net.dense(2, 4, 'relu').dense(4, 2, 'softmax').loss('cross_entropy');
    
    const json = net.save();
    const parsed = JSON.parse(json);
    
    assert.equal(parsed.version, 1);
    assert.equal(parsed.loss, 'cross_entropy');
    assert.equal(parsed.layers.length, 2);
    assert.equal(parsed.layers[0].type, 'Dense');
    assert.equal(parsed.layers[0].inputSize, 2);
    assert.equal(parsed.layers[0].outputSize, 4);
    assert.equal(parsed.layers[0].activation, 'relu');
  });

  it('should handle deep networks', () => {
    const net = new Network();
    for (let i = 0; i < 5; i++) {
      net.dense(8, 8, 'relu');
    }
    net.dense(8, 2, 'sigmoid').loss('mse');
    
    const input = Matrix.random(1, 8);
    const original = net.predict(input);
    
    const loaded = Network.load(net.save());
    const restored = loaded.predict(input);
    
    for (let i = 0; i < original.data.length; i++) {
      assert.ok(Math.abs(original.data[i] - restored.data[i]) < 1e-10);
    }
  });

  it('should continue training after load', () => {
    const net = new Network();
    net.dense(2, 4, 'relu').dense(4, 1, 'sigmoid').loss('mse');
    
    const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
    const targets = Matrix.fromArray([[0],[1],[1],[0]]);
    
    // Train partially
    for (let i = 0; i < 500; i++) net.trainBatch(inputs, targets, 0.5);
    const loss1 = net.trainBatch(inputs, targets, 0.5);
    
    // Save, load, continue training
    const loaded = Network.load(net.save());
    for (let i = 0; i < 500; i++) loaded.trainBatch(inputs, targets, 0.5);
    const loss2 = loaded.trainBatch(inputs, targets, 0.5);
    
    // Should improve (or at least not crash)
    assert.ok(isFinite(loss2), 'Loss should be finite after continued training');
  });

  it('should handle batch predictions after load', () => {
    const net = new Network();
    net.dense(3, 4, 'relu').dense(4, 2, 'softmax').loss('cross_entropy');
    
    const loaded = Network.load(net.save());
    
    // Batch input
    const input = Matrix.random(5, 3);
    const pred = loaded.predict(input);
    assert.equal(pred.rows, 5);
    assert.equal(pred.cols, 2);
  });
});
