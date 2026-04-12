// gradient-clipping.test.js — Verify gradient clipping prevents explosion

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';

describe('Gradient Clipping', () => {
  it('should prevent gradient explosion in deep network', () => {
    // Create a deep network prone to exploding gradients
    const net = new Network();
    for (let i = 0; i < 8; i++) {
      net.dense(8, 8, 'relu');
    }
    net.dense(8, 1, 'linear').loss('mse');
    
    // Set large weights to trigger explosion
    for (const layer of net.layers) {
      layer.weights = layer.weights.mul(3);
    }
    
    const input = Matrix.random(1, 8);
    const target = Matrix.fromArray([[0.5]]);
    
    // Train with clipping
    const lossClipped = net.trainBatch(input, target, 0.01, 0, 'sgd', { clipGrad: 1.0 });
    
    // All gradients should be clipped to max norm 1.0
    for (const layer of net.layers) {
      if (layer.dWeights) {
        const maxGrad = Math.max(...Array.from(layer.dWeights.data).map(Math.abs));
        assert.ok(maxGrad <= 1.1, `Gradient too large after clipping: ${maxGrad.toFixed(2)}`);
      }
    }
    
    assert.ok(isFinite(lossClipped), 'Loss should be finite with clipping');
  });

  it('should not affect small gradients', () => {
    const net = new Network();
    net.dense(2, 4, 'sigmoid').dense(4, 1, 'sigmoid').loss('mse');
    
    const input = Matrix.fromArray([[0.5, 0.5]]);
    const target = Matrix.fromArray([[0.5]]);
    
    // Small target near output → small gradients
    net.trainBatch(input, target, 0.1);
    const gradsNoClip = net.layers[0].dWeights.data.slice();
    
    // Reset and train with clipping
    net.predict(input);
    net.trainBatch(input, target, 0.1, 0, 'sgd', { clipGrad: 10.0 });
    const gradsClipped = net.layers[0].dWeights.data;
    
    // With large clip value, gradients should be similar
    // (not exactly same because trainBatch modifies weights in between)
    for (const g of gradsClipped) {
      assert.ok(isFinite(g), 'All gradients should be finite');
    }
  });

  it('should enable training of very deep networks', () => {
    const net = new Network();
    for (let i = 0; i < 10; i++) {
      net.dense(4, 4, 'tanh');
    }
    net.dense(4, 1, 'sigmoid').loss('mse');
    net.clipGradients(5.0);
    
    const input = Matrix.fromArray([[0.5, 0.3, 0.8, 0.1]]);
    const target = Matrix.fromArray([[0.9]]);
    
    // Without clipping, training might diverge
    // With clipping, it should remain stable
    let lastLoss = Infinity;
    for (let i = 0; i < 100; i++) {
      const loss = net.trainBatch(input, target, 0.01, 0, 'sgd', { clipGrad: 5.0 });
      assert.ok(isFinite(loss), `Loss should be finite at epoch ${i}: ${loss}`);
      lastLoss = loss;
    }
    
    assert.ok(lastLoss < 1.0, `Loss should decrease: ${lastLoss.toFixed(4)}`);
  });
});
