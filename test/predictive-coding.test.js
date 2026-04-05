import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { PredictiveCodingLayer } from '../src/predictive-coding.js';
import { Matrix } from '../src/matrix.js';

describe('PredictiveCodingLayer', () => {
  it('should create a layer with correct dimensions', () => {
    const layer = new PredictiveCodingLayer(10, 5);
    assert.equal(layer.size, 10);
    assert.equal(layer.inputSize, 5);
    assert.equal(layer.mu.rows, 10);
    assert.equal(layer.mu.cols, 1);
    assert.equal(layer.W.rows, 5);
    assert.equal(layer.W.cols, 10);
    assert.equal(layer.b.rows, 5);
    assert.equal(layer.b.cols, 1);
  });

  it('should generate predictions with correct shape', () => {
    const layer = new PredictiveCodingLayer(8, 4);
    const pred = layer.predict();
    assert.equal(pred.rows, 4);
    assert.equal(pred.cols, 1);
    // Sigmoid output should be in [0, 1]
    for (let i = 0; i < pred.data.length; i++) {
      assert.ok(pred.data[i] >= 0 && pred.data[i] <= 1);
    }
  });

  it('should compute prediction errors', () => {
    const layer = new PredictiveCodingLayer(8, 4);
    const actual = new Matrix(4, 1, new Float64Array([0.5, 0.3, 0.8, 0.1]));
    const error = layer.computeError(actual);
    assert.equal(error.rows, 4);
    assert.equal(error.cols, 1);
    // Error = actual - predicted
    const predicted = layer.predict();
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(error.data[i] - (actual.data[i] - predicted.data[i])) < 1e-10);
    }
  });

  it('should update values (inference step)', () => {
    const layer = new PredictiveCodingLayer(8, 4, { inferenceRate: 0.1 });
    const actual = new Matrix(4, 1, new Float64Array([1, 0, 1, 0]));
    
    layer.computeError(actual);
    const muBefore = new Float64Array(layer.mu.data);
    layer.updateValues(null); // Top layer, no error from above
    
    // Values should have changed
    let changed = false;
    for (let i = 0; i < layer.mu.data.length; i++) {
      if (Math.abs(layer.mu.data[i] - muBefore[i]) > 1e-10) changed = true;
    }
    assert.ok(changed, 'Value nodes should update during inference');
  });

  it('should update weights (learning step)', () => {
    const layer = new PredictiveCodingLayer(8, 4, { learningRate: 0.01 });
    const actual = new Matrix(4, 1, new Float64Array([1, 0, 1, 0]));
    
    layer.computeError(actual);
    const wBefore = new Float64Array(layer.W.data);
    layer.updateWeights();
    
    let changed = false;
    for (let i = 0; i < layer.W.data.length; i++) {
      if (Math.abs(layer.W.data[i] - wBefore[i]) > 1e-10) changed = true;
    }
    assert.ok(changed, 'Weights should update during learning');
  });

  it('should reduce prediction error over inference steps', () => {
    const layer = new PredictiveCodingLayer(8, 4, { inferenceRate: 0.1 });
    const target = new Matrix(4, 1, new Float64Array([0.8, 0.2, 0.9, 0.1]));

    // Initial error
    layer.computeError(target);
    const initialEnergy = layer.energy;

    // Run inference steps
    for (let i = 0; i < 50; i++) {
      layer.computeError(target);
      layer.updateValues(null);
    }

    layer.computeError(target);
    const finalEnergy = layer.energy;

    assert.ok(finalEnergy < initialEnergy, 
      `Energy should decrease: ${initialEnergy.toFixed(4)} → ${finalEnergy.toFixed(4)}`);
  });

  it('should reduce error with learning (weight updates)', () => {
    const layer = new PredictiveCodingLayer(4, 2, { 
      learningRate: 0.01, inferenceRate: 0.1, activation: 'sigmoid' 
    });
    const target = new Matrix(2, 1, new Float64Array([0.8, 0.2]));

    // Record initial error
    layer.computeError(target);
    const initialEnergy = layer.energy;

    // Multiple learning episodes
    for (let ep = 0; ep < 20; ep++) {
      layer.resetValues();
      // Inference phase (settle)
      for (let i = 0; i < 30; i++) {
        layer.computeError(target);
        layer.updateValues(null);
      }
      // Learning phase
      layer.computeError(target);
      layer.updateWeights();
    }

    // Check final error
    layer.resetValues();
    for (let i = 0; i < 30; i++) {
      layer.computeError(target);
      layer.updateValues(null);
    }
    layer.computeError(target);
    const finalEnergy = layer.energy;

    assert.ok(finalEnergy < initialEnergy, 
      `Energy should decrease with learning: ${initialEnergy.toFixed(4)} → ${finalEnergy.toFixed(4)}`);
  });

  it('should support different activations', () => {
    for (const act of ['sigmoid', 'tanh', 'relu', 'linear']) {
      const layer = new PredictiveCodingLayer(4, 2, { activation: act });
      const pred = layer.predict();
      assert.equal(pred.rows, 2);
      assert.ok(!pred.data.some(isNaN), `${act} produced NaN`);
    }
  });

  it('should clone correctly', () => {
    const layer = new PredictiveCodingLayer(8, 4);
    const clone = layer.clone();
    assert.equal(clone.size, layer.size);
    assert.equal(clone.inputSize, layer.inputSize);
    // Modifying clone should not affect original
    clone.W.data[0] = 999;
    assert.notEqual(layer.W.data[0], 999);
  });

  it('should handle error from above (middle layer)', () => {
    const layer = new PredictiveCodingLayer(8, 4, { inferenceRate: 0.1 });
    const actual = new Matrix(4, 1, new Float64Array([0.5, 0.5, 0.5, 0.5]));
    const errorFromAbove = new Matrix(8, 1);
    for (let i = 0; i < 8; i++) errorFromAbove.data[i] = 0.1;

    layer.computeError(actual);
    const muBefore = new Float64Array(layer.mu.data);
    layer.updateValues(errorFromAbove);

    let changed = false;
    for (let i = 0; i < layer.mu.data.length; i++) {
      if (Math.abs(layer.mu.data[i] - muBefore[i]) > 1e-10) changed = true;
    }
    assert.ok(changed, 'Should update with error from above');
  });
});
