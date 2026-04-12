// model-zoo.test.js — Tests for pre-configured model architectures
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ModelZoo } from '../src/model-zoo.js';
import { Matrix } from '../src/matrix.js';

describe('ModelZoo', () => {
  it('xor: learns XOR in 500 epochs', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const net = ModelZoo.xor();
      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);
      for (let e = 0; e < 500; e++) net.trainBatch(inputs, targets, 0.5);
      const pred = net.predict(inputs);
      if (pred.get(0, 0) < 0.3 && pred.get(1, 0) > 0.7) passed = true;
    }
    assert.ok(passed, 'XOR model should learn in 1 of 3 attempts');
  });

  it('binaryClassifier: correct shape', () => {
    const net = ModelZoo.binaryClassifier(10);
    const pred = net.predict(Matrix.random(5, 10));
    assert.equal(pred.rows, 5);
    assert.equal(pred.cols, 1);
  });

  it('classifier: correct shape', () => {
    const net = ModelZoo.classifier(10, 3);
    const pred = net.predict(Matrix.random(5, 10));
    assert.equal(pred.rows, 5);
    assert.equal(pred.cols, 3);
  });

  it('regression: learns linear function', () => {
    const net = ModelZoo.regression(1);
    const inputs = Matrix.fromArray([[1], [2], [3], [4], [5]]);
    const targets = Matrix.fromArray([[2], [4], [6], [8], [10]]);
    for (let e = 0; e < 200; e++) net.trainBatch(inputs, targets, 0.01);
    const pred = net.predict(Matrix.fromArray([[3]]));
    assert.ok(Math.abs(pred.get(0, 0) - 6) < 2, `Should predict ~6: ${pred.get(0, 0).toFixed(1)}`);
  });

  it('autoencoder: encoder → decoder roundtrip', () => {
    const { net } = ModelZoo.autoencoder(4, 2);
    const input = Matrix.fromArray([[0.5, 0.3, 0.8, 0.1]]);
    for (let e = 0; e < 200; e++) net.trainBatch(input, input, 0.1);
    const recon = net.predict(input);
    let mse = 0;
    for (let i = 0; i < 4; i++) mse += (recon.get(0, i) - input.get(0, i)) ** 2;
    mse /= 4;
    assert.ok(mse < 0.1, `Autoencoder should reconstruct: MSE=${mse.toFixed(4)}`);
  });

  it('timeSeries: correct shape', () => {
    const net = ModelZoo.timeSeries(10, 1, 3);
    const pred = net.predict(Matrix.random(5, 10));
    assert.equal(pred.rows, 5);
    assert.equal(pred.cols, 3);
  });

  it('tiny: works', () => {
    const net = ModelZoo.tiny();
    const pred = net.predict(Matrix.random(1, 2));
    assert.equal(pred.cols, 1);
    assert.ok(pred.data.every(Number.isFinite));
  });

  it('deep: 5 layers produce finite output', () => {
    const net = ModelZoo.deep(4, 2);
    const pred = net.predict(Matrix.random(3, 4));
    assert.equal(pred.cols, 2);
    assert.ok(pred.data.every(Number.isFinite));
  });

  it('wide: single wide layer works', () => {
    const net = ModelZoo.wide(4, 1, 128);
    const pred = net.predict(Matrix.random(1, 4));
    assert.ok(pred.data.every(Number.isFinite));
  });
});
