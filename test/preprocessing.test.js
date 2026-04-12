// preprocessing.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { StandardScaler, MinMaxScaler, oneHotEncode, trainTestSplit } from '../src/preprocessing.js';
import { Matrix } from '../src/matrix.js';

describe('StandardScaler', () => {
  it('produces zero mean', () => {
    const data = Matrix.fromArray([[1, 10], [2, 20], [3, 30]]);
    const scaler = new StandardScaler();
    const scaled = scaler.fitTransform(data);
    for (let j = 0; j < 2; j++) {
      let sum = 0;
      for (let i = 0; i < 3; i++) sum += scaled.get(i, j);
      assert.ok(Math.abs(sum / 3) < 0.01, `Col ${j} mean should be ~0`);
    }
  });

  it('produces unit variance', () => {
    const data = Matrix.fromArray([[1, 10], [2, 20], [3, 30]]);
    const scaler = new StandardScaler();
    const scaled = scaler.fitTransform(data);
    for (let j = 0; j < 2; j++) {
      let sumSq = 0;
      for (let i = 0; i < 3; i++) sumSq += scaled.get(i, j) ** 2;
      assert.ok(Math.abs(sumSq / 3 - 1) < 0.1, `Col ${j} var should be ~1`);
    }
  });

  it('inverse transform recovers original', () => {
    const data = Matrix.fromArray([[1, 10], [2, 20], [3, 30]]);
    const scaler = new StandardScaler();
    const scaled = scaler.fitTransform(data);
    const recovered = scaler.inverseTransform(scaled);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) {
        assert.ok(Math.abs(recovered.get(i, j) - data.get(i, j)) < 0.01);
      }
    }
  });
});

describe('MinMaxScaler', () => {
  it('scales to [0, 1]', () => {
    const data = Matrix.fromArray([[1], [5], [10]]);
    const scaler = new MinMaxScaler();
    const scaled = scaler.fitTransform(data);
    assert.ok(Math.abs(scaled.get(0, 0) - 0) < 0.01);
    assert.ok(Math.abs(scaled.get(2, 0) - 1) < 0.01);
  });
});

describe('oneHotEncode', () => {
  it('encodes correctly', () => {
    const encoded = oneHotEncode([0, 1, 2, 1], 3);
    assert.equal(encoded.rows, 4);
    assert.equal(encoded.cols, 3);
    assert.equal(encoded.get(0, 0), 1);
    assert.equal(encoded.get(1, 1), 1);
    assert.equal(encoded.get(2, 2), 1);
  });

  it('auto-detects numClasses', () => {
    const encoded = oneHotEncode([0, 3, 1]);
    assert.equal(encoded.cols, 4); // 0-3
  });
});

describe('trainTestSplit', () => {
  it('correct sizes', () => {
    const inputs = Matrix.random(100, 2);
    const targets = Matrix.random(100, 1);
    const { trainInputs, testInputs } = trainTestSplit(inputs, targets, 0.2);
    assert.equal(trainInputs.rows, 80);
    assert.equal(testInputs.rows, 20);
  });

  it('no data loss', () => {
    const inputs = Matrix.random(50, 1);
    const targets = Matrix.random(50, 1);
    const { trainInputs, testInputs } = trainTestSplit(inputs, targets, 0.3);
    assert.equal(trainInputs.rows + testInputs.rows, 50);
  });
});
