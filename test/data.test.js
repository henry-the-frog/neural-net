// data.test.js — Tests for data utilities

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import {
  shuffle, trainTestSplit, normalize, applyNormalization,
  minMaxScale, addNoise, createBatches, oneHotEncode
} from '../src/data.js';

describe('Data Utilities', () => {
  describe('shuffle', () => {
    it('should preserve all data points', () => {
      const inputs = Matrix.fromArray([[1], [2], [3], [4], [5]]);
      const targets = Matrix.fromArray([[10], [20], [30], [40], [50]]);
      
      const { inputs: si, targets: st } = shuffle(inputs, targets);
      
      assert.equal(si.rows, 5);
      assert.equal(st.rows, 5);
      
      // Sum should be preserved
      const inputSum = Array.from(si.data).reduce((a, b) => a + b, 0);
      assert.equal(inputSum, 15);
    });

    it('should keep input-target pairs aligned', () => {
      const inputs = Matrix.fromArray([[1], [2], [3]]);
      const targets = Matrix.fromArray([[10], [20], [30]]);
      
      const { inputs: si, targets: st } = shuffle(inputs, targets);
      
      // Each input should have its matching target (input × 10)
      for (let i = 0; i < si.rows; i++) {
        assert.equal(st.get(i, 0), si.get(i, 0) * 10,
          `Pair ${i} mismatched: ${si.get(i, 0)} → ${st.get(i, 0)}`);
      }
    });
  });

  describe('trainTestSplit', () => {
    it('should split into correct proportions', () => {
      const inputs = Matrix.fromArray(Array.from({ length: 100 }, () => [1]));
      const targets = Matrix.fromArray(Array.from({ length: 100 }, () => [0]));
      
      const { train, test } = trainTestSplit(inputs, targets, 0.2);
      assert.equal(train.inputs.rows, 80);
      assert.equal(test.inputs.rows, 20);
    });

    it('should not lose data', () => {
      const inputs = Matrix.fromArray([[1], [2], [3], [4], [5]]);
      const targets = Matrix.fromArray([[0], [0], [0], [0], [0]]);
      
      const { train, test } = trainTestSplit(inputs, targets, 0.4);
      assert.equal(train.inputs.rows + test.inputs.rows, 5);
    });
  });

  describe('normalize', () => {
    it('should produce zero mean', () => {
      const data = Matrix.fromArray([[1, 10], [3, 20], [5, 30]]);
      const { normalized, mean } = normalize(data);
      
      // Check mean of normalized is ~0
      for (let j = 0; j < normalized.cols; j++) {
        let sum = 0;
        for (let i = 0; i < normalized.rows; i++) sum += normalized.get(i, j);
        assert.ok(Math.abs(sum / normalized.rows) < 1e-10, `Column ${j} mean not zero`);
      }
    });

    it('should produce unit variance', () => {
      const data = Matrix.fromArray([[1, 10], [3, 20], [5, 30]]);
      const { normalized } = normalize(data);
      
      for (let j = 0; j < normalized.cols; j++) {
        let sumSq = 0;
        for (let i = 0; i < normalized.rows; i++) {
          sumSq += normalized.get(i, j) * normalized.get(i, j);
        }
        const variance = sumSq / normalized.rows;
        assert.ok(Math.abs(variance - 1) < 0.01, `Column ${j} variance: ${variance}`);
      }
    });

    it('should allow re-applying with same stats', () => {
      const data = Matrix.fromArray([[1, 10], [3, 20], [5, 30]]);
      const { normalized, mean, std } = normalize(data);
      
      const reapplied = applyNormalization(data, mean, std);
      for (let i = 0; i < normalized.data.length; i++) {
        assert.ok(Math.abs(normalized.data[i] - reapplied.data[i]) < 1e-10);
      }
    });
  });

  describe('minMaxScale', () => {
    it('should scale to [0, 1]', () => {
      const data = Matrix.fromArray([[1], [5], [10]]);
      const { scaled } = minMaxScale(data);
      
      assert.ok(Math.abs(scaled.get(0, 0) - 0) < 1e-10); // min → 0
      assert.ok(Math.abs(scaled.get(2, 0) - 1) < 1e-10); // max → 1
      assert.ok(scaled.get(1, 0) > 0 && scaled.get(1, 0) < 1); // mid
    });
  });

  describe('addNoise', () => {
    it('should perturb data', () => {
      const data = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
      const noisy = addNoise(data, 0.1);
      
      // Should be different from original
      let diffCount = 0;
      for (let i = 0; i < data.data.length; i++) {
        if (Math.abs(data.data[i] - noisy.data[i]) > 0) diffCount++;
      }
      assert.ok(diffCount > 0, 'Noise should change at least some values');
    });

    it('should preserve rough magnitude', () => {
      const data = Matrix.fromArray([[100, 200]]);
      const noisy = addNoise(data, 0.01);
      
      assert.ok(Math.abs(noisy.get(0, 0) - 100) < 1, 'Noise should be small');
    });
  });

  describe('createBatches', () => {
    it('should create correct number of batches', () => {
      const inputs = Matrix.fromArray(Array.from({ length: 10 }, () => [1]));
      const targets = Matrix.fromArray(Array.from({ length: 10 }, () => [0]));
      
      const batches = createBatches(inputs, targets, 3);
      assert.equal(batches.length, 4); // 3, 3, 3, 1
      assert.equal(batches[0].inputs.rows, 3);
      assert.equal(batches[3].inputs.rows, 1);
    });

    it('should contain all data', () => {
      const inputs = Matrix.fromArray([[1], [2], [3], [4], [5]]);
      const targets = Matrix.fromArray([[0], [0], [0], [0], [0]]);
      
      const batches = createBatches(inputs, targets, 2);
      let total = 0;
      for (const b of batches) total += b.inputs.rows;
      assert.equal(total, 5);
    });
  });

  describe('oneHotEncode', () => {
    it('should create correct one-hot vectors', () => {
      const encoded = oneHotEncode([0, 1, 2]);
      assert.equal(encoded.rows, 3);
      assert.equal(encoded.cols, 3);
      assert.equal(encoded.get(0, 0), 1);
      assert.equal(encoded.get(1, 1), 1);
      assert.equal(encoded.get(2, 2), 1);
    });

    it('should have zeros except at the class index', () => {
      const encoded = oneHotEncode([2], 5);
      assert.equal(encoded.cols, 5);
      assert.equal(encoded.get(0, 2), 1);
      let sum = 0;
      for (let j = 0; j < 5; j++) sum += encoded.get(0, j);
      assert.equal(sum, 1);
    });
  });
});
