import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { RBM } from '../src/rbm.js';
import { Matrix } from '../src/matrix.js';
import { createMiniDigits, trainTestSplit } from '../src/mnist.js';

describe('Restricted Boltzmann Machine', () => {
  describe('Construction', () => {
    it('should create with correct dimensions', () => {
      const rbm = new RBM(10, 5);
      assert.equal(rbm.numVisible, 10);
      assert.equal(rbm.numHidden, 5);
      assert.equal(rbm.W.rows, 10);
      assert.equal(rbm.W.cols, 5);
      assert.equal(rbm.a.rows, 10);
      assert.equal(rbm.b.rows, 5);
    });
  });

  describe('Forward/Backward', () => {
    it('should compute hidden probabilities', () => {
      const rbm = new RBM(4, 3);
      const v = new Matrix(4, 1, new Float64Array([1, 0, 1, 0]));
      const hProbs = rbm.hiddenProbs(v);
      assert.equal(hProbs.rows, 3);
      // Sigmoid output in [0, 1]
      for (let i = 0; i < 3; i++) {
        assert.ok(hProbs.data[i] >= 0 && hProbs.data[i] <= 1);
      }
    });

    it('should compute visible probabilities', () => {
      const rbm = new RBM(4, 3);
      const h = new Matrix(3, 1, new Float64Array([1, 0, 1]));
      const vProbs = rbm.visibleProbs(h);
      assert.equal(vProbs.rows, 4);
      for (let i = 0; i < 4; i++) {
        assert.ok(vProbs.data[i] >= 0 && vProbs.data[i] <= 1);
      }
    });

    it('should sample binary hidden units', () => {
      const rbm = new RBM(4, 3);
      const v = new Matrix(4, 1, new Float64Array([1, 0, 1, 0]));
      const { samples } = rbm.sampleHidden(v);
      for (let i = 0; i < 3; i++) {
        assert.ok(samples.data[i] === 0 || samples.data[i] === 1);
      }
    });
  });

  describe('Training', () => {
    it('should reduce reconstruction error', () => {
      const rbm = new RBM(4, 8, { learningRate: 0.1 });
      const data = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
      ];

      const { history } = rbm.train(data, { epochs: 50 });
      assert.equal(history.length, 50);
      // Error should decrease
      assert.ok(history[history.length - 1] < history[0],
        `Error should decrease: ${history[0].toFixed(4)} → ${history[history.length - 1].toFixed(4)}`);
    });

    it('should learn binary patterns', () => {
      const rbm = new RBM(4, 8, { learningRate: 0.1, momentum: 0.5 });
      const patterns = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ];

      rbm.train(patterns, { epochs: 100 });

      // Reconstruct should be close to input
      for (const pattern of patterns) {
        const { reconstruction } = rbm.reconstruct(pattern);
        const error = pattern.reduce((sum, p, i) => sum + (p - reconstruction.data[i]) ** 2, 0) / 4;
        assert.ok(error < 0.3, `Reconstruction error too high: ${error.toFixed(4)}`);
      }
    });

    it('should handle Matrix input', () => {
      const rbm = new RBM(4, 3);
      const v = new Matrix(4, 1, new Float64Array([1, 0, 1, 0]));
      const { reconstructionError } = rbm.trainStep(v);
      assert.ok(typeof reconstructionError === 'number');
      assert.ok(!isNaN(reconstructionError));
    });
  });

  describe('Free Energy', () => {
    it('should compute free energy', () => {
      const rbm = new RBM(4, 3);
      const energy = rbm.freeEnergy([1, 0, 1, 0]);
      assert.ok(typeof energy === 'number');
      assert.ok(!isNaN(energy));
    });

    it('trained patterns should have lower free energy', () => {
      const rbm = new RBM(4, 8, { learningRate: 0.1 });
      const pattern = [1, 0, 1, 0];
      const antiPattern = [0, 1, 0, 1];

      // Train only on one pattern
      for (let i = 0; i < 100; i++) {
        rbm.trainStep(new Matrix(4, 1, new Float64Array(pattern)));
      }

      const trainedEnergy = rbm.freeEnergy(pattern);
      const novelEnergy = rbm.freeEnergy(antiPattern);

      // Trained pattern should have lower (more negative) energy
      assert.ok(trainedEnergy < novelEnergy,
        `Trained energy (${trainedEnergy.toFixed(2)}) should be < novel (${novelEnergy.toFixed(2)})`);
    });
  });

  describe('Feature Extraction', () => {
    it('should encode and decode', () => {
      const rbm = new RBM(4, 3);
      const encoded = rbm.encode([1, 0, 1, 0]);
      assert.equal(encoded.length, 3);
      encoded.forEach(v => assert.ok(v >= 0 && v <= 1));

      const decoded = rbm.decode(encoded);
      assert.equal(decoded.length, 4);
      decoded.forEach(v => assert.ok(v >= 0 && v <= 1));
    });

    it('should learn useful features from digits', () => {
      const data = createMiniDigits({ samplesPerDigit: 10, noise: 0.02 });
      const rbm = new RBM(64, 32, { learningRate: 0.01, momentum: 0.5 });

      const inputs = data.images.map(img => Array.from(img.data));
      const { history } = rbm.train(inputs, { epochs: 20 });

      // Error should decrease
      assert.ok(history[history.length - 1] < history[0]);

      // Encoded features should be different for different digits
      const feat0 = rbm.encode(inputs[0]);
      const feat1 = rbm.encode(inputs[1]);
      let diff = 0;
      for (let i = 0; i < 32; i++) diff += Math.abs(feat0[i] - feat1[i]);
      // Features should differ (not all identical)
      assert.ok(diff > 0.5, `Features should differ between samples, diff=${diff.toFixed(3)}`);
    });
  });

  describe('Generation', () => {
    it('should generate visible samples', () => {
      const rbm = new RBM(4, 3);
      rbm.train([[1, 0, 1, 0]], { epochs: 10 });

      const sample = rbm.generate(50);
      assert.equal(sample.rows, 4);
      for (let i = 0; i < 4; i++) {
        assert.ok(sample.data[i] >= 0 && sample.data[i] <= 1);
      }
    });
  });

  describe('Callbacks', () => {
    it('should call onEpoch', () => {
      const rbm = new RBM(4, 3);
      const calls = [];
      rbm.train([[1, 0, 1, 0]], {
        epochs: 3,
        onEpoch: (data) => calls.push(data),
      });
      assert.equal(calls.length, 3);
      assert.equal(calls[0].epoch, 0);
    });
  });
});
