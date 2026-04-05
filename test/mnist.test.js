import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { parseIDX, createMiniDigits, trainTestSplit, accuracy, predict, evaluate, confusionMatrix } from '../src/mnist.js';
import { Network } from '../src/network.js';
import { Dense } from '../src/layer.js';
import { Adam } from '../src/optimizer.js';
import { Matrix } from '../src/matrix.js';

describe('MNIST Utilities', () => {
  describe('parseIDX', () => {
    it('should parse a label IDX buffer', () => {
      // Magic: 0x00000801 = labels, 1 dimension
      // Count: 3
      // Data: [0, 5, 9]
      const buf = new ArrayBuffer(4 + 4 + 3);
      const view = new DataView(buf);
      view.setUint32(0, 0x00000801); // magic: type=uint8, ndims=1
      view.setUint32(4, 3);           // count
      const data = new Uint8Array(buf, 8);
      data[0] = 0; data[1] = 5; data[2] = 9;

      const result = parseIDX(buf);
      assert.deepStrictEqual(result.dims, [3]);
      assert.equal(result.data[0], 0);
      assert.equal(result.data[1], 5);
      assert.equal(result.data[2], 9);
    });

    it('should parse an image IDX buffer', () => {
      // Magic: 0x00000803 = images, 3 dimensions
      // Count: 2, rows: 2, cols: 2
      const pixelCount = 2 * 2 * 2;
      const buf = new ArrayBuffer(4 + 12 + pixelCount);
      const view = new DataView(buf);
      view.setUint32(0, 0x00000803); // magic: type=uint8, ndims=3
      view.setUint32(4, 2);           // count
      view.setUint32(8, 2);           // rows
      view.setUint32(12, 2);          // cols
      const data = new Uint8Array(buf, 16);
      for (let i = 0; i < pixelCount; i++) data[i] = i * 30;

      const result = parseIDX(buf);
      assert.deepStrictEqual(result.dims, [2, 2, 2]);
      assert.equal(result.data.length, pixelCount);
    });
  });

  describe('createMiniDigits', () => {
    it('should create default dataset (10 samples per digit)', () => {
      const data = createMiniDigits();
      assert.equal(data.images.length, 100); // 10 digits × 10 samples
      assert.equal(data.labels.length, 100);
      assert.equal(data.rawLabels.length, 100);
      assert.equal(data.rows, 8);
      assert.equal(data.cols, 8);
    });

    it('should create images with correct dimensions', () => {
      const data = createMiniDigits({ samplesPerDigit: 3 });
      assert.equal(data.images.length, 30);
      assert.equal(data.images[0].rows, 64); // 8×8 flattened
      assert.equal(data.images[0].cols, 1);
    });

    it('should have all 10 digit classes', () => {
      const data = createMiniDigits({ samplesPerDigit: 5 });
      const classes = new Set(data.rawLabels);
      assert.equal(classes.size, 10);
    });

    it('should produce normalized pixel values', () => {
      const data = createMiniDigits({ noise: 0 });
      for (const img of data.images) {
        for (let i = 0; i < img.data.length; i++) {
          assert.ok(img.data[i] >= 0 && img.data[i] <= 1, `Pixel out of range: ${img.data[i]}`);
        }
      }
    });

    it('should shuffle the data', () => {
      const data = createMiniDigits();
      // Check that first 10 labels are not all the same
      const firstTen = data.rawLabels.slice(0, 10);
      const unique = new Set(firstTen);
      assert.ok(unique.size > 1, 'Data should be shuffled');
    });
  });

  describe('trainTestSplit', () => {
    it('should split dataset correctly', () => {
      const data = createMiniDigits({ samplesPerDigit: 10 });
      const { train, test } = trainTestSplit(data.images, data.labels, data.rawLabels, 0.2);
      assert.equal(train.images.length, 80);
      assert.equal(test.images.length, 20);
      assert.equal(train.labels.length, 80);
      assert.equal(test.labels.length, 20);
    });
  });

  describe('accuracy', () => {
    it('should compute correct accuracy', () => {
      assert.equal(accuracy([0, 1, 2, 3], [0, 1, 2, 3]), 1.0);
      assert.equal(accuracy([0, 1, 2, 3], [0, 0, 0, 0]), 0.25);
      assert.equal(accuracy([1, 1, 1, 1], [1, 1, 1, 1]), 1.0);
    });
  });

  describe('confusionMatrix', () => {
    it('should compute confusion matrix', () => {
      const preds = [0, 0, 1, 1, 2];
      const truth = [0, 1, 1, 2, 2];
      const cm = confusionMatrix(preds, truth, 3);
      assert.equal(cm[0][0], 1); // true 0 predicted 0
      assert.equal(cm[1][0], 1); // true 1 predicted 0
      assert.equal(cm[1][1], 1); // true 1 predicted 1
      assert.equal(cm[2][1], 1); // true 2 predicted 1
      assert.equal(cm[2][2], 1); // true 2 predicted 2
    });
  });

  describe('End-to-end training', () => {
    it('should train a simple network on mini digits', () => {
      const data = createMiniDigits({ samplesPerDigit: 30, noise: 0.02 });
      const { train, test } = trainTestSplit(data.images, data.labels, data.rawLabels, 0.2);

      const net = new Network();
      net.add(new Dense(64, 32, 'relu'));
      net.add(new Dense(32, 10, 'softmax'));
      net.loss('crossEntropy');

      // Pack into Matrix format expected by Network.train
      const n = train.images.length;
      const inputData = new Float64Array(n * 64);
      const targetData = new Float64Array(n * 10);
      for (let i = 0; i < n; i++) {
        inputData.set(train.images[i].data, i * 64);
        targetData.set(train.labels[i].data, i * 10);
      }
      const inputs = new Matrix(n, 64, inputData);
      const targets = new Matrix(n, 10, targetData);

      // Train
      net.train({ inputs, targets }, {
        epochs: 100,
        learningRate: 0.005,
        batchSize: 16,
      });

      // Evaluate
      const acc = evaluate(net, test.images, test.rawLabels);
      assert.ok(acc > 0.4, `Expected accuracy > 40%, got ${(acc * 100).toFixed(1)}%`);
    });

    it('should produce valid predictions with predict()', () => {
      const data = createMiniDigits({ samplesPerDigit: 5 });
      const net = new Network();
      net.add(new Dense(64, 16, 'relu'));
      net.add(new Dense(16, 10, 'softmax'));
      const pred = predict(net, data.images[0]);
      assert.ok(pred >= 0 && pred <= 9, `Prediction should be 0-9, got ${pred}`);
    });
  });
});
