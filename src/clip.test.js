// clip.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { clipLoss, zeroShotClassify } from './clip.js';

describe('CLIP', () => {
  test('loss is finite', () => {
    const images = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 1, 0, 0]),
    ];
    const texts = [
      new Float64Array([0.9, 0.1, 0, 0]),
      new Float64Array([0.1, 0.9, 0, 0]),
    ];
    const { loss } = clipLoss(images, texts);
    assert.ok(isFinite(loss), `Loss should be finite: ${loss}`);
  });

  test('loss is lower for matched pairs', () => {
    const images = [
      new Float64Array([1, 0, 0]), new Float64Array([0, 1, 0]),
    ];
    const matched = [
      new Float64Array([0.9, 0.1, 0]), new Float64Array([0.1, 0.9, 0]),
    ];
    const mismatched = [
      new Float64Array([0.1, 0.9, 0]), new Float64Array([0.9, 0.1, 0]),
    ];
    
    const matchedLoss = clipLoss(images, matched).loss;
    const mismatchedLoss = clipLoss(images, mismatched).loss;
    assert.ok(matchedLoss < mismatchedLoss, `Matched ${matchedLoss} < mismatched ${mismatchedLoss}`);
  });

  test('accuracy is 1.0 for perfectly aligned embeddings', () => {
    const images = [
      new Float64Array([1, 0, 0]), new Float64Array([0, 1, 0]), new Float64Array([0, 0, 1]),
    ];
    const texts = [
      new Float64Array([1, 0, 0]), new Float64Array([0, 1, 0]), new Float64Array([0, 0, 1]),
    ];
    const { i2tAccuracy, t2iAccuracy } = clipLoss(images, texts);
    assert.equal(i2tAccuracy, 1.0);
    assert.equal(t2iAccuracy, 1.0);
  });

  test('zero-shot classify returns correct class', () => {
    const imageEmb = new Float64Array([0.9, 0.1, 0]);
    const classEmbs = [
      new Float64Array([1, 0, 0]), // "cat"
      new Float64Array([0, 1, 0]), // "dog"
      new Float64Array([0, 0, 1]), // "car"
    ];
    const { classIdx } = zeroShotClassify(imageEmb, classEmbs);
    assert.equal(classIdx, 0, 'Should classify as class 0 (most similar)');
  });

  test('zero-shot returns scores for all classes', () => {
    const imageEmb = new Float64Array([1, 0]);
    const classEmbs = [new Float64Array([1, 0]), new Float64Array([0, 1])];
    const { scores } = zeroShotClassify(imageEmb, classEmbs);
    assert.equal(scores.length, 2);
    assert.ok(scores[0] > scores[1]);
  });
});
