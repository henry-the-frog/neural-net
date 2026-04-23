// contrastive.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { cosineSimilarity, ntXentLoss, infoNCELoss, tripletLoss } from './contrastive.js';

describe('Contrastive Learning', () => {
  test('cosine similarity of identical vectors is 1', () => {
    const a = new Float64Array([1, 2, 3]);
    assert.ok(Math.abs(cosineSimilarity(a, a) - 1) < 0.001);
  });

  test('cosine similarity of orthogonal vectors is 0', () => {
    const a = new Float64Array([1, 0, 0]);
    const b = new Float64Array([0, 1, 0]);
    assert.ok(Math.abs(cosineSimilarity(a, b)) < 0.001);
  });

  test('cosine similarity of opposite vectors is -1', () => {
    const a = new Float64Array([1, 0, 0]);
    const b = new Float64Array([-1, 0, 0]);
    assert.ok(Math.abs(cosineSimilarity(a, b) + 1) < 0.001);
  });

  test('NT-Xent loss is finite', () => {
    const embeddings = [
      new Float64Array([1, 0, 0]),
      new Float64Array([0.9, 0.1, 0]), // Positive pair
      new Float64Array([0, 1, 0]),
      new Float64Array([0.1, 0.9, 0]), // Positive pair
    ];
    const { loss } = ntXentLoss(embeddings);
    assert.ok(isFinite(loss), `Loss should be finite: ${loss}`);
  });

  test('NT-Xent loss is lower for well-separated pairs', () => {
    // Good: pairs are close to each other, far from others
    const good = [
      new Float64Array([1, 0, 0]), new Float64Array([0.95, 0.05, 0]),
      new Float64Array([0, 1, 0]), new Float64Array([0.05, 0.95, 0]),
    ];
    // Bad: pairs are close to wrong partners
    const bad = [
      new Float64Array([1, 0, 0]), new Float64Array([0, 1, 0]),
      new Float64Array([0.9, 0.1, 0]), new Float64Array([0.1, 0.9, 0]),
    ];
    
    const goodLoss = ntXentLoss(good).loss;
    const badLoss = ntXentLoss(bad).loss;
    assert.ok(goodLoss < badLoss, `Good ${goodLoss} should be < bad ${badLoss}`);
  });

  test('infoNCE loss decreases when positive is closer', () => {
    const anchor = new Float64Array([1, 0, 0]);
    const close = new Float64Array([0.95, 0.05, 0]);
    const far = new Float64Array([0.5, 0.5, 0]);
    const negatives = [new Float64Array([0, 1, 0]), new Float64Array([0, 0, 1])];
    
    const lossClose = infoNCELoss(anchor, close, negatives);
    const lossFar = infoNCELoss(anchor, far, negatives);
    assert.ok(lossClose < lossFar, `Close positive loss ${lossClose} should be < far ${lossFar}`);
  });

  test('triplet loss is 0 when margin satisfied', () => {
    const anchor = new Float64Array([1, 0, 0]);
    const positive = new Float64Array([0.9, 0.1, 0]); // Close
    const negative = new Float64Array([-1, 0, 0]); // Far
    const loss = tripletLoss(anchor, positive, negative, 0.2);
    assert.equal(loss, 0, 'Should be 0 when negative is much farther than positive');
  });

  test('triplet loss is positive when margin violated', () => {
    const anchor = new Float64Array([0, 0, 0]);
    const positive = new Float64Array([2, 0, 0]); // Far from anchor
    const negative = new Float64Array([0.5, 0, 0]); // Closer than positive!
    const loss = tripletLoss(anchor, positive, negative, 0.2);
    assert.ok(loss > 0, `Loss should be positive when negative is closer: ${loss}`);
  });
});
