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
    // Pairing convention: (i, i+N) — [view1_s0, view1_s1, view2_s0, view2_s1]
    const embeddings = [
      new Float64Array([1, 0, 0]),       // sample 0, view 1
      new Float64Array([0, 1, 0]),       // sample 1, view 1
      new Float64Array([0.9, 0.1, 0]),   // sample 0, view 2
      new Float64Array([0.1, 0.9, 0]),   // sample 1, view 2
    ];
    const loss = ntXentLoss(embeddings);
    assert.ok(isFinite(loss), `Loss should be finite: ${loss}`);
  });

  test('NT-Xent loss is lower for well-separated pairs', () => {
    // Good: positive pairs (i, i+N) are close, negatives are far
    // Layout: [s0_v1, s1_v1, s0_v2, s1_v2]
    const good = [
      new Float64Array([1, 0, 0]), new Float64Array([0, 1, 0]),     // views 1
      new Float64Array([0.95, 0.05, 0]), new Float64Array([0.05, 0.95, 0]), // views 2 (close to view 1)
    ];
    // Bad: positive pairs are orthogonal
    const bad = [
      new Float64Array([1, 0, 0]), new Float64Array([0.9, 0.1, 0]), // views 1
      new Float64Array([0, 1, 0]), new Float64Array([0.1, 0.9, 0]), // views 2 (pair 0→2 is orth, pair 1→3 is orth)
    ];
    
    const goodLoss = ntXentLoss(good);
    const badLoss = ntXentLoss(bad);
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
    const positive = new Float64Array([0.9, 0.1, 0]);
    const negative = new Float64Array([-1, 0, 0]);
    const loss = tripletLoss(anchor, positive, negative, 0.2);
    assert.equal(loss, 0, 'Should be 0 when negative is much farther than positive');
  });

  test('triplet loss is positive when margin violated', () => {
    const anchor = new Float64Array([0, 0, 0]);
    const positive = new Float64Array([2, 0, 0]);
    const negative = new Float64Array([0.5, 0, 0]);
    const loss = tripletLoss(anchor, positive, negative, 0.2);
    assert.ok(loss > 0, `Loss should be positive when negative is closer: ${loss}`);
  });
});
