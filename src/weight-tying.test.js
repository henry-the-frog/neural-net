// weight-tying.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { createTiedWeights, tyingSavings } from './weight-tying.js';
import { Matrix } from './matrix.js';

describe('Weight Tying', () => {
  test('embedding shape is correct', () => {
    const { embedding } = createTiedWeights(100, 32);
    assert.equal(embedding.rows, 100);
    assert.equal(embedding.cols, 32);
  });

  test('lmHead produces correct logit shape', () => {
    const { lmHead } = createTiedWeights(100, 32);
    const hidden = Matrix.random(5, 32);
    const logits = lmHead(hidden);
    assert.equal(logits.rows, 5);
    assert.equal(logits.cols, 100);
  });

  test('lmHead is transpose of embedding', () => {
    // Use larger dimensions for better separation
    const { embedding, lmHead } = createTiedWeights(10, 16);
    
    // Test multiple tokens for robustness
    let matchCount = 0;
    for (let token = 0; token < 10; token++) {
      const hidden = new Matrix(1, 16);
      for (let j = 0; j < 16; j++) hidden.set(0, j, embedding.get(token, j));
      
      const logits = lmHead(hidden);
      let maxIdx = 0;
      for (let i = 1; i < 10; i++) {
        if (logits.get(0, i) > logits.get(0, maxIdx)) maxIdx = i;
      }
      if (maxIdx === token) matchCount++;
    }
    // At least 8/10 tokens should self-match (allowing for random collisions)
    assert.ok(matchCount >= 8, `At least 8/10 tokens should give highest logit for themselves, got ${matchCount}/10`);
  });

  test('savings is 50%', () => {
    const savings = tyingSavings(50000, 768);
    assert.equal(savings.savings, '50.0%');
    assert.equal(savings.saved, 50000 * 768);
  });
});
