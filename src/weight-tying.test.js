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
    const { embedding, lmHead } = createTiedWeights(10, 4);
    // Test: embedding row i dotted with hidden should give logit for token i
    const hidden = new Matrix(1, 4);
    for (let j = 0; j < 4; j++) hidden.set(0, j, embedding.get(3, j));
    
    const logits = lmHead(hidden);
    // Logit for token 3 should be highest (it's dot product of token 3 embedding with itself)
    let maxIdx = 0;
    for (let i = 1; i < 10; i++) {
      if (logits.get(0, i) > logits.get(0, maxIdx)) maxIdx = i;
    }
    assert.equal(maxIdx, 3, 'Token 3 embedding should give highest logit for token 3');
  });

  test('savings is 50%', () => {
    const savings = tyingSavings(50000, 768);
    assert.equal(savings.savings, '50.0%');
    assert.equal(savings.saved, 50000 * 768);
  });
});
