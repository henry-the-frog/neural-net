import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { analogySolve, wordSimilarity } from './embedding-similarity.js';

describe('Embedding Similarity', () => {
  test('identical vectors have similarity 1', () => {
    assert.ok(Math.abs(wordSimilarity([1,0], [1,0]) - 1) < 0.01);
  });

  test('orthogonal vectors have similarity 0', () => {
    assert.ok(Math.abs(wordSimilarity([1,0], [0,1])) < 0.01);
  });

  test('analogy returns ranked results', () => {
    const embs = [[1,0], [0,1], [1,1], [-1,0]];
    const results = analogySolve(embs, [1,0], [0,1], [1,1], 2);
    assert.equal(results.length, 2);
    assert.ok(results[0].score >= results[1].score);
  });
});
