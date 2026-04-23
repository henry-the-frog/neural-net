import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { buildIndex, search } from './cosine-sim-search.js';

describe('Cosine Search', () => {
  test('finds most similar embedding', () => {
    const embeddings = [[1,0,0], [0,1,0], [0,0,1]];
    const index = buildIndex(embeddings);
    const results = search([0.9, 0.1, 0], index, 1);
    assert.equal(results[0].idx, 0);
  });

  test('returns topK results', () => {
    const embeddings = [[1,0], [0,1], [0.5,0.5]];
    const index = buildIndex(embeddings);
    const results = search([1,0], index, 2);
    assert.equal(results.length, 2);
  });

  test('scores are sorted descending', () => {
    const embeddings = [[1,0], [0,1], [0.5,0.5]];
    const index = buildIndex(embeddings);
    const results = search([1,0], index, 3);
    assert.ok(results[0].score >= results[1].score);
    assert.ok(results[1].score >= results[2].score);
  });
});
