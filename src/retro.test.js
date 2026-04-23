import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Retrieval-Augmented Language Model pre-training (REALM/RETRO concepts)
describe('RETRO', () => {
  function chunkAndRetrieve(tokens, chunkSize, retrieveFn) {
    const chunks = [];
    for (let i = 0; i < tokens.length; i += chunkSize) {
      const chunk = tokens.slice(i, i + chunkSize);
      const neighbors = retrieveFn(chunk);
      chunks.push({ chunk, neighbors });
    }
    return chunks;
  }

  test('chunks sequence correctly', () => {
    const tokens = [1,2,3,4,5,6,7,8];
    const chunks = chunkAndRetrieve(tokens, 4, () => [[9,10,11,12]]);
    assert.equal(chunks.length, 2);
    assert.deepEqual(chunks[0].chunk, [1,2,3,4]);
  });

  test('retrieves neighbors for each chunk', () => {
    const tokens = [1,2,3,4];
    const chunks = chunkAndRetrieve(tokens, 2, (c) => [c.map(x => x + 10)]);
    assert.deepEqual(chunks[0].neighbors[0], [11, 12]);
  });
});
