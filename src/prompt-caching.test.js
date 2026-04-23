import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Prompt Caching', () => {
  const cache = new Map();
  
  function hashPrompt(tokens) {
    return tokens.join(',');
  }
  
  function getCached(tokens) {
    const key = hashPrompt(tokens);
    return cache.get(key) || null;
  }
  
  function setCached(tokens, kvState) {
    cache.set(hashPrompt(tokens), kvState);
  }

  test('cache hit returns state', () => {
    setCached([1,2,3], { keys: [1], values: [2] });
    const result = getCached([1,2,3]);
    assert.deepEqual(result, { keys: [1], values: [2] });
  });

  test('cache miss returns null', () => {
    assert.equal(getCached([9,9,9]), null);
  });
});
