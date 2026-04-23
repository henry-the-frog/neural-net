// beam-search.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { beamSearch } from './beam-search.js';

describe('Beam Search', () => {
  const vocabSize = 8;
  
  // Simple model: always prefers token (previous + 1) % vocabSize
  function mockForward(tokens) {
    const last = tokens[tokens.length - 1];
    const logits = new Float64Array(vocabSize).fill(-5);
    logits[(last + 1) % vocabSize] = 5; // Strong preference for next token
    logits[(last + 2) % vocabSize] = 2; // Secondary preference
    return [logits]; // Array of logit arrays (one per position)
  }

  test('returns beamWidth results', () => {
    const results = beamSearch(mockForward, [0], 5, 3);
    assert.ok(results.length >= 1);
    assert.ok(results.length <= 3);
  });

  test('best beam follows strong preference', () => {
    const results = beamSearch(mockForward, [0], 5, 2);
    // Best beam should follow the strong preference: 0 → 1 → 2 → 3 → 4 → 5
    const best = results[0].tokens;
    for (let i = 1; i < best.length; i++) {
      assert.equal(best[i], (best[i-1] + 1) % vocabSize,
        `Expected ${(best[i-1]+1)%vocabSize} at position ${i}, got ${best[i]}`);
    }
  });

  test('beams are sorted by score', () => {
    const results = beamSearch(mockForward, [0], 5, 4);
    for (let i = 1; i < results.length; i++) {
      assert.ok(results[i].score <= results[i-1].score,
        `Beams should be sorted by score: ${results[i].score} > ${results[i-1].score}`);
    }
  });

  test('EOS token terminates beam', () => {
    function eosForward(tokens) {
      const logits = new Float64Array(vocabSize).fill(-5);
      if (tokens.length >= 3) {
        logits[7] = 10; // EOS token = 7
      } else {
        logits[1] = 5;
      }
      return [logits];
    }
    
    const results = beamSearch(eosForward, [0], 10, 2, 7);
    // Should have short sequences terminated by EOS
    assert.ok(results[0].tokens.length <= 5);
  });

  test('beamWidth=1 is greedy decoding', () => {
    const results = beamSearch(mockForward, [0], 5, 1);
    assert.equal(results.length, 1);
    const best = results[0].tokens;
    // Greedy: always pick highest logit
    for (let i = 1; i < best.length; i++) {
      assert.equal(best[i], (best[i-1] + 1) % vocabSize);
    }
  });
});
