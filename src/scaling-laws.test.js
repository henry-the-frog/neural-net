import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { chinchillaOptimal, computeForTraining, isOverTrained } from './scaling-laws.js';

describe('Scaling Laws', () => {
  test('chinchilla optimal: more compute → bigger model', () => {
    const small = chinchillaOptimal(1e15);
    const large = chinchillaOptimal(1e18);
    assert.ok(large.params > small.params);
    assert.ok(large.tokens > small.tokens);
  });

  test('compute for training: 6ND', () => {
    assert.equal(computeForTraining(1e9, 1e12), 6e21);
  });

  test('LLaMA-2 70B is over-trained (2T tokens / 70B params = ~29)', () => {
    assert.ok(isOverTrained(70e9, 2e12));
  });
});
