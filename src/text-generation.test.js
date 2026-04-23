import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { greedyDecode, repetitionCheck } from './text-generation.js';

describe('Text Generation', () => {
  test('greedy decode produces tokens', () => {
    const logitsFn = () => [0, 5, 1]; // Always predicts token 1
    const tokens = greedyDecode(logitsFn, [0], 3);
    assert.equal(tokens.length, 4); // 1 prompt + 3 generated
  });

  test('repetition check detects loops', () => {
    const tokens = [1, 2, 3, 1, 2, 3];
    assert.ok(repetitionCheck(tokens, 3));
  });

  test('no repetition for varied tokens', () => {
    assert.ok(!repetitionCheck([1, 2, 3, 4, 5, 6], 3));
  });
});
