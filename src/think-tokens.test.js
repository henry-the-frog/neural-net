import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { extractThinking, thinkingBudget } from './think-tokens.js';

describe('Think Tokens', () => {
  test('extracts thinking and output', () => {
    const tokens = [1, 2, 100, 3, 4, 101, 5, 6]; // 100=start, 101=end
    const { thinking, output } = extractThinking(tokens, 100, 101);
    assert.deepEqual(thinking, [3, 4]);
    assert.deepEqual(output, [1, 2, 5, 6]);
  });

  test('no thinking tokens → all output', () => {
    const { thinking, output } = extractThinking([1, 2, 3], 100, 101);
    assert.equal(thinking.length, 0);
    assert.deepEqual(output, [1, 2, 3]);
  });

  test('thinking budget computes ratios', () => {
    const budget = thinkingBudget(100, 50);
    assert.ok(Math.abs(budget.thinkRatio - 2/3) < 0.01);
    assert.equal(budget.overhead, 2);
  });
});
