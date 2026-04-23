import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { healedPrompt } from './token-healing.js';

describe('Token Healing', () => {
  const simpleTokenizer = (text) => text.split(/(\s+)/).filter(Boolean);

  test('healed prompt removes last partial token', () => {
    const result = healedPrompt('Hello wor', simpleTokenizer);
    assert.equal(result.text, 'Hello ');
    assert.equal(result.constrainPrefix, 'wor');
  });

  test('backtrack length matches removed token', () => {
    const result = healedPrompt('test ing', simpleTokenizer);
    assert.equal(result.backtrack, result.constrainPrefix.length);
  });
});
