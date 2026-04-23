import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { constitutionalCritique, revisionPrompt, rankByPrinciples } from './constitutional-ai.js';

describe('Constitutional AI', () => {
  test('critique prompt includes principle and response', () => {
    const prompt = constitutionalCritique('Hello world', 'Be helpful');
    assert.ok(prompt.includes('Be helpful'));
    assert.ok(prompt.includes('Hello world'));
  });

  test('revision prompt includes original and feedback', () => {
    const prompt = revisionPrompt('Bad response', 'Too vague');
    assert.ok(prompt.includes('Bad response'));
    assert.ok(prompt.includes('Too vague'));
  });

  test('rankByPrinciples sorts by score', () => {
    const responses = ['good', 'bad'];
    const principles = ['helpful'];
    const scoreFn = (r, p) => r === 'good' ? 1 : 0;
    const ranked = rankByPrinciples(responses, principles, scoreFn);
    assert.equal(ranked[0].response, 'good');
  });
});
