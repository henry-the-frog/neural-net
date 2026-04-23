import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Reinforcement Learning from AI Feedback', () => {
  function rlaifScore(response, criteria) {
    return criteria.reduce((score, c) => score + (c.check(response) ? c.weight : 0), 0);
  }

  test('scores based on criteria', () => {
    const criteria = [
      { check: r => r.length > 5, weight: 1 },
      { check: r => !r.includes('bad'), weight: 2 },
    ];
    assert.equal(rlaifScore('good response', criteria), 3);
    assert.equal(rlaifScore('bad', criteria), 0);
  });
});
