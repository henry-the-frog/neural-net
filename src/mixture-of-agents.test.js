import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Mixture of Agents (Wang et al., 2024) — aggregate multiple model outputs
describe('Mixture of Agents', () => {
  function aggregateResponses(responses, scoreFn) {
    return responses.map((r, i) => ({ response: r, score: scoreFn(r) }))
      .sort((a, b) => b.score - a.score)[0].response;
  }

  test('selects highest scored response', () => {
    const best = aggregateResponses(
      ['short', 'this is a longer response', 'medium one'],
      r => r.length
    );
    assert.equal(best, 'this is a longer response');
  });

  test('handles single response', () => {
    const result = aggregateResponses(['only'], r => 1);
    assert.equal(result, 'only');
  });
});
