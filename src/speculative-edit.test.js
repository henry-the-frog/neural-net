import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Speculative Editing', () => {
  // Model generates, then verifier accepts/rejects tokens
  function speculativeEdit(draft, verify) {
    const accepted = [];
    for (const token of draft) {
      if (verify(token, accepted)) accepted.push(token);
      else break;
    }
    return accepted;
  }

  test('accepts all valid tokens', () => {
    const draft = [1, 2, 3];
    const accepted = speculativeEdit(draft, () => true);
    assert.deepEqual(accepted, [1, 2, 3]);
  });

  test('stops at first rejected token', () => {
    const accepted = speculativeEdit([1, 2, 3], (t) => t < 3);
    assert.deepEqual(accepted, [1, 2]);
  });
});
