// token-healing.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { healTokenBoundary, getTokensWithPrefix } from './token-healing.js';
import { BPETokenizer } from './bpe.js';

describe('Token Healing', () => {
  function makeTok() {
    const tok = new BPETokenizer();
    tok.train('hello world hello world goodbye world', 30, ['<|eos|>']);
    return tok;
  }

  it('no healing needed for complete tokens', () => {
    const tok = makeTok();
    const tokens = tok.encode('hello world');
    const result = healTokenBoundary(tok, tokens);
    // No constraint prefix (last token is complete)
    assert.equal(result.constraintPrefix.length, 0);
    assert.deepEqual(result.healedPrompt, tokens);
  });

  it('getTokensWithPrefix finds matching tokens', () => {
    const tok = makeTok();
    const matches = getTokensWithPrefix(tok, 'h');
    assert.ok(matches.length > 0, 'Should find tokens starting with h');
    for (const id of matches) {
      assert.ok(tok.idToToken[id].startsWith('h'));
    }
  });

  it('empty prompt returns empty result', () => {
    const tok = makeTok();
    const result = healTokenBoundary(tok, []);
    assert.deepEqual(result.healedPrompt, []);
    assert.equal(result.constraintPrefix, '');
  });

  it('getTokensWithPrefix returns empty for no match', () => {
    const tok = makeTok();
    const matches = getTokensWithPrefix(tok, 'zzzzz');
    assert.equal(matches.length, 0);
  });
});
