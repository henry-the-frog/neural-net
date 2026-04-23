import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Tokenizer Alignment', () => {
  function alignTokensToChars(tokens, text) {
    const alignments = [];
    let pos = 0;
    for (const token of tokens) {
      const start = pos;
      const end = pos + token.length;
      alignments.push({ token, start, end });
      pos = end;
    }
    return alignments;
  }

  test('alignment covers full text', () => {
    const tokens = ['Hello', ' ', 'world'];
    const alignments = alignTokensToChars(tokens, 'Hello world');
    assert.equal(alignments[0].start, 0);
    assert.equal(alignments[2].end, 11);
  });

  test('no gaps in alignment', () => {
    const tokens = ['ab', 'cd', 'ef'];
    const a = alignTokensToChars(tokens, 'abcdef');
    for (let i = 1; i < a.length; i++) {
      assert.equal(a[i].start, a[i-1].end);
    }
  });
});
