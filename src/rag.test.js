import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { retrieveTopK, buildRAGPrompt, chunkText } from './rag.js';

describe('RAG', () => {
  test('retrieve returns topK docs', () => {
    const docs = ['cat sat', 'dog ran', 'bird flew'];
    const scoreFn = (q, d) => d.includes('cat') ? 1 : 0;
    const results = retrieveTopK('cat', docs, scoreFn, 1);
    assert.equal(results[0].doc, 'cat sat');
  });

  test('buildRAGPrompt includes context and query', () => {
    const prompt = buildRAGPrompt('What?', [{ doc: 'Info here' }]);
    assert.ok(prompt.includes('Info here'));
    assert.ok(prompt.includes('What?'));
  });

  test('chunkText splits with overlap', () => {
    const text = 'a'.repeat(100);
    const chunks = chunkText(text, 40, 10);
    assert.ok(chunks.length >= 3);
    assert.equal(chunks[0].length, 40);
  });
});
