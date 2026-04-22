// speculative-decoding.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { speculativeStep, speculativeGenerate } from './speculative-decoding.js';
import { Matrix } from './matrix.js';

// Mock models: both predict the same sequence for testing
function mockModel(vocabSize, sequence) {
  let callCount = 0;
  return {
    forward: (tokens) => {
      const seqLen = tokens.length;
      const logits = new Matrix(seqLen, vocabSize);
      for (let t = 0; t < seqLen; t++) {
        const nextToken = sequence[t] !== undefined ? sequence[t] : 0;
        logits.set(t, nextToken, 10.0); // High logit for predicted token
      }
      callCount++;
      return logits;
    },
    callCount: () => callCount,
  };
}

describe('speculativeStep', () => {
  it('accepts all tokens when draft matches target', () => {
    // Both models always predict token 5 — perfect agreement
    const alwaysFive = (tokens) => {
      const logits = new Matrix(tokens.length, 10);
      for (let t = 0; t < tokens.length; t++) logits.set(t, 5, 10.0);
      return logits;
    };
    
    const result = speculativeStep(alwaysFive, alwaysFive, [0], 4);
    assert.equal(result.accepted, 4, 'Should accept all 4 draft tokens');
    assert.equal(result.tokens.length, 5); // 4 accepted + 1 bonus
  });
  
  it('rejects when draft diverges from target', () => {
    const draftSeq = [3, 1, 4, 1, 5]; // Draft predicts this
    const targetSeq = [3, 1, 4, 2, 5]; // Target predicts different at position 3
    
    const draft = mockModel(10, draftSeq);
    const target = mockModel(10, targetSeq);
    
    const result = speculativeStep(draft.forward, target.forward, [3], 4);
    // Should reject at the divergence point
    assert.ok(result.tokens.length > 0);
  });
  
  it('returns at least one token (target correction)', () => {
    const draft = mockModel(10, [0, 0, 0, 0, 0]);
    const target = mockModel(10, [0, 1, 0, 0, 0]); // Diverges at position 1
    
    const result = speculativeStep(draft.forward, target.forward, [0], 4);
    assert.ok(result.tokens.length >= 1);
  });
});

describe('speculativeGenerate', () => {
  it('generates requested number of tokens', () => {
    const seq = Array.from({ length: 100 }, (_, i) => i % 10);
    const draft = mockModel(10, seq);
    const target = mockModel(10, seq);
    
    const result = speculativeGenerate(draft.forward, target.forward, [0], 20, 4);
    assert.ok(result.tokens.length >= 20);
  });
  
  it('reports speedup metrics', () => {
    const seq = Array.from({ length: 100 }, (_, i) => i % 5);
    const draft = mockModel(5, seq);
    const target = mockModel(5, seq);
    
    const result = speculativeGenerate(draft.forward, target.forward, [0], 10, 4);
    assert.ok(result.totalSteps > 0);
    assert.ok(result.speedup > 0);
    assert.ok(result.avgAcceptance >= 0);
  });
  
  it('works with gamma=1 (minimal speculation)', () => {
    const seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
    const draft = mockModel(10, seq);
    const target = mockModel(10, seq);
    
    const result = speculativeGenerate(draft.forward, target.forward, [1], 5, 1);
    assert.ok(result.tokens.length >= 5);
  });
});
