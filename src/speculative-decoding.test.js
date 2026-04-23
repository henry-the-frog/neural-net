// speculative-decoding.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { speculativeDecodeStep } from './speculative-decoding.js';

describe('Speculative Decoding', () => {
  // Simple mock models: draft predicts uniform, target has strong preferences
  const vocabSize = 10;
  
  function mockDraftForward(tokens) {
    // Uniform distribution (weak model)
    return new Float64Array(vocabSize).fill(0);
  }
  
  function mockTargetForward(tokens) {
    // Returns logits for each position
    const result = [];
    for (let i = 0; i < tokens.length; i++) {
      const logits = new Float64Array(vocabSize).fill(0);
      logits[tokens[i] % vocabSize] = 5; // Prefer repeating previous token
      result.push(logits);
    }
    return result;
  }

  test('returns at least 1 token', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 4);
    assert.ok(result.tokens.length >= 1);
  });

  test('returns at most K+1 tokens', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 4);
    assert.ok(result.tokens.length <= 5); // K + 1
  });

  test('acceptance rate is between 0 and 1', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 4);
    assert.ok(result.acceptanceRate >= 0);
    assert.ok(result.acceptanceRate <= 1);
  });

  test('tokens are valid indices', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 4);
    for (const t of result.tokens) {
      assert.ok(t >= 0 && t < vocabSize, `Token ${t} should be in [0, ${vocabSize})`);
    }
  });

  test('accepted count matches reported', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 4);
    assert.ok(result.accepted >= 0);
    assert.ok(result.accepted <= result.total);
  });

  test('K=1 produces 1-2 tokens', () => {
    const result = speculativeDecodeStep(mockDraftForward, mockTargetForward, [1, 2, 3], 1);
    assert.ok(result.tokens.length >= 1 && result.tokens.length <= 2);
  });

  test('identical draft and target → high acceptance', () => {
    // When draft = target, acceptance should be 100%
    function identicalModel(tokens) { return new Float64Array(vocabSize).fill(1); }
    function identicalTarget(tokens) {
      return tokens.map(() => new Float64Array(vocabSize).fill(1));
    }
    
    let totalAccepted = 0, totalK = 0;
    for (let trial = 0; trial < 10; trial++) {
      const result = speculativeDecodeStep(identicalModel, identicalTarget, [0], 4);
      totalAccepted += result.accepted;
      totalK += result.total;
    }
    // With identical models, acceptance should be very high
    assert.ok(totalAccepted / totalK > 0.8, 
      `Identical models acceptance ${totalAccepted}/${totalK} should be >80%`);
  });
});
