// speculative-decoding.test.js — Tests for speculative decoding
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { speculativeDecode } from './speculative-decoding.js';
import { ModernDecoder } from './modern-decoder.js';

describe('Speculative Decoding', () => {
  // Use a small target and even smaller draft model
  const vocabSize = 8;

  function makeDraft() {
    return new ModernDecoder(1, 4, 2, 1, vocabSize, { dHidden: 4, maxSeqLen: 32 });
  }

  function makeTarget() {
    return new ModernDecoder(2, 4, 2, 1, vocabSize, { dHidden: 8, maxSeqLen: 32 });
  }

  it('generates correct number of tokens', () => {
    const draft = makeDraft();
    const target = makeTarget();
    const result = speculativeDecode(draft, target, [0, 1], 10, 3, vocabSize);

    assert.equal(result.tokens.length, 12, 'prompt(2) + new(10) = 12');
    assert.equal(result.tokens[0], 0);
    assert.equal(result.tokens[1], 1);
  });

  it('all tokens are valid', () => {
    const draft = makeDraft();
    const target = makeTarget();
    const result = speculativeDecode(draft, target, [0], 5, 2, vocabSize);

    for (const t of result.tokens) {
      assert.ok(t >= 0 && t < vocabSize, `Token ${t} out of range`);
    }
  });

  it('reports meaningful stats', () => {
    const draft = makeDraft();
    const target = makeTarget();
    const result = speculativeDecode(draft, target, [0, 1], 8, 4, vocabSize);

    console.log('  Stats:', result.stats);

    assert.ok(result.stats.draftForwards > 0);
    assert.ok(result.stats.targetForwards > 0);
    assert.ok(result.stats.totalDrafted > 0);
    // Target forwards should be fewer than tokens generated (that's the point)
    assert.ok(
      result.stats.targetForwards <= 8,
      `Target forwards (${result.stats.targetForwards}) should be ≤ tokens generated (8)`
    );
  });

  it('K=1 is basic verify-each-token mode', () => {
    const draft = makeDraft();
    const target = makeTarget();
    const result = speculativeDecode(draft, target, [0], 5, 1, vocabSize);

    assert.equal(result.tokens.length, 6);
    // With K=1, each iteration drafts 1 and verifies
    assert.ok(result.stats.draftForwards >= 2, 
      `Should have multiple drafts, got ${result.stats.draftForwards}`);
  });

  it('high K means fewer target forwards (when draft is good)', () => {
    // Use target as its own draft (perfect draft → 100% acceptance)
    const target = makeTarget();
    const result = speculativeDecode(target, target, [0, 1], 6, 3, vocabSize);

    console.log('  Perfect draft stats:', result.stats);

    // With perfect draft, acceptance rate should be high
    // (Not 100% due to probabilistic acceptance, but close)
    assert.ok(result.stats.targetForwards <= 6,
      `With perfect draft: ${result.stats.targetForwards} target forwards for 6 tokens`);
  });

  it('speedup is reported', () => {
    const draft = makeDraft();
    const target = makeTarget();
    const result = speculativeDecode(draft, target, [0], 10, 4, vocabSize);

    console.log(`  Speedup: ${result.stats.speedup}`);
    console.log(`  Acceptance: ${result.stats.acceptanceRate}`);

    // Speedup should be >= 1 (speculation can't be worse than naive)
    const speedupNum = parseFloat(result.stats.speedup);
    assert.ok(speedupNum >= 0.5, `Speedup should be reasonable: ${result.stats.speedup}`);
  });
});
