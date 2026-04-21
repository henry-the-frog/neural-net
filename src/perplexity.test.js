// perplexity.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computePerplexity, corpusPerplexity, compareModels, theoreticalBaselines } from './perplexity.js';
import { ModernDecoder } from './modern-decoder.js';

describe('Perplexity', () => {
  const vocabSize = 8;

  function makeModel() {
    return new ModernDecoder(1, 4, 2, 1, vocabSize, { dHidden: 8, maxSeqLen: 32 });
  }

  it('random model has PPL close to vocab size', () => {
    const model = makeModel();
    const seq = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
    const result = computePerplexity(model, seq, vocabSize);

    console.log(`  PPL: ${result.perplexity.toFixed(2)} (expected ~${vocabSize} for random)`);
    // Random model: PPL should be roughly V (±factor of 2-3 due to random weights)
    assert.ok(result.perplexity > 1, 'PPL should be > 1');
    assert.ok(result.perplexity < vocabSize * 5, `PPL should be reasonable: ${result.perplexity}`);
  });

  it('log probability is negative', () => {
    const model = makeModel();
    const result = computePerplexity(model, [0, 1, 2, 3], vocabSize);
    assert.ok(result.avgLogProb < 0, 'Avg log prob should be negative');
  });

  it('shorter sequences give valid results', () => {
    const model = makeModel();
    const result = computePerplexity(model, [0, 1], vocabSize);
    assert.equal(result.numTokens, 1);
    assert.ok(isFinite(result.perplexity));
  });

  it('single token gives infinity', () => {
    const model = makeModel();
    const result = computePerplexity(model, [0], vocabSize);
    assert.equal(result.perplexity, Infinity);
  });

  it('corpus perplexity aggregates correctly', () => {
    const model = makeModel();
    const sequences = [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [0, 2, 4, 6],
    ];
    const result = corpusPerplexity(model, sequences, vocabSize);
    assert.equal(result.numSequences, 3);
    assert.ok(result.totalTokens > 0);
    assert.ok(isFinite(result.perplexity));
  });

  it('model comparison works', () => {
    const m1 = makeModel();
    const m2 = makeModel(); // different random weights
    const sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 0, 1]];

    const result = compareModels(m1, m2, sequences, vocabSize);
    assert.ok(['model1', 'model2'].includes(result.winner));
    console.log(`  Model1 PPL: ${result.model1.perplexity.toFixed(2)}, Model2 PPL: ${result.model2.perplexity.toFixed(2)}`);
    console.log(`  Winner: ${result.winner} (${result.improvement} improvement)`);
  });

  it('theoretical baselines are correct', () => {
    const baselines = theoreticalBaselines(vocabSize);
    assert.equal(baselines.random, vocabSize);
    assert.equal(baselines.perfect, 1.0);
    assert.ok(baselines.unigram < baselines.random);
  });
});
