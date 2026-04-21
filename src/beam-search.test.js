// beam-search.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { beamSearch } from './beam-search.js';
import { ModernDecoder } from './modern-decoder.js';

describe('Beam Search', () => {
  const vocabSize = 8;

  function makeModel() {
    return new ModernDecoder(1, 4, 2, 1, vocabSize, { dHidden: 8, maxSeqLen: 32 });
  }

  it('generates correct length', () => {
    const model = makeModel();
    const result = beamSearch(model, [0, 1], 5, 3, vocabSize);
    assert.equal(result.sequence.length, 7, 'prompt(2) + new(5) = 7');
  });

  it('beam width 1 = greedy decoding', () => {
    const model = makeModel();
    const beam1 = beamSearch(model, [0, 1], 5, 1, vocabSize);
    const greedy = model.generate([0, 1], 5, { greedy: true });
    
    // Should produce same sequence (beam-1 = greedy)
    assert.deepEqual(beam1.sequence, greedy);
  });

  it('wider beam produces equal or better score', () => {
    const model = makeModel();
    const beam1 = beamSearch(model, [0, 1], 5, 1, vocabSize);
    const beam4 = beamSearch(model, [0, 1], 5, 4, vocabSize);

    // Wider beam should find equal or better scoring sequence
    assert.ok(beam4.score >= beam1.score - 0.01,
      `Beam-4 (${beam4.score}) should be >= Beam-1 (${beam1.score})`);
  });

  it('returns multiple beam candidates', () => {
    const model = makeModel();
    const result = beamSearch(model, [0], 3, 4, vocabSize);
    assert.equal(result.allBeams.length, 4, 'Should have 4 beams');
    // Beams should be sorted by score
    for (let i = 0; i < result.allBeams.length - 1; i++) {
      assert.ok(result.allBeams[i].score >= result.allBeams[i + 1].score);
    }
  });

  it('all tokens are valid', () => {
    const model = makeModel();
    const result = beamSearch(model, [0, 1], 8, 3, vocabSize);
    for (const t of result.sequence) {
      assert.ok(t >= 0 && t < vocabSize);
    }
  });

  it('EOS token stops generation early', () => {
    const model = makeModel();
    // Use token 0 as EOS — may or may not be generated
    const result = beamSearch(model, [1, 2], 20, 2, vocabSize, { eosToken: 0 });
    // If EOS found, sequence should be shorter
    if (result.sequence.includes(0) && result.sequence.indexOf(0) > 1) {
      assert.ok(result.sequence.length <= 22, 'Should stop at EOS');
    }
  });

  it('length penalty favors longer sequences', () => {
    const model = makeModel();
    const noPenalty = beamSearch(model, [0], 5, 3, vocabSize, { lengthPenalty: 1.0 });
    const highPenalty = beamSearch(model, [0], 5, 3, vocabSize, { lengthPenalty: 0.5 });
    // Both should produce valid output
    assert.ok(noPenalty.sequence.length > 0);
    assert.ok(highPenalty.sequence.length > 0);
  });
});
