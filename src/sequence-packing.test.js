import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { packSequences, packingEfficiency } from './sequence-packing.js';

describe('Sequence Packing', () => {
  test('packs short sequences together', () => {
    const seqs = [[1,2], [3,4], [5,6]];
    const packed = packSequences(seqs, 10);
    assert.ok(packed.length < 3, 'Should pack into fewer bins');
  });

  test('long sequence gets its own bin', () => {
    const seqs = [[1,2,3,4,5,6,7,8]];
    const packed = packSequences(seqs, 10);
    assert.equal(packed.length, 1);
  });

  test('packing efficiency > naive padding', () => {
    const seqs = [[1,2], [3], [4,5,6], [7]];
    const eff = packingEfficiency(seqs, 10);
    assert.ok(eff > 0.3, `Efficiency should be decent: ${eff}`);
  });
});
