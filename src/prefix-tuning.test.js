import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { PrefixTuning } from './prefix-tuning.js';
import { Matrix } from './matrix.js';

describe('Prefix Tuning', () => {
  test('prepend increases sequence length', () => {
    const pt = new PrefixTuning(5, 8, 2);
    const x = Matrix.random(10, 8);
    const combined = pt.prependToInput(x, 0);
    assert.equal(combined.rows, 15); // 5 prefix + 10 input
    assert.equal(combined.cols, 8);
  });

  test('paramCount is nLayers * prefixLen * dModel', () => {
    const pt = new PrefixTuning(10, 64, 12);
    assert.equal(pt.paramCount(), 12 * 10 * 64);
  });

  test('prefix params are much less than model params', () => {
    const pt = new PrefixTuning(20, 768, 12);
    // GPT-2 small has ~124M params, prefix has:
    const prefixParams = pt.paramCount();
    assert.ok(prefixParams < 200000, `Prefix params should be small: ${prefixParams}`);
  });
});
