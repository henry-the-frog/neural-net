import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { transformerParams, formatParams } from './parameter-count.js';

describe('Parameter Count', () => {
  test('GPT-2 small has ~124M params', () => {
    const params = transformerParams({
      vocabSize: 50257, dModel: 768, nHeads: 12, nLayers: 12, dFF: 3072
    });
    assert.ok(params > 100e6 && params < 200e6, `Expected ~124-162M, got ${formatParams(params)}`);
  });

  test('weight tying saves vocabSize * dModel', () => {
    const untied = transformerParams({ vocabSize: 1000, dModel: 64, nHeads: 4, nLayers: 2, dFF: 128 });
    const tied = transformerParams({ vocabSize: 1000, dModel: 64, nHeads: 4, nLayers: 2, dFF: 128, tiedWeights: true });
    assert.equal(untied - tied, 1000 * 64);
  });

  test('formatParams formats correctly', () => {
    assert.equal(formatParams(1500000000), '1.5B');
    assert.equal(formatParams(124000000), '124.0M');
    assert.equal(formatParams(50000), '50.0K');
  });
});
