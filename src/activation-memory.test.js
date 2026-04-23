import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Activation Memory', () => {
  function estimateActivationMemory(batchSize, seqLen, dModel, nLayers, dtype = 'fp16') {
    const bytesPerElement = dtype === 'fp32' ? 4 : 2;
    // Per layer: attention scores + QKV + FFN hidden
    const perLayer = batchSize * seqLen * dModel * 4 * bytesPerElement; // rough
    return perLayer * nLayers;
  }

  test('memory scales with batch size', () => {
    const m1 = estimateActivationMemory(1, 1024, 768, 12);
    const m2 = estimateActivationMemory(2, 1024, 768, 12);
    assert.equal(m2, m1 * 2);
  });

  test('fp16 uses half the memory of fp32', () => {
    const fp16 = estimateActivationMemory(1, 1024, 768, 12, 'fp16');
    const fp32 = estimateActivationMemory(1, 1024, 768, 12, 'fp32');
    assert.equal(fp16, fp32 / 2);
  });
});
