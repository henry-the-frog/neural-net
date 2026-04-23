import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Inference Engine', () => {
  function estimateLatency(seqLen, dModel, nLayers, nHeads, flopsPerSecond = 1e12) {
    // Approximate FLOPs per token: 2 * nLayers * (4*dModel² + 2*dModel*seqLen)
    const flopsPerToken = 2 * nLayers * (4 * dModel * dModel + 2 * dModel * seqLen);
    return flopsPerToken / flopsPerSecond * 1000; // milliseconds
  }

  test('longer sequence = more latency', () => {
    const short = estimateLatency(100, 768, 12, 12);
    const long = estimateLatency(4096, 768, 12, 12);
    assert.ok(long > short);
  });

  test('latency in reasonable range', () => {
    const ms = estimateLatency(512, 768, 12, 12, 1e12);
    assert.ok(ms > 0 && ms < 100);
  });
});
