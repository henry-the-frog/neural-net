import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { linearRoPEScaling, yarnScaling } from './rope-scaling.js';

describe('RoPE Scaling', () => {
  test('linear scaling reduces frequency', () => {
    const orig = 1.0;
    const scaled = linearRoPEScaling(orig, 4);
    assert.equal(scaled, 0.25);
  });

  test('YaRN: high freq unchanged', () => {
    const highFreq = 100; // Short wavelength
    const scaled = yarnScaling(highFreq, 64, 4);
    assert.equal(scaled, highFreq);
  });

  test('YaRN: low freq fully scaled', () => {
    const lowFreq = 0.001; // Very long wavelength
    const scaled = yarnScaling(lowFreq, 64, 4);
    assert.ok(Math.abs(scaled - lowFreq / 4) < 0.001);
  });
});
