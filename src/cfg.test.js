// cfg.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { classifierFreeGuidance, conditionalDropout, dynamicGuidanceScale, rescaledCFG } from './cfg.js';

describe('Classifier-Free Guidance', () => {
  test('guidance scale 1.0 returns conditional prediction', () => {
    const cond = new Float64Array([1, 2, 3]);
    const uncond = new Float64Array([0, 0, 0]);
    const guided = classifierFreeGuidance(cond, uncond, 1.0);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(guided[i] - cond[i]) < 1e-10);
    }
  });

  test('guidance scale 0 returns unconditional prediction', () => {
    const cond = new Float64Array([1, 2, 3]);
    const uncond = new Float64Array([5, 5, 5]);
    const guided = classifierFreeGuidance(cond, uncond, 0);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(guided[i] - uncond[i]) < 1e-10);
    }
  });

  test('higher guidance amplifies difference', () => {
    const cond = new Float64Array([2, 2]);
    const uncond = new Float64Array([1, 1]);
    const g3 = classifierFreeGuidance(cond, uncond, 3);
    const g7 = classifierFreeGuidance(cond, uncond, 7);
    
    // g = uncond + w*(cond-uncond) = 1 + w*1 = 1+w
    assert.ok(Math.abs(g3[0] - 4) < 1e-10); // 1 + 3*1
    assert.ok(Math.abs(g7[0] - 8) < 1e-10); // 1 + 7*1
  });

  test('conditional dropout drops at expected rate', () => {
    let dropped = 0;
    const N = 1000;
    for (let i = 0; i < N; i++) {
      const { isDropped } = conditionalDropout('class_5', 0.1);
      if (isDropped) dropped++;
    }
    // Should be ~100 ± 30
    assert.ok(dropped > 50 && dropped < 200, `Drop rate should be ~10%, got ${dropped/N*100}%`);
  });

  test('dynamic guidance scale - constant returns base', () => {
    assert.equal(dynamicGuidanceScale(7.5, 500, 1000, 'constant'), 7.5);
  });

  test('dynamic guidance scale - linear decreases with progress', () => {
    const early = dynamicGuidanceScale(7.5, 100, 1000, 'linear');
    const late = dynamicGuidanceScale(7.5, 900, 1000, 'linear');
    assert.ok(early > late, 'Early steps should have higher guidance');
  });

  test('rescaledCFG has similar std to conditional', () => {
    const cond = new Float64Array([1, 2, 3, 4, 5]);
    const uncond = new Float64Array([0, 0, 0, 0, 0]);
    const rescaled = rescaledCFG(cond, uncond, 10, 1.0);
    
    // Compute std of conditional and rescaled
    const condStd = Math.sqrt(cond.reduce((s, v) => s + (v - 3) ** 2, 0) / 5);
    const mean = rescaled.reduce((a, b) => a + b) / 5;
    const rStd = Math.sqrt(rescaled.reduce((s, v) => s + (v - mean) ** 2, 0) / 5);
    
    assert.ok(Math.abs(condStd - rStd) < condStd * 0.3, 
      `Rescaled std ${rStd} should be close to conditional std ${condStd}`);
  });
});
