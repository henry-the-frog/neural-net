// ema.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { EMA, PolyakAveraging } from './ema.js';

describe('EMA', () => {
  test('initial EMA equals initial params', () => {
    const params = new Float64Array([1, 2, 3]);
    const ema = new EMA(params, 0.999);
    const shadow = ema.get();
    for (let i = 0; i < 3; i++) {
      assert.equal(shadow[i], params[i]);
    }
  });

  test('EMA tracks slowly changing params', () => {
    const ema = new EMA(new Float64Array([0, 0, 0]), 0.99);
    
    // Push params toward [1, 1, 1]
    for (let step = 0; step < 100; step++) {
      ema.update(new Float64Array([1, 1, 1]));
    }
    
    const shadow = ema.get();
    // After 100 steps with decay 0.99, should be close to [1, 1, 1]
    for (let i = 0; i < 3; i++) {
      assert.ok(shadow[i] > 0.5, `EMA should trend toward target: ${shadow[i]}`);
    }
  });

  test('higher decay = slower tracking', () => {
    const fast = new EMA(new Float64Array([0]), 0.9);
    const slow = new EMA(new Float64Array([0]), 0.9999);
    
    // Need enough steps for warmup to not dominate, but not so many that both converge
    for (let i = 0; i < 20; i++) {
      fast.update(new Float64Array([1]));
      slow.update(new Float64Array([1]));
    }
    
    // With 20 steps: warmup effective decay = min(decay, (1+t)/(10+t))
    // At t=20: warmup = 21/30 = 0.7, so fast uses 0.7 and slow uses 0.7 too (< 0.9999)
    // Need to test where warmup is saturated
    for (let i = 0; i < 100; i++) {
      fast.update(new Float64Array([1]));
      slow.update(new Float64Array([1]));
    }
    
    assert.ok(fast.get()[0] >= slow.get()[0], 
      `Fast ${fast.get()[0]} should be >= slow ${slow.get()[0]}`);
  });

  test('warmup reduces effective decay for early steps', () => {
    const ema = new EMA(new Float64Array([0]), 0.999);
    ema.update(new Float64Array([10]));
    // After 1 step, warmup: actual_decay = min(0.999, 2/11) ≈ 0.182
    // shadow ≈ 0.182 * 0 + 0.818 * 10 ≈ 8.18
    assert.ok(ema.get()[0] > 5, `Warmup should use lower decay: ${ema.get()[0]}`);
  });

  test('apply copies EMA to target', () => {
    const ema = new EMA(new Float64Array([5, 10, 15]), 0.99);
    const target = new Float64Array(3);
    ema.apply(target);
    assert.equal(target[0], 5);
    assert.equal(target[1], 10);
  });

  test('Polyak averaging is simple mean', () => {
    const polyak = new PolyakAveraging(new Float64Array([0]));
    polyak.update(new Float64Array([10]));
    polyak.update(new Float64Array([20]));
    // Mean of [0, 10, 20] = 10
    assert.equal(polyak.get()[0], 10);
  });
});
