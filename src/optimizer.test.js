// optimizer.test.js — Tests for ScheduledOptimizer and SGD
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ScheduledOptimizer, SGD } from './optimizer.js';
import { AdamW } from './adamw.js';
import { WarmupScheduler, CosineDecay, ConstantLR } from './lr-scheduler.js';

describe('SGD', () => {
  it('updates parameters in correct direction', () => {
    const sgd = new SGD({ lr: 0.1 });
    const params = new Float64Array([1.0, 2.0, 3.0]);
    const grads = new Float64Array([0.5, -0.5, 0.0]);
    sgd.update('p', params, grads);
    assert.ok(Math.abs(params[0] - 0.95) < 1e-10);
    assert.ok(Math.abs(params[1] - 2.05) < 1e-10);
    assert.ok(Math.abs(params[2] - 3.0) < 1e-10);
  });
  
  it('momentum accumulates velocity', () => {
    const sgd = new SGD({ lr: 0.1, momentum: 0.9 });
    const params = new Float64Array([0.0]);
    const grads = new Float64Array([1.0]);
    
    sgd.update('p', params, grads);
    const afterFirst = params[0]; // -0.1 * 1.0 = -0.1
    assert.ok(Math.abs(afterFirst - (-0.1)) < 1e-10);
    
    sgd.update('p', params, grads);
    // v = 0.9 * 1.0 + 1.0 = 1.9
    // params = -0.1 - 0.1 * 1.9 = -0.29
    assert.ok(Math.abs(params[0] - (-0.29)) < 1e-10);
  });
  
  it('respects lr override', () => {
    const sgd = new SGD({ lr: 0.1 });
    const params = new Float64Array([1.0]);
    const grads = new Float64Array([1.0]);
    sgd.update('p', params, grads, 0.5);
    assert.ok(Math.abs(params[0] - 0.5) < 1e-10);
  });
});

describe('ScheduledOptimizer', () => {
  it('passes scheduler LR to optimizer', () => {
    const sgd = new SGD();
    const sched = new ConstantLR(0.05);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    const params = new Float64Array([1.0]);
    const grads = new Float64Array([1.0]);
    opt.update('p', params, grads);
    assert.ok(Math.abs(params[0] - 0.95) < 1e-10); // 1 - 0.05 * 1
  });
  
  it('schedulerStep advances LR', () => {
    const sgd = new SGD();
    const sched = new CosineDecay(0.1, 100);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    const lr0 = opt.getLR();
    opt.schedulerStep();
    opt.schedulerStep();
    const lr2 = opt.getLR();
    assert.ok(lr2 < lr0, 'LR should decrease after steps');
  });
  
  it('works with AdamW', () => {
    const adamw = new AdamW({ weightDecay: 0.01 });
    const sched = new WarmupScheduler(new CosineDecay(0.001, 100), 10);
    const opt = new ScheduledOptimizer(adamw, sched);
    
    const params = new Float64Array([1.0, 2.0]);
    const grads = new Float64Array([0.1, 0.2]);
    
    // Should not throw
    for (let i = 0; i < 20; i++) {
      opt.schedulerStep();
      opt.update('test', params, grads);
    }
    
    // Params should have changed
    assert.notEqual(params[0], 1.0);
    assert.notEqual(params[1], 2.0);
    assert.ok(opt.getLR() > 0);
  });
  
  it('reset clears state', () => {
    const sgd = new SGD({ momentum: 0.9 });
    const sched = new CosineDecay(0.1, 100);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    const params = new Float64Array([0.0]);
    const grads = new Float64Array([1.0]);
    
    for (let i = 0; i < 10; i++) {
      opt.schedulerStep();
      opt.update('p', params, grads);
    }
    
    opt.reset();
    assert.equal(opt.getStep(), 0);
    assert.ok(Math.abs(opt.getLR() - 0.1) < 1e-10);
  });
  
  it('getStep tracks scheduler steps', () => {
    const opt = new ScheduledOptimizer(new SGD(), new ConstantLR(0.1));
    assert.equal(opt.getStep(), 0);
    opt.schedulerStep();
    opt.schedulerStep();
    assert.equal(opt.getStep(), 2);
  });
});
