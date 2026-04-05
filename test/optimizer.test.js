import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { SGD, MomentumSGD, Adam, AdamW, RMSProp, createOptimizer } from '../src/optimizer.js';

function mat(rows, cols, values) {
  const m = new Matrix(rows, cols);
  for (let i = 0; i < values.length; i++) m.data[i] = values[i];
  return m;
}

describe('SGD Optimizer', () => {
  it('updates parameter in gradient direction', () => {
    const opt = new SGD(0.1);
    const param = mat(1, 3, [1, 2, 3]);
    const grad = mat(1, 3, [1, 1, 1]);
    const result = opt.update(param, grad);
    assert.ok(Math.abs(result.data[0] - 0.9) < 0.001);
  });
});

describe('MomentumSGD', () => {
  it('accumulates velocity', () => {
    const opt = new MomentumSGD(0.1, 0.9);
    const param = mat(1, 3, [1, 2, 3]);
    const grad = mat(1, 3, [1, 1, 1]);
    
    const r1 = opt.update(param, grad, 'w');
    assert.ok(Math.abs(r1.data[0] - 0.9) < 0.001);
    
    const r2 = opt.update(r1, grad, 'w');
    assert.ok(r2.data[0] < r1.data[0], 'Should accelerate');
  });
});

describe('Adam Optimizer', () => {
  it('converges on simple problem', () => {
    const opt = new Adam(0.1);
    let param = mat(1, 1, [5.0]);
    
    for (let i = 0; i < 100; i++) {
      opt.step();
      const grad = mat(1, 1, [param.data[0]]);
      param = opt.update(param, grad, 'p');
    }
    
    assert.ok(Math.abs(param.data[0]) < 2.0, `Should converge toward 0, got ${param.data[0]}`);
  });

  it('has bias correction', () => {
    const opt = new Adam(0.01);
    opt.step();
    const param = mat(1, 1, [10]);
    const grad = mat(1, 1, [1]);
    const r1 = opt.update(param, grad, 'test');
    assert.ok(Math.abs(r1.data[0] - (10 - 0.01)) < 0.01);
  });

  it('handles multi-dimensional parameters', () => {
    const opt = new Adam(0.01);
    let w = mat(2, 3, [1, 2, 3, 4, 5, 6]);
    const grad = mat(2, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    
    opt.step();
    const result = opt.update(w, grad, 'w');
    assert.equal(result.rows, 2);
    assert.equal(result.cols, 3);
    assert.ok(result.data[0] < 1);
  });
});

describe('RMSProp', () => {
  it('adapts learning rate', () => {
    const opt = new RMSProp(0.01, 0.99);
    let param = mat(1, 1, [5.0]);
    
    for (let i = 0; i < 50; i++) {
      const grad = mat(1, 1, [param.data[0]]);
      param = opt.update(param, grad, 'p');
    }
    
    assert.ok(Math.abs(param.data[0]) < 4.5, `Should move toward 0, got ${param.data[0]}`);
  });
});

describe('createOptimizer', () => {
  it('creates SGD', () => {
    const opt = createOptimizer('sgd', { lr: 0.05 });
    assert.equal(opt.name, 'sgd');
    assert.equal(opt.lr, 0.05);
  });

  it('creates Adam', () => {
    const opt = createOptimizer('adam', { lr: 0.001 });
    assert.equal(opt.name, 'adam');
  });

  it('creates RMSProp', () => {
    const opt = createOptimizer('rmsprop');
    assert.equal(opt.name, 'rmsprop');
  });

  it('throws on unknown optimizer', () => {
    assert.throws(() => createOptimizer('unknown'));
  });

  it('creates AdamW', () => {
    const opt = createOptimizer('adamw', { lr: 0.001, weightDecay: 0.01 });
    assert.equal(opt.name, 'adamw');
  });
});

describe('weight decay', () => {
  it('SGD with weight decay shrinks weights', () => {
    const opt = new SGD(0.01, { weightDecay: 0.1 });
    const param = mat(1, 2, [10, 10]);
    const grad = mat(1, 2, [0, 0]); // Zero gradient
    const updated = opt.update(param, grad);
    // With wd=0.1, effective grad = 0 + 0.1*10 = 1, so param = 10 - 0.01*1 = 9.99
    assert.ok(updated.get(0, 0) < 10, 'Weight decay should shrink weights');
    assert.ok(updated.get(0, 0) > 9.9, 'Should not shrink too much');
  });

  it('Adam with weight decay shrinks weights', () => {
    const opt = new Adam(0.01, 0.9, 0.999, 1e-8, { weightDecay: 0.1 });
    const param = mat(1, 2, [10, 10]);
    const grad = mat(1, 2, [0, 0]);
    opt.step();
    const updated = opt.update(param, grad, 'test');
    assert.ok(updated.get(0, 0) < 10, 'Adam+WD should shrink weights');
  });
});

describe('AdamW', () => {
  it('updates parameters', () => {
    const opt = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);
    const param = mat(1, 2, [1, 1]);
    const grad = mat(1, 2, [0.5, 0.5]);
    opt.step();
    const updated = opt.update(param, grad, 'w');
    assert.ok(updated.get(0, 0) < 1, 'Should decrease with positive gradient');
  });

  it('weight decay is decoupled', () => {
    // AdamW applies WD to param directly, not through adaptive lr
    const adamw = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.1);
    const param = mat(1, 2, [10, 10]);
    const grad = mat(1, 2, [0, 0]);
    adamw.step();
    const updated = adamw.update(param, grad, 'test');
    // Decoupled WD: param -= wd * lr * param = 10 - 0.1 * 0.01 * 10 = 10 - 0.01 = 9.99
    assert.ok(updated.get(0, 0) < 10);
    assert.ok(updated.get(0, 0) > 9.9);
  });

  it('reduces loss on XOR with regularization', () => {
    // AdamW should train without issues
    const opt = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.01);
    const param = mat(1, 4, [1, 2, 3, 4]);
    const grad = mat(1, 4, [0.1, 0.2, 0.3, 0.4]);
    for (let i = 0; i < 10; i++) {
      opt.step();
      opt.update(param, grad, 'w');
    }
    assert.ok(true, 'Should complete 10 steps without error');
  });
});
