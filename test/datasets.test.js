// datasets.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Datasets } from '../src/datasets.js';

describe('Datasets', () => {
  it('xor: 4 points, correct labels', () => {
    const { inputs, targets } = Datasets.xor();
    assert.equal(inputs.rows, 4);
    assert.equal(targets.get(0, 0), 0); // 0 XOR 0 = 0
    assert.equal(targets.get(1, 0), 1); // 0 XOR 1 = 1
  });

  it('spiral: correct shape', () => {
    const { inputs, targets } = Datasets.spiral(50, 3);
    assert.equal(inputs.rows, 150);
    assert.equal(inputs.cols, 2);
    assert.equal(targets.cols, 3); // one-hot
  });

  it('moons: correct shape and classes', () => {
    const { inputs, targets } = Datasets.moons(100);
    assert.equal(inputs.rows, 100);
    assert.equal(targets.cols, 1);
    const classes = new Set();
    for (let i = 0; i < 100; i++) classes.add(targets.get(i, 0));
    assert.equal(classes.size, 2);
  });

  it('circles: inner and outer', () => {
    const { inputs, targets } = Datasets.circles(200);
    assert.equal(inputs.rows, 200);
    let class0 = 0, class1 = 0;
    for (let i = 0; i < 200; i++) {
      if (targets.get(i, 0) === 0) class0++;
      else class1++;
    }
    assert.equal(class0, 100);
    assert.equal(class1, 100);
  });

  it('blobs: correct clusters', () => {
    const { inputs, targets } = Datasets.blobs(150, 3);
    assert.equal(inputs.rows, 150);
    assert.equal(targets.cols, 3);
  });

  it('sine: regression data', () => {
    const { inputs, targets } = Datasets.sine(50);
    assert.equal(inputs.rows, 50);
    // First point should be sin(-pi) ≈ 0
    assert.ok(Math.abs(targets.get(0, 0)) < 0.1);
  });

  it('linear: regression data', () => {
    const { inputs, targets } = Datasets.linear(50, 0);
    assert.equal(inputs.rows, 50);
    // At x=0 (center point), y ≈ 1 (2*0 + 1)
    const midIdx = 25;
    assert.ok(Math.abs(targets.get(midIdx, 0) - 1) < 0.1);
  });

  it('all datasets produce finite values', () => {
    for (const [name, gen] of Object.entries(Datasets)) {
      const { inputs, targets } = gen();
      assert.ok(inputs.data.every(Number.isFinite), `${name} inputs should be finite`);
      assert.ok(targets.data.every(Number.isFinite), `${name} targets should be finite`);
    }
  });
});
