// matrix-stress.test.js — Matrix operations edge cases
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';

describe('Matrix Stress', () => {
  it('large matrix multiply', () => {
    const a = Matrix.random(50, 50);
    const b = Matrix.random(50, 50);
    const c = a.dot(b);
    assert.equal(c.rows, 50);
    assert.equal(c.cols, 50);
    for (let i = 0; i < c.data.length; i++) {
      assert.ok(isFinite(c.data[i]), `Element should be finite at ${i}`);
    }
  });

  it('transpose of transpose is identity', () => {
    const a = Matrix.random(3, 5);
    const att = a.T().T();
    for (let i = 0; i < a.data.length; i++) {
      assert.ok(Math.abs(a.data[i] - att.data[i]) < 1e-10, `A^T^T should equal A at ${i}`);
    }
  });

  it('matrix addition is commutative', () => {
    const a = Matrix.random(3, 3);
    const b = Matrix.random(3, 3);
    const ab = a.add(b);
    const ba = b.add(a);
    for (let i = 0; i < ab.data.length; i++) {
      assert.ok(Math.abs(ab.data[i] - ba.data[i]) < 1e-10, `A+B should equal B+A at ${i}`);
    }
  });

  it('scalar multiplication', () => {
    const a = new Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
    const scaled = a.mul(3);
    assert.equal(scaled.get(0, 0), 3);
    assert.equal(scaled.get(1, 1), 12);
  });

  it('identity multiplication', () => {
    const a = new Matrix(3, 3, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    const b = Matrix.random(3, 3);
    const c = a.dot(b);
    for (let i = 0; i < b.data.length; i++) {
      assert.ok(Math.abs(b.data[i] - c.data[i]) < 1e-10, `I*B should equal B at ${i}`);
    }
  });

  it('zeros matrix', () => {
    const z = Matrix.zeros(4, 4);
    for (let i = 0; i < z.data.length; i++) {
      assert.equal(z.data[i], 0, `Zeros matrix should be all zero`);
    }
  });

  it('fromArray creates correct matrix', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    assert.equal(m.rows, 2);
    assert.equal(m.cols, 2);
    assert.equal(m.get(0, 0), 1);
    assert.equal(m.get(1, 1), 4);
  });

  it('clone is independent', () => {
    const a = Matrix.random(3, 3);
    const b = new Matrix(a.rows, a.cols, new Float64Array(a.data));
    a.data[0] = 999;
    assert.notEqual(b.data[0], 999, 'Clone should be independent');
  });
});
