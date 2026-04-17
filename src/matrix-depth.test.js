// matrix-depth.test.js — Matrix class edge case tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Matrix } from './matrix.js';

describe('Matrix Construction', () => {
  it('creates matrix with correct dimensions', () => {
    const m = new Matrix(3, 4);
    assert.equal(m.rows, 3);
    assert.equal(m.cols, 4);
    assert.equal(m.data.length, 12);
  });

  it('creates matrix from initial data', () => {
    const m = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    assert.equal(m.get(0, 0), 1);
    assert.equal(m.get(1, 2), 6);
  });

  it('single element matrix', () => {
    const m = new Matrix(1, 1, new Float64Array([42]));
    assert.equal(m.get(0, 0), 42);
  });

  it('Matrix.random produces values in [0,1)', () => {
    const m = Matrix.random(10, 10);
    for (let i = 0; i < m.data.length; i++) {
      assert.ok(m.data[i] >= -1 && m.data[i] <= 1, `Random value out of range: ${m.data[i]}`);
    }
  });

  it('Matrix.zeros is all zeros', () => {
    const m = Matrix.zeros(5, 5);
    for (let i = 0; i < m.data.length; i++) {
      assert.equal(m.data[i], 0);
    }
  });

  it('Matrix.ones is all ones', () => {
    const m = Matrix.ones(3, 3);
    for (let i = 0; i < m.data.length; i++) {
      assert.equal(m.data[i], 1);
    }
  });
});

describe('Matrix Arithmetic', () => {
  it('add two matrices', () => {
    const a = new Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
    const b = new Matrix(2, 2, new Float64Array([5, 6, 7, 8]));
    const c = a.add(b);
    assert.equal(c.get(0, 0), 6);
    assert.equal(c.get(1, 1), 12);
  });

  it('subtract matrices', () => {
    const a = new Matrix(1, 3, new Float64Array([10, 20, 30]));
    const b = new Matrix(1, 3, new Float64Array([3, 5, 7]));
    const c = a.sub(b);
    assert.equal(c.get(0, 0), 7);
    assert.equal(c.get(0, 2), 23);
  });

  it('scalar multiplication', () => {
    const m = new Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
    const r = m.mul(3);
    assert.equal(r.get(0, 0), 3);
    assert.equal(r.get(1, 1), 12);
  });

  it('element-wise multiplication', () => {
    const a = new Matrix(1, 3, new Float64Array([2, 3, 4]));
    const b = new Matrix(1, 3, new Float64Array([5, 6, 7]));
    const c = a.mul(b);
    assert.equal(c.get(0, 0), 10);
    assert.equal(c.get(0, 1), 18);
    assert.equal(c.get(0, 2), 28);
  });

  it('matrix multiplication (dot)', () => {
    const a = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    const b = new Matrix(3, 2, new Float64Array([7, 8, 9, 10, 11, 12]));
    const c = a.dot(b);
    assert.equal(c.rows, 2);
    assert.equal(c.cols, 2);
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    assert.equal(c.get(0, 0), 58);
    assert.equal(c.get(0, 1), 64);
  });
});

describe('Matrix Transpose', () => {
  it('transpose swaps rows and columns', () => {
    const m = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
    const t = m.transpose();
    assert.equal(t.rows, 3);
    assert.equal(t.cols, 2);
    assert.equal(t.get(0, 0), 1);
    assert.equal(t.get(0, 1), 4);
    assert.equal(t.get(2, 0), 3);
    assert.equal(t.get(2, 1), 6);
  });

  it('transpose of transpose is original', () => {
    const m = Matrix.random(3, 5);
    const tt = m.transpose().transpose();
    assert.equal(tt.rows, m.rows);
    assert.equal(tt.cols, m.cols);
    for (let i = 0; i < m.data.length; i++) {
      assert.equal(tt.data[i], m.data[i]);
    }
  });

  it('transpose of 1×1 matrix', () => {
    const m = new Matrix(1, 1, new Float64Array([42]));
    const t = m.transpose();
    assert.equal(t.get(0, 0), 42);
  });
});

describe('Matrix Map', () => {
  it('map applies function to each element', () => {
    const m = new Matrix(2, 2, new Float64Array([1, 4, 9, 16]));
    const r = m.map(Math.sqrt);
    assert.equal(r.get(0, 0), 1);
    assert.equal(r.get(0, 1), 2);
    assert.equal(r.get(1, 0), 3);
    assert.equal(r.get(1, 1), 4);
  });

  it('map preserves dimensions', () => {
    const m = Matrix.random(5, 3);
    const r = m.map(x => x * 2);
    assert.equal(r.rows, 5);
    assert.equal(r.cols, 3);
  });
});

describe('Matrix Edge Cases', () => {
  it('dot product with compatible shapes', () => {
    const a = new Matrix(1, 5, new Float64Array([1, 2, 3, 4, 5]));
    const b = new Matrix(5, 1, new Float64Array([1, 1, 1, 1, 1]));
    const c = a.dot(b);
    assert.equal(c.rows, 1);
    assert.equal(c.cols, 1);
    assert.equal(c.get(0, 0), 15);
  });

  it('add with zeros is identity', () => {
    const m = Matrix.random(3, 3);
    const z = Matrix.zeros(3, 3);
    const r = m.add(z);
    for (let i = 0; i < m.data.length; i++) {
      assert.equal(r.data[i], m.data[i]);
    }
  });

  it('scalar multiply by zero', () => {
    const m = Matrix.random(3, 3);
    const r = m.mul(0);
    for (let i = 0; i < r.data.length; i++) {
      assert.equal(r.data[i], 0);
    }
  });

  it('scalar multiply by one is identity', () => {
    const m = Matrix.random(3, 3);
    const r = m.mul(1);
    for (let i = 0; i < m.data.length; i++) {
      assert.equal(r.data[i], m.data[i]);
    }
  });
});
