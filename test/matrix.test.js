import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';

describe('Matrix construction', () => {
  it('creates zeros', () => {
    const m = Matrix.zeros(3, 4);
    assert.equal(m.rows, 3);
    assert.equal(m.cols, 4);
    assert.equal(m.get(0, 0), 0);
  });

  it('creates from 2D array', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    assert.equal(m.get(0, 0), 1);
    assert.equal(m.get(1, 1), 4);
  });

  it('creates from 1D array (column vector)', () => {
    const m = Matrix.fromArray([1, 2, 3]);
    assert.equal(m.rows, 3);
    assert.equal(m.cols, 1);
    assert.equal(m.get(2, 0), 3);
  });

  it('one-hot encoding', () => {
    const m = Matrix.oneHot([0, 2, 1], 3);
    assert.equal(m.get(0, 0), 1);
    assert.equal(m.get(0, 1), 0);
    assert.equal(m.get(1, 2), 1);
    assert.equal(m.get(2, 1), 1);
  });
});

describe('Matrix arithmetic', () => {
  it('dot product', () => {
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = a.dot(b);
    assert.equal(c.get(0, 0), 19); // 1*5 + 2*7
    assert.equal(c.get(1, 1), 50); // 3*6 + 4*8
  });

  it('element-wise add', () => {
    const a = Matrix.fromArray([[1, 2], [3, 4]]);
    const b = Matrix.fromArray([[5, 6], [7, 8]]);
    const c = a.add(b);
    assert.equal(c.get(0, 0), 6);
    assert.equal(c.get(1, 1), 12);
  });

  it('scalar add', () => {
    const m = Matrix.fromArray([[1, 2]]).add(10);
    assert.equal(m.get(0, 0), 11);
    assert.equal(m.get(0, 1), 12);
  });

  it('element-wise subtract', () => {
    const c = Matrix.fromArray([[5, 6]]).sub(Matrix.fromArray([[1, 2]]));
    assert.equal(c.get(0, 0), 4);
    assert.equal(c.get(0, 1), 4);
  });

  it('element-wise multiply (Hadamard)', () => {
    const c = Matrix.fromArray([[2, 3]]).mul(Matrix.fromArray([[4, 5]]));
    assert.equal(c.get(0, 0), 8);
    assert.equal(c.get(0, 1), 15);
  });

  it('scalar multiply', () => {
    const m = Matrix.fromArray([[1, 2]]).mul(3);
    assert.equal(m.get(0, 0), 3);
    assert.equal(m.get(0, 1), 6);
  });

  it('transpose', () => {
    const m = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
    const t = m.T();
    assert.equal(t.rows, 3);
    assert.equal(t.cols, 2);
    assert.equal(t.get(0, 0), 1);
    assert.equal(t.get(0, 1), 4);
    assert.equal(t.get(2, 1), 6);
  });
});

describe('Matrix operations', () => {
  it('map', () => {
    const m = Matrix.fromArray([[1, 4, 9]]).map(Math.sqrt);
    assert.ok(Math.abs(m.get(0, 0) - 1) < 1e-10);
    assert.ok(Math.abs(m.get(0, 1) - 2) < 1e-10);
    assert.ok(Math.abs(m.get(0, 2) - 3) < 1e-10);
  });

  it('sum', () => {
    assert.equal(Matrix.fromArray([[1, 2], [3, 4]]).sum(), 10);
  });

  it('sumAxis 0 (sum rows)', () => {
    const s = Matrix.fromArray([[1, 2], [3, 4]]).sumAxis(0);
    assert.equal(s.rows, 1);
    assert.equal(s.cols, 2);
    assert.equal(s.get(0, 0), 4);
    assert.equal(s.get(0, 1), 6);
  });

  it('sumAxis 1 (sum cols)', () => {
    const s = Matrix.fromArray([[1, 2], [3, 4]]).sumAxis(1);
    assert.equal(s.rows, 2);
    assert.equal(s.cols, 1);
    assert.equal(s.get(0, 0), 3);
    assert.equal(s.get(1, 0), 7);
  });

  it('argmax', () => {
    const m = Matrix.fromArray([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]);
    const idx = m.argmax();
    assert.equal(idx[0], 1);
    assert.equal(idx[1], 0);
  });

  it('max', () => {
    assert.equal(Matrix.fromArray([[1, 5, 3]]).max(), 5);
  });

  it('clone', () => {
    const a = Matrix.fromArray([[1, 2]]);
    const b = a.clone();
    b.set(0, 0, 99);
    assert.equal(a.get(0, 0), 1); // Original unchanged
    assert.equal(b.get(0, 0), 99);
  });

  it('broadcasting: row vector add', () => {
    const m = Matrix.fromArray([[1, 2], [3, 4]]);
    const v = Matrix.fromArray([[10, 20]]);
    const r = m.add(v);
    assert.equal(r.get(0, 0), 11);
    assert.equal(r.get(1, 1), 24);
  });
});

describe('Matrix randomize', () => {
  it('Xavier initialization', () => {
    const m = Matrix.random(100, 50);
    // Xavier scale should be sqrt(2/(100+50)) ≈ 0.115
    // Most values should be in [-0.3, 0.3]
    let inRange = 0;
    for (let i = 0; i < m.data.length; i++) {
      if (Math.abs(m.data[i]) < 0.5) inRange++;
    }
    assert.ok(inRange / m.data.length > 0.9, 'Most values should be small');
  });
});

describe('Matrix shape errors', () => {
  it('dot product shape mismatch', () => {
    const a = Matrix.fromArray([[1, 2]]);
    const b = Matrix.fromArray([[3, 4]]);
    assert.throws(() => a.dot(b)); // 1×2 · 1×2 fails
  });
});
