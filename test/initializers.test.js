import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { xavierUniform, xavierNormal, heUniform, heNormal, lecunNormal, zeros, ones, createInitializer } from '../src/initializers.js';

function stats(matrix) {
  let sum = 0, sumSq = 0;
  const n = matrix.data.length;
  for (let i = 0; i < n; i++) {
    sum += matrix.data[i];
    sumSq += matrix.data[i] * matrix.data[i];
  }
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  return { mean, variance, std: Math.sqrt(variance) };
}

describe('Weight Initializers', () => {
  describe('xavierUniform', () => {
    it('has correct variance', () => {
      const m = xavierUniform(100, 100, 100, 100);
      const s = stats(m);
      // Var(U(-a,a)) = a²/3, limit = sqrt(6/200) = 0.1732, Var = 0.01
      assert.ok(Math.abs(s.variance - 0.01) < 0.005, `Var: ${s.variance}`);
    });
    it('has near-zero mean', () => {
      const m = xavierUniform(100, 100, 100, 100);
      assert.ok(Math.abs(stats(m).mean) < 0.05);
    });
  });

  describe('xavierNormal', () => {
    it('has correct variance', () => {
      const m = xavierNormal(100, 100, 100, 100);
      const s = stats(m);
      // Var = 2/(100+100) = 0.01
      assert.ok(Math.abs(s.variance - 0.01) < 0.005, `Var: ${s.variance}`);
    });
  });

  describe('heUniform', () => {
    it('has correct variance for ReLU', () => {
      const m = heUniform(100, 100, 100);
      const s = stats(m);
      // limit = sqrt(6/100), Var = limit²/3 = 6/(100*3) = 0.02
      assert.ok(Math.abs(s.variance - 0.02) < 0.01, `Var: ${s.variance}`);
    });
  });

  describe('heNormal', () => {
    it('has correct variance for ReLU', () => {
      const m = heNormal(100, 100, 100);
      const s = stats(m);
      // Var = 2/100 = 0.02
      assert.ok(Math.abs(s.variance - 0.02) < 0.01, `Var: ${s.variance}`);
    });
    it('is different for different fanIn', () => {
      const m1 = heNormal(100, 100, 10);
      const m2 = heNormal(100, 100, 1000);
      assert.ok(stats(m1).variance > stats(m2).variance, 'Smaller fanIn → larger variance');
    });
  });

  describe('lecunNormal', () => {
    it('has correct variance', () => {
      const m = lecunNormal(100, 100, 100);
      const s = stats(m);
      // Var = 1/100 = 0.01
      assert.ok(Math.abs(s.variance - 0.01) < 0.005, `Var: ${s.variance}`);
    });
  });

  describe('zeros and ones', () => {
    it('zeros produces all zeros', () => {
      const m = zeros(3, 3);
      for (let i = 0; i < m.data.length; i++) assert.equal(m.data[i], 0);
    });
    it('ones produces all ones', () => {
      const m = ones(3, 3);
      for (let i = 0; i < m.data.length; i++) assert.equal(m.data[i], 1);
    });
  });

  describe('createInitializer', () => {
    it('creates xavier by name', () => {
      const init = createInitializer('xavier_uniform');
      const m = init(10, 10, 10, 10);
      assert.equal(m.rows, 10);
    });
    it('aliases work', () => {
      assert.ok(createInitializer('glorot_uniform'));
      assert.ok(createInitializer('kaiming_normal'));
    });
    it('throws on unknown', () => {
      assert.throws(() => createInitializer('unknown'));
    });
  });
});
