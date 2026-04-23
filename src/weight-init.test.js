import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { xavierUniform, xavierNormal, kaimingNormal, orthogonal } from './weight-init.js';

describe('Weight Init', () => {
  test('xavier uniform is bounded', () => {
    const init = xavierUniform(100, 100);
    const limit = Math.sqrt(6.0 / 200);
    for (let i = 0; i < 100; i++) {
      const v = init();
      assert.ok(Math.abs(v) <= limit + 0.001);
    }
  });

  test('xavier normal has correct variance', () => {
    const init = xavierNormal(100, 100);
    const samples = Array.from({length: 10000}, init);
    const mean = samples.reduce((a,b) => a+b) / samples.length;
    const variance = samples.reduce((a,b) => a + (b-mean)**2, 0) / samples.length;
    const expectedVar = 2.0 / 200;
    assert.ok(Math.abs(variance - expectedVar) < 0.005);
  });

  test('kaiming normal: variance = 2/fanIn', () => {
    const init = kaimingNormal(100);
    const samples = Array.from({length: 10000}, init);
    const mean = samples.reduce((a,b) => a+b) / samples.length;
    const variance = samples.reduce((a,b) => a + (b-mean)**2, 0) / samples.length;
    assert.ok(Math.abs(variance - 0.02) < 0.005);
  });

  test('orthogonal: rows are unit vectors', () => {
    const M = orthogonal(4);
    for (let i = 0; i < 4; i++) {
      const norm = Math.sqrt(M[i].reduce((s, v) => s + v*v, 0));
      assert.ok(Math.abs(norm - 1) < 0.01);
    }
  });

  test('orthogonal: rows are orthogonal', () => {
    const M = orthogonal(4);
    const dot = M[0].reduce((s, v, i) => s + v * M[1][i], 0);
    assert.ok(Math.abs(dot) < 0.01, `Rows should be orthogonal, dot=${dot}`);
  });
});
