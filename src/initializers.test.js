// initializers.test.js — Test weight initialization strategies
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { 
  xavierUniform, xavierNormal, heUniform, heNormal, 
  lecunNormal, zeros, ones, createInitializer 
} from './initializers.js';

function stats(matrix) {
  const data = matrix.data;
  const n = data.length;
  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < n; i++) {
    sum += data[i];
    sumSq += data[i] * data[i];
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  return { mean, variance, std: Math.sqrt(variance), min, max, n };
}

describe('Weight Initializers', () => {
  test('xavierUniform: correct shape and bounds', () => {
    const m = xavierUniform(100, 50, 100, 50);
    assert.equal(m.rows, 100);
    assert.equal(m.cols, 50);
    const limit = Math.sqrt(6.0 / (100 + 50));
    const s = stats(m);
    assert.ok(s.max <= limit * 1.01, `max ${s.max} should be <= ${limit}`);
    assert.ok(s.min >= -limit * 1.01, `min ${s.min} should be >= ${-limit}`);
  });

  test('xavierUniform: variance approximates 2/(fanIn+fanOut)', () => {
    const m = xavierUniform(200, 100, 200, 100);
    const s = stats(m);
    const expectedVar = 2.0 / (200 + 100); // uniform var = (b-a)^2/12 = 4*limit^2/12 = limit^2/3
    // Actually for uniform(-limit, limit): var = limit^2/3 where limit = sqrt(6/(fin+fout))
    // var = 6/(fin+fout) / 3 = 2/(fin+fout)
    assert.ok(Math.abs(s.variance - expectedVar) < 0.005, 
      `variance ${s.variance} should be close to ${expectedVar}`);
  });

  test('xavierNormal: mean near 0, variance approximates 2/(fanIn+fanOut)', () => {
    const m = xavierNormal(200, 200, 200, 200);
    const s = stats(m);
    assert.ok(Math.abs(s.mean) < 0.05, `mean ${s.mean} should be near 0`);
    const expectedVar = 2.0 / (200 + 200);
    assert.ok(Math.abs(s.variance - expectedVar) < 0.003, 
      `variance ${s.variance} should be close to ${expectedVar}`);
  });

  test('heUniform: correct bounds for ReLU', () => {
    const fanIn = 128;
    const m = heUniform(128, 64, fanIn);
    const limit = Math.sqrt(6.0 / fanIn);
    const s = stats(m);
    assert.ok(s.max <= limit * 1.01);
    assert.ok(s.min >= -limit * 1.01);
  });

  test('heNormal: variance approximates 2/fanIn', () => {
    const fanIn = 256;
    const m = heNormal(256, 128, fanIn);
    const s = stats(m);
    const expectedVar = 2.0 / fanIn;
    assert.ok(Math.abs(s.mean) < 0.05, `mean ${s.mean} should be near 0`);
    assert.ok(Math.abs(s.variance - expectedVar) < 0.003, 
      `variance ${s.variance} should be close to ${expectedVar}`);
  });

  test('lecunNormal: variance approximates 1/fanIn', () => {
    const fanIn = 256;
    const m = lecunNormal(256, 128, fanIn);
    const s = stats(m);
    const expectedVar = 1.0 / fanIn;
    assert.ok(Math.abs(s.mean) < 0.05, `mean ${s.mean} should be near 0`);
    assert.ok(Math.abs(s.variance - expectedVar) < 0.002, 
      `variance ${s.variance} should be close to ${expectedVar}`);
  });

  test('zeros: all elements are 0', () => {
    const m = zeros(10, 5);
    const s = stats(m);
    assert.equal(s.mean, 0);
    assert.equal(s.variance, 0);
    assert.equal(s.min, 0);
    assert.equal(s.max, 0);
  });

  test('ones: all elements are 1', () => {
    const m = ones(10, 5);
    const s = stats(m);
    assert.equal(s.mean, 1);
    assert.equal(s.variance, 0);
  });

  test('createInitializer: all names work', () => {
    const names = ['xavier_uniform', 'glorot_uniform', 'xavier_normal', 'glorot_normal',
                   'he_uniform', 'kaiming_uniform', 'he_normal', 'kaiming_normal',
                   'lecun', 'lecun_normal', 'zeros', 'ones'];
    for (const name of names) {
      const init = createInitializer(name);
      assert.equal(typeof init, 'function', `${name} should return function`);
    }
  });

  test('createInitializer: unknown name throws', () => {
    assert.throws(() => createInitializer('nonexistent'), /Unknown initializer/);
  });

  test('He >> Xavier variance for same fanIn (ReLU appropriate)', () => {
    const fanIn = 128, fanOut = 64;
    const xavier = xavierNormal(fanIn, fanOut, fanIn, fanOut);
    const he = heNormal(fanIn, fanOut, fanIn);
    const xStats = stats(xavier);
    const hStats = stats(he);
    // He variance should be ~2x Xavier for same fanIn when fanIn ≈ fanOut
    assert.ok(hStats.variance > xStats.variance, 
      `He variance (${hStats.variance}) should be > Xavier (${xStats.variance})`);
  });

  test('randomNormal produces normally distributed values', () => {
    const m = xavierNormal(100, 100, 100, 100);
    const s = stats(m);
    // For normal distribution, ~99.7% should be within 3 std
    const within3std = m.data.filter(v => Math.abs(v - s.mean) < 3 * s.std).length;
    const pct = within3std / m.data.length;
    assert.ok(pct > 0.99, `${(pct*100).toFixed(1)}% within 3 std (expected >99%)`);
  });
});
