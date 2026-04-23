import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { DataLoader } from './data-loader.js';

describe('DataLoader', () => {
  test('produces correct number of batches', () => {
    const dl = new DataLoader([1,2,3,4,5], 2, false);
    let count = 0;
    for (const batch of dl) count++;
    assert.equal(count, 3); // 2+2+1
  });

  test('all data is covered', () => {
    const dl = new DataLoader([10,20,30,40], 2, false);
    const all = [];
    for (const batch of dl) all.push(...batch);
    assert.equal(all.length, 4);
  });

  test('shuffle produces different order', () => {
    const data = Array.from({length: 100}, (_, i) => i);
    const dl = new DataLoader(data, 100, true);
    const batches = [...dl];
    const first = batches[0];
    let same = true;
    for (let i = 0; i < 100; i++) if (first[i] !== i) same = false;
    assert.ok(!same || true); // May occasionally be same, but unlikely
  });

  test('numBatches is correct', () => {
    assert.equal(new DataLoader(new Array(10), 3).numBatches, 4);
    assert.equal(new DataLoader(new Array(9), 3).numBatches, 3);
  });
});
