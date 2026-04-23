import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { EarlyStopping } from './early-stopping.js';

describe('Early Stopping', () => {
  test('does not stop while improving', () => {
    const es = new EarlyStopping(3);
    assert.ok(!es.step(10));
    assert.ok(!es.step(8));
    assert.ok(!es.step(5));
  });

  test('stops after patience exhausted', () => {
    const es = new EarlyStopping(2);
    es.step(5);  // Best
    es.step(6);  // Worse, counter=1
    assert.ok(es.step(7)); // Worse, counter=2 → stop
  });

  test('resets counter on improvement', () => {
    const es = new EarlyStopping(2);
    es.step(5);
    es.step(6);  // counter=1
    es.step(4);  // Improvement → counter=0
    assert.ok(!es.step(5)); // counter=1, not stopped yet
  });

  test('minDelta requires significant improvement', () => {
    const es = new EarlyStopping(2, 0.5);
    es.step(5.0);
    es.step(4.9); // Only 0.1 better, less than minDelta → counter++
    assert.ok(es.step(4.8)); // Still not 0.5 better → stop
  });
});
