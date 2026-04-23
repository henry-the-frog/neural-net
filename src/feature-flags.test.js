import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { FeatureFlags, DEFAULT_FLAGS } from './feature-flags.js';

describe('Feature Flags', () => {
  test('get returns default', () => {
    const ff = new FeatureFlags({ useGQA: true });
    assert.equal(ff.get('useGQA'), true);
    assert.equal(ff.get('missing', 42), 42);
  });

  test('isEnabled checks boolean', () => {
    const ff = FeatureFlags.fromConfig(DEFAULT_FLAGS);
    assert.ok(ff.isEnabled('useRoPE'));
    assert.ok(!ff.isEnabled('useMixedPrecision'));
  });

  test('toJSON serializes', () => {
    const ff = new FeatureFlags({ a: 1 });
    assert.deepEqual(ff.toJSON(), { a: 1 });
  });
});
