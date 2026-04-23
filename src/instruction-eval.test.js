import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Instruction Tuning evaluation (IFEval concepts)
describe('Instruction Following', () => {
  function checkConstraints(response, constraints) {
    return constraints.map(c => ({
      constraint: c.name,
      passed: c.check(response),
    }));
  }

  test('length constraint', () => {
    const constraints = [{ name: 'maxLen', check: r => r.length <= 100 }];
    const results = checkConstraints('short response', constraints);
    assert.ok(results[0].passed);
  });

  test('format constraint', () => {
    const constraints = [{ name: 'hasBullets', check: r => r.includes('- ') }];
    assert.ok(!checkConstraints('no bullets here', constraints)[0].passed);
    assert.ok(checkConstraints('- bullet point', constraints)[0].passed);
  });
});
