import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Safety Classifier', () => {
  function classifySafety(text, dangerousPatterns) {
    for (const pattern of dangerousPatterns) {
      if (text.toLowerCase().includes(pattern)) return { safe: false, reason: pattern };
    }
    return { safe: true };
  }

  test('flags dangerous content', () => {
    const result = classifySafety('How to hack a system', ['hack', 'weapon']);
    assert.ok(!result.safe);
    assert.equal(result.reason, 'hack');
  });

  test('passes safe content', () => {
    assert.ok(classifySafety('Hello world', ['hack']).safe);
  });
});
