import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
describe('Chat Template', () => {
  function formatChat(messages, template = 'chatml') {
    if (template === 'chatml') {
      return messages.map(m => `<|${m.role}|>\n${m.content}<|end|>`).join('\n');
    }
    return messages.map(m => `[${m.role}]: ${m.content}`).join('\n');
  }
  test('chatml format', () => {
    const result = formatChat([{ role: 'user', content: 'Hi' }]);
    assert.ok(result.includes('<|user|>'));
    assert.ok(result.includes('Hi'));
  });
  test('plain format', () => {
    const result = formatChat([{ role: 'user', content: 'Hi' }], 'plain');
    assert.ok(result.includes('[user]: Hi'));
  });
});
