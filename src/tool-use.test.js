import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Tool Use / Function Calling', () => {
  function parseToolCall(text) {
    const match = text.match(/\[TOOL:(\w+)\]\((.*?)\)/);
    if (!match) return null;
    return { name: match[1], args: match[2] };
  }

  test('parses tool call', () => {
    const result = parseToolCall('Let me check [TOOL:search](query here)');
    assert.equal(result.name, 'search');
    assert.equal(result.args, 'query here');
  });

  test('returns null for no tool call', () => {
    assert.equal(parseToolCall('No tool here'), null);
  });
});
