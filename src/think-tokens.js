// think-tokens.js — Chain-of-Thought / thinking tokens (DeepSeek R1)
// Models the concept of special "thinking" tokens that are generated
// but not included in the final output.

export function extractThinking(tokens, thinkStart, thinkEnd) {
  const thinking = [];
  const output = [];
  let inThinking = false;
  
  for (const token of tokens) {
    if (token === thinkStart) { inThinking = true; continue; }
    if (token === thinkEnd) { inThinking = false; continue; }
    if (inThinking) thinking.push(token);
    else output.push(token);
  }
  
  return { thinking, output };
}

export function thinkingBudget(numThinkTokens, numOutputTokens) {
  return {
    thinkRatio: numThinkTokens / (numThinkTokens + numOutputTokens),
    totalTokens: numThinkTokens + numOutputTokens,
    overhead: numThinkTokens / Math.max(numOutputTokens, 1),
  };
}
