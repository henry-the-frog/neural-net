// text-generation.js — Simple text generation utilities
export function greedyDecode(logitsFn, prompt, maxLen = 50) {
  const tokens = [...prompt];
  for (let i = 0; i < maxLen; i++) {
    const logits = logitsFn(tokens);
    const nextToken = logits.indexOf(Math.max(...logits));
    tokens.push(nextToken);
  }
  return tokens;
}

export function repetitionCheck(tokens, windowSize = 10) {
  if (tokens.length < windowSize * 2) return false;
  const last = tokens.slice(-windowSize);
  const prev = tokens.slice(-windowSize * 2, -windowSize);
  return JSON.stringify(last) === JSON.stringify(prev);
}
