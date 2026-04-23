// token-healing.js — Token Healing for LLM generation
// When the prompt ends mid-token, the LLM might generate unexpected output.
// Token healing: backtrack to the last complete token boundary before generating.

export function findTokenBoundary(text, tokens) {
  // Find the longest prefix that aligns with a complete token
  let lastComplete = 0;
  let pos = 0;
  for (const token of tokens) {
    pos += token.length;
    if (pos <= text.length) lastComplete = pos;
  }
  return lastComplete;
}

export function healedPrompt(text, tokenizer) {
  const tokens = tokenizer(text);
  if (tokens.length === 0) return { text, backtrack: 0 };
  
  // Remove last token (might be incomplete) and regenerate from there
  const lastToken = tokens[tokens.length - 1];
  const healedText = text.slice(0, text.length - lastToken.length);
  
  return {
    text: healedText,
    backtrack: lastToken.length,
    constrainPrefix: lastToken,
  };
}
