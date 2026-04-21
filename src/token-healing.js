// token-healing.js — Token Healing for BPE Boundary Issues
// Paper/concept from Microsoft Guidance project
//
// Problem: When the prompt ends mid-token, BPE tokenizes differently
// than if the text continued. E.g., prompt "Hel" gets tokenized as ["Hel"],
// but the model trained on "Hello" → ["He", "llo"] never saw "Hel" alone.
//
// Solution: "Heal" the boundary by backing up one token and re-generating,
// constraining the first generated token to start with the backed-up token's text.

/**
 * Token healing: fix BPE boundary issues at the prompt/generation boundary.
 *
 * @param {object} tokenizer - BPE tokenizer with encode/decode
 * @param {number[]} promptTokenIds - tokenized prompt
 * @returns {{ healedPrompt: number[], constraintPrefix: string }}
 */
export function healTokenBoundary(tokenizer, promptTokenIds) {
  if (promptTokenIds.length === 0) {
    return { healedPrompt: [], constraintPrefix: '' };
  }

  // Get the last token's text
  const lastTokenId = promptTokenIds[promptTokenIds.length - 1];
  const lastTokenText = tokenizer.decode([lastTokenId]);

  // Check if this token could be a prefix of a longer token
  // by checking if any vocabulary entry starts with this text
  let isPrefix = false;
  for (let id = 0; id < tokenizer.idToToken.length; id++) {
    const token = tokenizer.idToToken[id];
    if (token && token.length > lastTokenText.length && token.startsWith(lastTokenText)) {
      isPrefix = true;
      break;
    }
  }

  if (!isPrefix) {
    // Last token is not a prefix of any longer token — no healing needed
    return { healedPrompt: promptTokenIds, constraintPrefix: '' };
  }

  // Back up one token: remove last token from prompt
  const healedPrompt = promptTokenIds.slice(0, -1);
  // The constraint: first generated token must start with lastTokenText
  return { healedPrompt, constraintPrefix: lastTokenText };
}

/**
 * Get tokens that start with a given prefix.
 * Used to constrain generation after token healing.
 *
 * @param {object} tokenizer
 * @param {string} prefix
 * @returns {number[]} valid token IDs
 */
export function getTokensWithPrefix(tokenizer, prefix) {
  const valid = [];
  for (let id = 0; id < tokenizer.idToToken.length; id++) {
    const token = tokenizer.idToToken[id];
    if (token && token.startsWith(prefix)) {
      valid.push(id);
    }
  }
  return valid;
}
