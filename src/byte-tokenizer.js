// byte-tokenizer.js — Byte-Level Tokenization
// Used in ByT5, Byte Latent Transformer (BLT), and other byte-level models.
// No vocabulary learning needed — every byte is a token.
// Advantages: no OOV, handles any language/script, robust to typos.

/**
 * Byte-level tokenizer: text → byte IDs, byte IDs → text.
 * Uses UTF-8 encoding (0-255 for bytes) with optional special token offset.
 */
export class ByteTokenizer {
  constructor(specialTokens = ['<pad>', '<bos>', '<eos>', '<unk>']) {
    this.specialTokens = new Map();
    this.idToSpecial = new Map();
    
    // Reserve IDs 0..N-1 for special tokens, then 256 byte IDs
    let id = 0;
    for (const token of specialTokens) {
      this.specialTokens.set(token, id);
      this.idToSpecial.set(id, token);
      id++;
    }
    
    this.byteOffset = id; // Byte 0 maps to this ID
    this.vocabSize = this.byteOffset + 256;
  }

  /**
   * Encode text to byte token IDs.
   * @param {string} text
   * @param {boolean} addSpecial - Whether to add BOS/EOS
   * @returns {Array<number>}
   */
  encode(text, addSpecial = true) {
    const ids = [];
    if (addSpecial && this.specialTokens.has('<bos>')) {
      ids.push(this.specialTokens.get('<bos>'));
    }
    
    const bytes = new TextEncoder().encode(text);
    for (const byte of bytes) {
      ids.push(byte + this.byteOffset);
    }
    
    if (addSpecial && this.specialTokens.has('<eos>')) {
      ids.push(this.specialTokens.get('<eos>'));
    }
    
    return ids;
  }

  /**
   * Decode byte token IDs back to text.
   * @param {Array<number>} ids
   * @param {boolean} skipSpecial
   * @returns {string}
   */
  decode(ids, skipSpecial = true) {
    const bytes = [];
    for (const id of ids) {
      if (this.idToSpecial.has(id)) {
        if (!skipSpecial) bytes.push(0); // Placeholder
        continue;
      }
      const byteVal = id - this.byteOffset;
      if (byteVal >= 0 && byteVal < 256) {
        bytes.push(byteVal);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  /**
   * Get token for a specific byte value.
   */
  byteToId(byte) {
    return byte + this.byteOffset;
  }

  /**
   * Get byte value for a token ID.
   */
  idToByte(id) {
    const byte = id - this.byteOffset;
    return byte >= 0 && byte < 256 ? byte : null;
  }
}

/**
 * Entropy-based patch boundary detection for BLT.
 * Groups bytes into patches based on entropy thresholds.
 * High-entropy bytes (unpredictable) get their own patch;
 * low-entropy runs (predictable) are grouped together.
 */
export function entropyBasedPatching(bytes, entropyThreshold = 0.5) {
  const patches = [];
  let currentPatch = [bytes[0]];
  
  for (let i = 1; i < bytes.length; i++) {
    // Simple entropy proxy: change in byte value
    const diff = Math.abs(bytes[i] - bytes[i - 1]);
    const localEntropy = diff / 255; // Normalized
    
    if (localEntropy > entropyThreshold) {
      patches.push(currentPatch);
      currentPatch = [bytes[i]];
    } else {
      currentPatch.push(bytes[i]);
    }
  }
  if (currentPatch.length > 0) patches.push(currentPatch);
  
  return patches;
}
