// sequence-packing.js — Efficient sequence packing for training
// Pack multiple short sequences into one long sequence to minimize padding waste.

export function packSequences(sequences, maxLen) {
  const bins = [];
  const remaining = [...sequences].sort((a, b) => b.length - a.length); // Longest first
  
  for (const seq of remaining) {
    let placed = false;
    for (const bin of bins) {
      if (bin.length + seq.length + 1 <= maxLen) { // +1 for separator
        bin.push(...seq);
        bin.push(-1); // Separator token
        placed = true;
        break;
      }
    }
    if (!placed) {
      bins.push([...seq]);
    }
  }
  
  return bins;
}

export function packingEfficiency(sequences, maxLen) {
  const packed = packSequences(sequences, maxLen);
  const totalTokens = sequences.reduce((s, seq) => s + seq.length, 0);
  const packedSlots = packed.length * maxLen;
  return totalTokens / packedSlots;
}
