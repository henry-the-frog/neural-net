// beam-search.js — Beam Search decoding
// Maintains top-B candidate sequences at each step.
// More deterministic than sampling but less creative.

/**
 * Beam search decoding.
 * @param {function} forward - Model forward: tokens → logits (seqLen × vocabSize)
 * @param {Array<number>} prompt - Initial tokens
 * @param {number} maxNewTokens - Max tokens to generate
 * @param {number} beamWidth - Number of beams (candidates)
 * @param {number} eosToken - End of sequence token (optional, -1 to disable)
 * @returns {Array<{ tokens: number[], score: number }>} Top beams sorted by score
 */
export function beamSearch(forward, prompt, maxNewTokens = 32, beamWidth = 4, eosToken = -1) {
  // Initialize beams
  let beams = [{ tokens: [...prompt], score: 0 }];
  const completed = [];
  
  for (let step = 0; step < maxNewTokens; step++) {
    const candidates = [];
    
    for (const beam of beams) {
      const logits = forward(beam.tokens);
      const lastLogits = logits[logits.length - 1] || logits; // Handle both matrix and array
      const vocabSize = lastLogits.length;
      
      // Log-softmax
      const maxLogit = Math.max(...lastLogits);
      const logProbs = new Float64Array(vocabSize);
      let logSumExp = 0;
      for (let i = 0; i < vocabSize; i++) {
        logSumExp += Math.exp(lastLogits[i] - maxLogit);
      }
      logSumExp = Math.log(logSumExp) + maxLogit;
      for (let i = 0; i < vocabSize; i++) {
        logProbs[i] = lastLogits[i] - logSumExp;
      }
      
      // Top-k expansion (only consider top beamWidth*2 tokens for efficiency)
      const indexed = Array.from(logProbs).map((lp, idx) => ({ lp, idx }));
      indexed.sort((a, b) => b.lp - a.lp);
      const topTokens = indexed.slice(0, beamWidth * 2);
      
      for (const { lp, idx } of topTokens) {
        const newBeam = {
          tokens: [...beam.tokens, idx],
          score: beam.score + lp,
        };
        
        if (idx === eosToken) {
          // Length-normalize the score
          newBeam.score /= Math.pow(newBeam.tokens.length, 0.6); // Length penalty
          completed.push(newBeam);
        } else {
          candidates.push(newBeam);
        }
      }
    }
    
    // Keep top beamWidth candidates
    candidates.sort((a, b) => b.score - a.score);
    beams = candidates.slice(0, beamWidth);
    
    if (beams.length === 0) break;
  }
  
  // Combine completed and active beams, sort by score
  const allBeams = [...completed, ...beams.map(b => ({
    ...b,
    score: b.score / Math.pow(b.tokens.length, 0.6),
  }))];
  allBeams.sort((a, b) => b.score - a.score);
  
  return allBeams.slice(0, beamWidth);
}
