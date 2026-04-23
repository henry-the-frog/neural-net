// multi-token-prediction.js — Multi-token prediction (Meta, 2024)
// Instead of predicting just the next token, predict N tokens ahead
// Uses N separate prediction heads

export function multiTokenPredictionLoss(logitsPerHead, targetTokens, startIdx) {
  let totalLoss = 0;
  const nHeads = logitsPerHead.length;
  
  for (let h = 0; h < nHeads; h++) {
    const targetIdx = startIdx + h + 1;
    if (targetIdx >= targetTokens.length) break;
    
    const logits = logitsPerHead[h];
    const target = targetTokens[targetIdx];
    
    // Cross-entropy
    const max = Math.max(...logits);
    let sumExp = 0;
    for (const l of logits) sumExp += Math.exp(l - max);
    totalLoss -= (logits[target] - max) - Math.log(sumExp);
  }
  
  return totalLoss / nHeads;
}
