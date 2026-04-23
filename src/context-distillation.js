// context-distillation.js — Context distillation (Anthropic technique)
// Train the model to internalize few-shot prompt behavior
// by distilling the prompted distribution into the unprompted model

export function contextDistillationLoss(promptedLogits, baseLogits, temperature = 1.0) {
  const n = promptedLogits.length;
  
  // Softmax both
  const prompted = softmax(promptedLogits, temperature);
  const base = softmax(baseLogits, temperature);
  
  // KL(prompted || base) — make base behave like prompted
  let kl = 0;
  for (let i = 0; i < n; i++) {
    if (prompted[i] > 1e-10) {
      kl += prompted[i] * Math.log(prompted[i] / (base[i] + 1e-10));
    }
  }
  
  return temperature * temperature * kl;
}

function softmax(logits, temp) {
  const scaled = logits.map(l => l / temp);
  const max = Math.max(...scaled);
  const exp = scaled.map(l => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(e => e / sum);
}
