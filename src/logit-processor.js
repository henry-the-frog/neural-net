// logit-processor.js — Logit processing pipeline for text generation
export class LogitProcessor {
  constructor() { this.processors = []; }
  add(fn) { this.processors.push(fn); return this; }
  
  process(logits, context) {
    let result = new Float64Array(logits);
    for (const fn of this.processors) result = fn(result, context);
    return result;
  }
}

export function temperatureProcessor(temp) {
  return (logits) => logits.map(l => l / temp);
}

export function repetitionPenaltyProcessor(penalty) {
  return (logits, ctx) => {
    const seen = new Set(ctx.generatedTokens || []);
    return logits.map((l, i) => seen.has(i) ? (l > 0 ? l / penalty : l * penalty) : l);
  };
}

export function banTokensProcessor(bannedIds) {
  return (logits) => logits.map((l, i) => bannedIds.includes(i) ? -Infinity : l);
}
