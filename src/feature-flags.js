// feature-flags.js — Neural network feature flag system for experiments
export class FeatureFlags {
  constructor(defaults = {}) {
    this.flags = { ...defaults };
  }
  
  set(key, value) { this.flags[key] = value; }
  get(key, fallback = false) { return this.flags[key] ?? fallback; }
  isEnabled(key) { return !!this.flags[key]; }
  toJSON() { return { ...this.flags }; }
  
  static fromConfig(config) {
    const ff = new FeatureFlags();
    ff.flags = { ...config };
    return ff;
  }
}

// Common ML experiment flags
export const DEFAULT_FLAGS = {
  useFlashAttention: true,
  useGQA: false,
  useRoPE: true,
  useSwiGLU: true,
  useMixedPrecision: false,
  useGradientCheckpointing: false,
  useEMA: true,
  dropoutRate: 0.1,
};
