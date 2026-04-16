// Test helpers for stochastic neural network tests
// Retry a function up to `attempts` times until it succeeds (no exception)
export function retry(fn, attempts = 3) {
  for (let i = 0; i < attempts; i++) {
    try {
      fn();
      return; // success
    } catch (e) {
      if (i === attempts - 1) throw e; // last attempt, rethrow
    }
  }
}
