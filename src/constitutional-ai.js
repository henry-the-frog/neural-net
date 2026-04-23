// constitutional-ai.js — Constitutional AI (Bai et al., 2022)
// Self-critique: model generates response, then critiques it against principles

export function constitutionalCritique(response, principle) {
  // Returns a critique prompt
  return `Critique the following response based on this principle: "${principle}"\n\nResponse: "${response}"\n\nCritique:`;
}

export function revisionPrompt(response, critique) {
  return `Please revise the following response based on this feedback:\n\nOriginal: "${response}"\nFeedback: "${critique}"\n\nRevised:`;
}

export function rankByPrinciples(responses, principles, scoreFn) {
  return responses.map((r, i) => {
    let totalScore = 0;
    for (const p of principles) {
      totalScore += scoreFn(r, p);
    }
    return { index: i, response: r, score: totalScore / principles.length };
  }).sort((a, b) => b.score - a.score);
}
