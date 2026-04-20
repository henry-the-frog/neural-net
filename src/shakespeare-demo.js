// shakespeare-demo.js — Train a tiny char-level transformer on Shakespeare
// Demonstrates the CharLM on actual English text

import { CharTokenizer, CharLM } from './char-lm.js';

// A small excerpt of Shakespeare (Hamlet's soliloquy + more)
const SHAKESPEARE = `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.`;

console.log('=== Shakespeare Character LM Demo ===\n');

// Setup
const tokenizer = new CharTokenizer().fit(SHAKESPEARE);
console.log(`Vocabulary: ${tokenizer.vocabSize} characters`);
console.log(`Text length: ${SHAKESPEARE.length} chars`);

const model = new CharLM({
  vocabSize: tokenizer.vocabSize,
  dModel: 32,     // Small model
  nHeads: 4,      // 4 attention heads
  nLayers: 2,     // 2 decoder blocks
  dFF: 64,        // FFN hidden size
  maxLen: 32,     // Context window
});
console.log(`Parameters: ${model.paramCount()}`);

// Tokenize
const tokens = tokenizer.encode(SHAKESPEARE);

// Training loop
const EPOCHS = 2000;
const WINDOW_SIZE = 24; // Training window
const LR = 0.003;

console.log(`\nTraining for ${EPOCHS} steps...\n`);

const losses = [];
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Random window from the text
  const start = Math.floor(Math.random() * (tokens.length - WINDOW_SIZE));
  const window = tokens.slice(start, start + WINDOW_SIZE);
  const loss = model.trainStep(window, LR);
  losses.push(loss);
  
  if (epoch % 500 === 0 || epoch === EPOCHS - 1) {
    const avgLoss = losses.slice(-100).reduce((a, b) => a + b) / Math.min(100, losses.length);
    console.log(`Step ${epoch}: loss = ${avgLoss.toFixed(4)}`);
  }
}

// Generate samples
console.log('\n--- Generated Samples ---\n');

const prompts = ['To be', 'The ', 'And ', 'For '];
for (const prompt of prompts) {
  const promptTokens = tokenizer.encode(prompt);
  const generated = model.generate(promptTokens, 80, 0.7);
  const text = tokenizer.decode(generated);
  console.log(`Prompt: "${prompt}"`);
  console.log(`Output: "${text}"\n`);
}

// Low temperature (more deterministic)
console.log('--- Low temperature (0.3) ---\n');
const lowTemp = model.generate(tokenizer.encode('To '), 60, 0.3);
console.log(`"${tokenizer.decode(lowTemp)}"\n`);

// High temperature (more creative)
console.log('--- High temperature (1.2) ---\n');
const highTemp = model.generate(tokenizer.encode('To '), 60, 1.2);
console.log(`"${tokenizer.decode(highTemp)}"\n`);
