// demo.js — End-to-end demo: Datasets + ModelZoo + TrainingLogger
//
// Run: node src/demo.js
//
// Demonstrates the full neural-net workflow:
// 1. Generate synthetic data (spiral dataset)
// 2. Create a model (from ModelZoo)
// 3. Train with logging
// 4. Evaluate accuracy
// 5. Display training chart

import { Datasets } from './datasets.js';
import { ModelZoo } from './model-zoo.js';
import { TrainingLogger } from './training-logger.js';
import { Matrix } from './matrix.js';

// 1. Generate data
console.log('📊 Generating spiral dataset...');
const { inputs, targets } = Datasets.spiral(100, 2);
console.log(`   ${inputs.rows} samples, ${inputs.cols}D input, 2 classes`);

// 2. Create model
console.log('\n🧠 Creating model...');
const net = ModelZoo.classifier(2, 2, 16);
console.log('   Architecture: 2 → 16 → 16 → 2 (ReLU, cross-entropy)');

// 3. Train
console.log('\n🏋️ Training...');
const logger = new TrainingLogger('spiral-classifier');

for (let epoch = 0; epoch < 200; epoch++) {
  const loss = net.trainBatch(inputs, targets, 0.1);
  
  // Compute accuracy
  const pred = net.predict(inputs);
  let correct = 0;
  for (let i = 0; i < inputs.rows; i++) {
    const predClass = pred.get(i, 0) > pred.get(i, 1) ? 0 : 1;
    const trueClass = targets.get(i, 0) > targets.get(i, 1) ? 0 : 1;
    if (predClass === trueClass) correct++;
  }
  const accuracy = correct / inputs.rows;
  
  logger.log({ epoch, loss, accuracy });
  
  if (epoch % 50 === 0) {
    console.log(`   Epoch ${epoch}: loss=${loss.toFixed(4)}, accuracy=${(accuracy * 100).toFixed(1)}%`);
  }
}

// 4. Results
console.log('\n📈 Training Summary:');
const summary = logger.summary();
console.log(`   Epochs: ${summary.epochs}`);
console.log(`   Best loss: ${summary.loss.min.toFixed(4)}`);
console.log(`   Final accuracy: ${(logger.tail(1)[0].accuracy * 100).toFixed(1)}%`);
console.log(`   Improving: ${summary.improving ? '✅' : '❌'}`);

// 5. Chart
console.log('\n📉 Loss Chart:');
console.log(logger.chart('loss', 40, 10));

// 6. Export
console.log('\n💾 Exporting...');
const json = logger.toJSON();
const csv = logger.toCSV();
console.log(`   JSON: ${json.length} chars`);
console.log(`   CSV: ${csv.split('\\n').length} rows`);

console.log('\n✅ Demo complete!');
