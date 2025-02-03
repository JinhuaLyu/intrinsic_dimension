from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import functions
import torch
from torch.utils.data import DataLoader
import evaluate
import time

# Load the fine-tuned model
device = "mps" if torch.backends.mps.is_available() else "cpu"
lr = 1e-4
learning_rate = 5e-5
batch_size = 16  # Per-device training batch size
frozen_layers = "none"
weight_decay = 0

output_dir = f"./t5_checkpoints/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}_frozen_{frozen_layers}_wd_{weight_decay}/checkpoint-7825"
output_dir_2 = f"./lora_checkpoints/lora-t5-translation-checkpoints_{lr}/checkpoint-8750"
output_dir_3 = f"./t5_checkpoints/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}/checkpoint-8000"
model = T5ForConditionalGeneration.from_pretrained(output_dir_3).to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load BLEU metric
bleu_metric = evaluate.load("bleu")

# Load and preprocess the dataset
raw_dataset = load_dataset("wmt14", "fr-en")
test_dataset = raw_dataset['test']  # Test set
processed_test = test_dataset.map(functions.preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Create a DataLoader for the test set
test_loader = DataLoader(processed_test, batch_size=32, collate_fn=functions.collate_fn)
start_time = time.time() # Start time
# Set the model to evaluation mode
model.eval()

# Lists to store predictions and references
predictions = []
references = []

# Iterate through the test dataset and generate predictions
for batch in test_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Model inference to generate translations
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
        decoded_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_references = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Append predictions and references
        predictions.extend(decoded_predictions)
        references.extend([[ref] for ref in decoded_references])  # BLEU requires a list of references for each prediction

# Compute BLEU score
bleu_score = bleu_metric.compute(predictions=predictions, references=references)
print(f"Test Dataset BLEU Score: {bleu_score['bleu'] * 100:.2f}")  # Multiply by 100 for percentage format

end_time = time.time()  # End time
print(f"Total Evaluation Time: {end_time - start_time:.2f} seconds")