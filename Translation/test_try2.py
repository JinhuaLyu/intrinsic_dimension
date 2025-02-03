from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import functions
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

# get the fine-tuned model
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_dir = "./t5-translation-checkpoints/checkpoint-7500"
model = T5ForConditionalGeneration.from_pretrained(output_dir).to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

raw_dataset = load_dataset("wmt14", "fr-en")
test_dataset = raw_dataset['test'] # Not used in this example, but you can use it for testing
processed_test = test_dataset.map(functions.preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# create a DataLoader for the test set in batches
test_loader = DataLoader(processed_test, batch_size=4, collate_fn=functions.collate_fn)

# evaluate the model
model.eval()

# Initialize cross-entropy loss accumulator
total_loss = 0
num_samples = 0

# Iterate through the test dataset
for batch in test_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Model forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Flatten logits and labels
    logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
    labels = labels.view(-1)  # (batch_size * seq_len)

    # Compute cross-entropy loss, ignoring padding tokens (-100)
    loss = cross_entropy(logits, labels, ignore_index=tokenizer.pad_token_id, reduction="sum")
    total_loss += loss.item()
    num_samples += (labels != tokenizer.pad_token_id).sum().item()

# Compute the average cross-entropy loss
average_loss = total_loss / num_samples
print(f"Test Dataset Cross-Entropy Loss: {average_loss:.4f}")