from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.nn.functional import cross_entropy

# Load pre-trained T5-small model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load W14 English-French dataset
dataset = load_dataset("wmt14", "fr-en")
test_data = dataset["test"]

# Extract inputs and targets from the test set
inputs = [f"translate English to French: {example['translation']['en']}" for example in test_data if 'translation' in example and 'en' in example['translation']]
targets = [example['translation']['fr'] for example in test_data if 'translation' in example and 'fr' in example['translation']]

# Function to compute cross-entropy loss
def compute_cross_entropy_loss(model, tokenizer, inputs, targets):
    model.eval()
    total_loss = 0
    num_samples = len(inputs)
    
    for inp, target in zip(inputs, targets):
        # Tokenize input and target
        input_ids = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = tokenizer(target, return_tensors="pt", padding=True, truncation=True).input_ids

        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)

        # Extract logits and compute loss
        logits = outputs.logits
        loss = cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=tokenizer.pad_token_id)
        total_loss += loss.item()

    return total_loss / num_samples

# Evaluate the model
average_loss = compute_cross_entropy_loss(model, tokenizer, inputs, targets)
print(f"Average Cross-Entropy Loss on W14 Test Set: {average_loss:.4f}")