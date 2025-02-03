from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score, accuracy_score
import evaluate

# Load pre-trained T5-small model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load W14 English-French dataset
dataset = load_dataset("wmt14", "fr-en")
test_data = dataset["test"]
print(len(test_data))

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

# # Function to compute EM, F1, and Accuracy
# def compute_metrics(model, tokenizer, inputs, targets):
#     model.eval()
#     em_count = 0
#     total_count = 0
#     f1_total = 0
#     correct_predictions = 0

#     for inp, target in zip(inputs, targets):
#         # Tokenize input and generate prediction
#         input_ids = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).input_ids
#         with torch.no_grad():
#             generated_ids = model.generate(input_ids, max_length=50)
#         prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#         # Compute Exact Match (EM)
#         if prediction.strip() == target.strip():
#             em_count += 1

#         # Compute F1 Score
#         pred_tokens = set(prediction.strip().split())
#         target_tokens = set(target.strip().split())
#         common_tokens = pred_tokens.intersection(target_tokens)

#         if len(common_tokens) > 0:
#             precision = len(common_tokens) / len(pred_tokens)
#             recall = len(common_tokens) / len(target_tokens)
#             f1_score = 2 * (precision * recall) / (precision + recall)
#         else:
#             f1_score = 0
#         f1_total += f1_score

#         total_count += 1

#     # Calculate overall metrics
#     em = em_count / total_count
#     avg_f1 = f1_total / total_count

#     return em, avg_f1

# Load BLEU metric
bleu_metric = evaluate.load("bleu")
# Function to compute BLEU score
def compute_bleu(model, tokenizer, inputs, targets):
    model.eval()
    predictions = []

    for inp in inputs:
        # Tokenize input and generate prediction
        input_ids = tokenizer(inp, return_tensors="pt", padding=True, truncation=True).input_ids
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=50)
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

    # Format data for BLEU computation
    references = [[target] for target in targets]  # Wrap each target in a list

    # Compute BLEU
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    return bleu_score["bleu"]


# Evaluate the model
average_loss = compute_cross_entropy_loss(model, tokenizer, inputs, targets)
print(f"Average Cross-Entropy Loss on W14 Test Set: {average_loss:.4f}")

# em, avg_f1 = compute_metrics(model, tokenizer, inputs, targets)
# print(f"Exact Match (EM) on W14 Test Set: {em:.4f}")
# print(f"F1 Score on W14 Test Set: {avg_f1:.4f}")

bleu_score = compute_bleu(model, tokenizer, inputs, targets)
print(f"BLEU Score on W14 Test Set: {bleu_score:.4f}")