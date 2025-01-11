from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import time

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # binary classification
raw_datasets = load_dataset("imdb")  # For demonstration

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)  # adjust max_length as needed

tokenized_datasets = raw_datasets.map(preprocess, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

## Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "k_lin", "v_lin"],
    task_type="SEQ_CLS"
)

device = torch.device("mps")
peft_model = get_peft_model(model, lora_config).to(device)

# Separate LoRA parameters
lora_A_params = [p for n, p in peft_model.named_parameters() if "lora_A" in n]
lora_B_params = [p for n, p in peft_model.named_parameters() if "lora_B" in n]

# Combine LoRA A and B parameters into one optimizer
lora_params = lora_A_params + lora_B_params
optimizer = AdamW(lora_params, lr=2e-5)

# Custom collate function to convert input_ids into a full tensor
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    labels = torch.tensor([item["label"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Custom training loop
epochs = 5
train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

total_training_time = 0
start_time = time.time()  # Start timer for training

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in tqdm(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = peft_model(**inputs)
        loss = cross_entropy(outputs.logits, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Loss after epoch {epoch + 1}: {loss.item()}")

total_training_time = time.time() - start_time
print(f"Total training time: {total_training_time:.2f} seconds")
# Evaluate model
peft_model.eval()
eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate_fn)

all_preds = []
all_labels = []
total_eval_loss = 0
num_batches = 0

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        labels = batch["labels"].to(device)
        outputs = peft_model(**inputs)
        loss = cross_entropy(outputs.logits, labels)  # Compute evaluation loss
        
        total_eval_loss += loss.item()
        num_batches += 1
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Calculate average evaluation loss
average_eval_loss = total_eval_loss / num_batches
print(f"Average Evaluation Loss: {average_eval_loss:.4f}")

f1 = f1_score(all_labels, all_preds, average="binary")
print("F1 Score (sklearn):", f1)

# Save the LoRA-modified model
peft_model.save_pretrained("lora_distilbert_imdb")

