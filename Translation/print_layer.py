import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import random, numpy as np, matplotlib.pyplot as plt, os, functions

# Hyperparameters
learning_rate, batch_size, num_epochs, weight_decay = 5e-5, 16, 20, 0
output_dir = f"./t5_checkpoints/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}"
os.makedirs("./results/plots", exist_ok=True)
os.makedirs("./results/csv", exist_ok=True)

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load dataset
dataset = load_dataset("wmt14", "fr-en")
train_data = dataset["train"].shuffle(seed=42).select(range(10000))
val_data = dataset["validation"].shuffle(seed=42).select(range(1000))

# Tokenizer and preprocessing
tokenizer = T5Tokenizer.from_pretrained("t5-small")
process = lambda data: data.map(functions.preprocess_function, batched=True, remove_columns=data.column_names)
train_dataset, val_dataset = process(train_data), process(val_data)

# Model, data collator, and trainer
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

# Print all the layers' names (modules)
print("Model Layers:")
for name, module in model.named_modules():
    print(name)