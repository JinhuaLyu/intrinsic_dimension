import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import matplotlib.pyplot as plt

# ----------------------------
# 1. Basic Settings and Directory Setup
# ----------------------------
os.makedirs("./results/plots", exist_ok=True)
os.makedirs("./results/csv", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 2. Load Dataset and Preprocess
# ----------------------------
# Load the WMT14 fr-en dataset (using a subset for demonstration)
dataset = load_dataset("wmt14", "fr-en")
train_data = dataset["train"].shuffle(seed=42).select(range(10000))
val_data = dataset["validation"].shuffle(seed=42).select(range(1000))

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    # Each sample is a dict, e.g., {"translation": {"fr": "...", "en": "..."}}
    # Add a translation prompt for the T5 model
    inputs = ["translate French to English: " + ex["fr"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    # Tokenize the target texts
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
val_dataset = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

# ----------------------------
# 3. Load Model and Register Hook for a Random Training Step
# ----------------------------
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Specify the parameter (layer) you are interested in (here, an encoder weight is used as an example)
layer_name = "decoder.block.4.layer.1.EncDecAttention.q.weight"

# Global variable to store the singular values of the gradient at the target step
target_step_singular_values = None

# Global counter to track the number of times the hook is called
global_hook_call_count = 0

# Choose a random target step (for demonstration, choose between 1 and 10)
target_step = 20
print(f"Target step for recording singular values is: {target_step}")

def check_singular_values(grad_output):
    global target_step_singular_values, global_hook_call_count, target_step
    global_hook_call_count += 1
    # Only record singular values if this is the target step and not recorded yet
    if global_hook_call_count == target_step and target_step_singular_values is None:
        grad_matrix = grad_output[0]  # grad_output is a tuple; the first element is the gradient matrix
        # Compute singular values (we don't need U and V)
        _, singular_values, _ = torch.linalg.svd(grad_matrix, full_matrices=False)
        target_step_singular_values = singular_values.detach().cpu().numpy()
        print(f"Recorded gradient singular values at step {global_hook_call_count}!")

# Register the hook on the target parameter
for name, param in model.named_parameters():
    if name == layer_name:
        param.register_hook(lambda grad: check_singular_values((grad,)))
        print(f"Hook registered on parameter {name}.")

# ----------------------------
# 4. Set Training Arguments and Initialize Trainer
# ----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./t5_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=0.05,             # Train for 1 epoch
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",       # No evaluation in this example
    logging_steps=1,
    save_steps=1000,
    learning_rate=1e-5,
    weight_decay=0,
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# ----------------------------
# 5. Train and Plot the Singular Values from the Random Step's Gradient
# ----------------------------
trainer.train()

# If the singular values have been recorded, print and plot them
if target_step_singular_values is not None:
    max_sv = np.max(target_step_singular_values)
    print(f"Max singular value at target step {target_step} is: {max_sv}")

    plt.figure(figsize=(8, 5))
    plt.plot(target_step_singular_values, marker='o')
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.title(f"Singular Values of the Gradient at Step {target_step}")
    plt.grid(True)
    plt.savefig("./results/plots/random_step_singular_values.png")
    plt.show()
else:
    print("No singular values recorded!")