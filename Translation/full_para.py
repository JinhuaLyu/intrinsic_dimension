import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AdamW
)
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import functions

# Hyperparameters
learning_rate = 1e-5
batch_size = 16  # Per-device training batch size
num_epochs = 20  # Total number of epochs
frozen_layers = "em_mlp"
output_dir=f"./t5_checkpoints/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}_frozen_{frozen_layers}"


save_dir = "./results/plots"
os.makedirs(save_dir, exist_ok=True)
csv_dir = "./results/csv"
os.makedirs(csv_dir, exist_ok=True)
plot_filename = os.path.join(save_dir,f"full_loss_curves_lr_{learning_rate}_bs_{batch_size}_frozen_{frozen_layers}.png")
train_filename = os.path.join(csv_dir, f"full_training_loss_lr_{learning_rate}_bs_{batch_size}_frozen_{frozen_layers}.csv")
eval_filename = os.path.join(csv_dir, f"full_evaluation_loss_lr_{learning_rate}_bs_{batch_size}_frozen_{frozen_layers}.csv")

# Set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Check device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 1) Load dataset
raw_dataset = load_dataset("wmt14", "fr-en")
train_dataset = raw_dataset["train"]
validation_dataset = raw_dataset["validation"]
# Shuffle datasets
train_dataset = train_dataset.shuffle(seed=42)
validation_dataset = validation_dataset.shuffle(seed=42)
# Select smaller subsets for faster training
small_train = train_dataset.select(range(0, 10000))
small_val = validation_dataset.select(range(0, 1000))

# 2) Load tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 3) Preprocess dataset
processed_train = small_train.map(
    functions.preprocess_function, 
    batched=True, 
    remove_columns=small_train.column_names
)
processed_val = small_val.map(
    functions.preprocess_function, 
    batched=True, 
    remove_columns=small_val.column_names
)
train_dataset_size = len(processed_train)  # Size of the training dataset
# Calculate total training steps
total_train_steps = (train_dataset_size // batch_size) * num_epochs
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# 4) Load base model
base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Freeze embeddings and first few layers to prevent overfitting
for name, param in base_model.named_parameters():
    if "shared" in name or "encoder.block.0" in name or "DenseReluDense" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True  # Ensure other parameters (Attention layers) are trainable

# Confirm which layers are frozen
for name, param in base_model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 5) Training arguments
# Calculate warm-up steps (e.g., 10% of total training steps)
warmup_steps = int(0.1 * total_train_steps)
# Calculate steps per epoch
steps_per_epoch = train_dataset_size // batch_size
save_steps = int(steps_per_epoch * 0.5)+1  # Save model every 0.5 epoch
logging_steps = int(steps_per_epoch * 0.1)  # Log every 0.1 epoch

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=500,
    learning_rate=learning_rate,  # Peak learning rate
    warmup_steps=warmup_steps,  # Warm-up for first 10% of steps
    lr_scheduler_type="cosine",  # Cosine learning rate decay
    fp16=False,
    report_to="none",
    seed=42,
    weight_decay=0.01
)

# 6) Custom Optimizer (AdamW)
optimizer = AdamW(base_model.parameters(), lr=learning_rate, weight_decay=0.01)

# Trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    data_collator=data_collator
)

# 7) Train
trainer.train()

# 8) Extract loss history
log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

# Extract training/evaluation losses
for entry in log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

# 9) Save Training Loss to CSV
with open(train_filename, "w") as f:
    f.write("step,training_loss\n")
    for step, loss in zip(train_steps, train_losses):
        f.write(f"{step},{loss}\n")

with open(eval_filename, "w") as f:
    f.write("step,evaluation_loss\n")
    for step, loss in zip(eval_steps, eval_losses):
        f.write(f"{step},{loss}\n")

# 10) Plot Training and Evaluation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Training Loss")
plt.plot(eval_steps, eval_losses, label="Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(f"Training and Evaluation Loss (LR={learning_rate})")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(plot_filename)
plt.show()

# 11) Evaluate
base_model.eval()