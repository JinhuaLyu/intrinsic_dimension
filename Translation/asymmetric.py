import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model
import functions
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# 0) Hyperparameters
learning_rate = 1e-4
num_epochs = 10

save_dir = "./results/plots"
os.makedirs(save_dir, exist_ok=True)
csv_dir = "./results/csv"
os.makedirs(csv_dir, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# 1) Load dataset
raw_dataset = load_dataset("wmt14", "fr-en")
train_dataset = raw_dataset['train']
validation_dataset = raw_dataset['validation']
train_dataset = train_dataset.shuffle(seed=42)
validation_dataset = validation_dataset.shuffle(seed=42)
small_train = train_dataset.select(range(0, 10000))
small_val = validation_dataset.select(range(0, 1000))

# 2) Load tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 3) Preprocess dataset
processed_train = small_train.map(functions.preprocess_function, batched=True, remove_columns=small_train.column_names)
processed_val = small_val.map(functions.preprocess_function, batched=True, remove_columns=small_val.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# 4) Load base model
base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# 5) Create LoRA configuration and wrap model
target_modules = ["q", "k", "v", "o"]
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
peft_model = get_peft_model(base_model, peft_config)

# 6) Freeze matrix B (keep A trainable)
for name, param in peft_model.named_parameters():
    if "lora_B" in name:  # Freeze B
        param.requires_grad = False
    if "lora_A" in name:  # Ensure A is trainable
        param.requires_grad = True

trainable_params, total_params = functions.count_trainable_parameters(peft_model)
print(f"Trainable Parameters: {trainable_params}")
print(f"Total Parameters: {total_params}")
print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")

train_dataset_size = len(processed_train)
batch_size = 4
steps_per_epoch = train_dataset_size // batch_size
save_steps = int(steps_per_epoch * 0.5)

# 7) Training Loop
print(f"Training with learning rate: {learning_rate}")
training_args = TrainingArguments(
    output_dir=f"./lora_A_checkpoints/lora-t5-A-only_{learning_rate}",
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=500,
    fp16=False,
    learning_rate=learning_rate,
    report_to="none"
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    data_collator=data_collator
)

trainer.train()

log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="Training Loss")
plt.plot(eval_steps, eval_losses, label="Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(f"LoRA (A-only) Training Loss (LR={learning_rate})")
plt.legend()
plt.grid(True)

plot_filename = os.path.join(save_dir, f"lora_A_only_loss_lr_{learning_rate}.png")
plt.savefig(plot_filename)
plt.show()

peft_model.eval()
