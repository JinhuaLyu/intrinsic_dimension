import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import random, numpy as np, matplotlib.pyplot as plt, os, functions

# Hyperparameters
learning_rate, batch_size, num_epochs, weight_decay = 1e-5, 16, 10, 0
layer_name = "encoder.block.0.layer.1.DenseReluDense.wi.weight"
# Extract short name "b5l1v": "b" + block number + "l" + layer number + the parameter letter
parts = layer_name.split('.')  # parts = ["decoder", "block", "5", "layer", "1", "EncDecAttention", "v", "weight"]
layer_short = parts[0] + "b" + parts[2] + "l" + parts[4] + parts[6]
output_dir = f"./t5_checkpoints_grank/t5-translation-checkpoints-lr_{learning_rate}_bs_{batch_size}_{layer_short}"

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

# Load model and data collator
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

# Hook to capture gradients and check for low-rank
low_rank_info = []

def check_low_rank(grad_output):
    grad_matrix = grad_output[0].detach().cpu().numpy()
    rank = np.linalg.matrix_rank(grad_matrix)
    min_dim = min(grad_matrix.shape)
    is_low_rank = rank < (0.7 * min_dim)
    low_rank_info.append((rank, is_low_rank))

for name, param in model.named_parameters():
    if name == layer_name:
        param.register_hook(lambda grad: check_low_rank((grad,)))

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size, per_device_eval_batch_size=4,
    evaluation_strategy="steps", eval_steps=250, save_steps=250, logging_steps=500,
    learning_rate=learning_rate, weight_decay=weight_decay, report_to="none", seed=42
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=data_collator
)
trainer.train()

# Extract and save losses
log_history = trainer.state.log_history
train_losses = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
eval_losses = [(entry["step"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]

np.savetxt(f"./results/csv/training_loss_grank_{layer_short}.csv", train_losses, delimiter=",", header="step,training_loss", comments="")
np.savetxt(f"./results/csv/evaluation_loss_grank_{layer_short}.csv", eval_losses, delimiter=",", header="step,evaluation_loss", comments="")

with open(f"./results/csv/low_rank_info_grank_{layer_short}.csv", "w") as f:
    f.write("step,rank,is_low_rank\n")
    for idx, (rank, is_low_rank) in enumerate(low_rank_info):
        f.write(f"{idx},{rank},{is_low_rank}\n")

plt.figure(figsize=(8, 5))
plt.plot(*zip(*train_losses), label="Training Loss")
plt.plot(*zip(*eval_losses), label="Evaluation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(f"Training & Evaluation Loss (LR={learning_rate})")
plt.legend()
plt.grid(True)
plt.savefig(f"./results/plots/loss_curve_grank_{layer_short}.png")
plt.show()

model.eval()