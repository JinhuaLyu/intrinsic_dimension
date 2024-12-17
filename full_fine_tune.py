from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # binary classification

# Load dataset
raw_datasets = load_dataset("imdb")

# Preprocess the dataset
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = raw_datasets.map(preprocess, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))  # Subset for demo
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# Set device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Training arguments
training_args = TrainingArguments(
    output_dir="./full_finetune_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    logging_steps=50,
    learning_rate=2e-5,  # Lower learning rate for full fine-tuning
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"  # No external reporting
)

# Trainer
trainer = Trainer(
    model=model.to(device),  # Use original model directly
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
print("Evaluate the model")
metrics = trainer.evaluate()
print(metrics)

# Save the fully fine-tuned model
model.save_pretrained("full_finetune_distilbert_imdb")
tokenizer.save_pretrained("full_finetune_distilbert_imdb")
