from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import f1_score

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # binary classification
raw_datasets = load_dataset("imdb")  # For demonstration

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512) # adjust max_length as needed

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

## Train the model
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"  # Set to "wandb" or "tensorboard" if you want logging
)

trainer = Trainer(
    model= peft_model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()
print(trainer.optimizer)

print("Allocated memory (MB):", torch.mps.current_allocated_memory() / (1024 * 1024))

predictions = trainer.predict(eval_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
f1 = f1_score(predictions.label_ids, preds, average="binary")
print("F1 Score (sklearn):", f1)


## Evaluate the model
print("Evaluate the model")
metrics = trainer.evaluate()
print(metrics)

# ## Run predictions on new data
# inputs = tokenizer("This movie was absolutely wonderful!", return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}
# outputs = peft_model(**inputs)
# predictions = outputs.logits.argmax(dim=-1)
# print("Predicted label:", predictions.item())

# Save the LoRA-modified model
peft_model.save_pretrained("lora_distilbert_imdb")

