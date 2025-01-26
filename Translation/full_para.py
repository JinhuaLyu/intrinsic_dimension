import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)


# Check device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load dataset
raw_dataset = load_dataset("wmt14", "fr-en")
train_dataset = raw_dataset['train']
validation_dataset = raw_dataset['validation']

# Select smaller subsets for faster training
small_train = train_dataset.select(range(0, 10000)) 
small_val = validation_dataset.select(range(0, 1000))

# 2) Load tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 3) Preprocess dataset
def preprocess_function(examples):
    source_texts = [
        "translate English to French: " + translation["en"]
        for translation in examples["translation"]
    ]
    target_texts = [
        translation["fr"]
        for translation in examples["translation"]
    ]

    model_inputs = tokenizer(
        source_texts, max_length=128, truncation=True, padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_texts, max_length=128, truncation=True, padding="max_length"
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

processed_train = small_train.map(preprocess_function, batched=True, remove_columns=small_train.column_names)
processed_val = small_val.map(preprocess_function, batched=True, remove_columns=small_val.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# 4) Load base model
base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Ensure all parameters are trainable
for param in base_model.parameters():
    param.requires_grad = True


# 5) Training arguments
train_dataset_size = len(processed_train)  # Size of the training dataset
batch_size = 4  # Per-device training batch size
num_epochs = 10  # Total number of epochs

# Calculate steps per epoch and for 0.5 epoch
steps_per_epoch = train_dataset_size // batch_size
save_steps = int(steps_per_epoch * 0.5)

training_args = TrainingArguments(
    output_dir="./t5-translation-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",  # Evaluate every `eval_steps`
    eval_steps=save_steps,  # Evaluate every 0.5 epoch
    save_steps=save_steps,  # Save every 0.5 epoch
    logging_steps=500,
    learning_rate=1e-4,
    fp16=False,
    report_to="none"
)

# 6) Trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    data_collator=data_collator
)

# 7) Train
trainer.train()

# 8) Evaluate
base_model.eval()
