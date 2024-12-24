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

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1) Load dataset
raw_dataset = load_dataset("opus_books", "en-fr")
train_dataset = raw_dataset["train"]
validation_size = 1000
small_val = train_dataset.select(range(0, validation_size)) 
small_train = train_dataset.select(range(validation_size, len(train_dataset)))  


# 2) Load tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 3) Preprocess
def preprocess_function(examples):
    # Handle batched inputs: examples["translation"] is a list of dictionaries
    source_texts = [
        "translate English to French: " + translation["en"]
        for translation in examples["translation"]
    ]
    target_texts = [
        translation["fr"]
        for translation in examples["translation"]
    ]
    
    # Tokenize source and target texts
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

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# 4) Load base model
base_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# 5) Create LoRA configuration and wrap model
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
peft_model = get_peft_model(base_model, peft_config)

# 6) Training arguments
training_args = TrainingArguments(
    output_dir="./lora-t5-translation-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    fp16=False,
    learning_rate=1e-4,
    report_to="none"
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    data_collator=data_collator
)

# 7) Train
trainer.train()

# 8) Evaluate
peft_model.eval()
test_texts = [
    "translate English to French: This is a test sentence.",
    "translate English to French: I love fine-tuning with LoRA."
]
inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True).to(peft_model.device)
with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_length=50)
translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for inp, outp in zip(test_texts, translated_texts):
    print(f"Input: {inp}")
    print(f"Output: {outp}")
    print("---")