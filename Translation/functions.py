from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import torch
from torch.nn.utils.rnn import pad_sequence

model_name = "t5-small"

def preprocess_function(examples):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
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

# Custom collate function to convert input_ids into a full tensor
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    labels = torch.tensor([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params