# data/data_utils.py
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from typing import Union

def load_and_preprocess_dataset(task_name: str, tokenizer: AutoTokenizer, max_length: int = 128, seed: int = 42) -> DatasetDict:
    """
    Loads and preprocesses the specified GLUE task (MRPC or QQP).

    Parameters:
      task_name (str): The GLUE task name ("mrpc" or "qqp").
      tokenizer (AutoTokenizer): A Hugging Face tokenizer.
      max_length (int): Maximum sequence length for tokenization.
      seed (int): Random seed for shuffling.

    Returns:
      DatasetDict: A dictionary with preprocessed "train", "validation", and "test" splits.
    """
    # Load the raw GLUE dataset (it will be cached automatically)
    raw_datasets = load_dataset("glue", task_name)

    # Define keys for sentence pairs based on task
    if task_name.lower() == "qqp":
        sentence1_key, sentence2_key = "question1", "question2"
    elif task_name.lower() == "mrpc":
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    # Define the preprocessing function
    def preprocess_function(examples: dict) -> dict:
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Apply preprocessing in batch mode
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Rename the label column to "labels" for consistency with Transformers
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Shuffle training and validation sets for randomness
    tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=seed)
    tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle(seed=seed)

    # If test set is not available (e.g., for QQP), you might consider splitting validation or returning None
    if "test" not in tokenized_datasets:
        # Optionally, you could split validation to create a test set.
        pass

    return tokenized_datasets