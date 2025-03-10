# training/trainer_utils.py
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, Any

def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """
    Compute accuracy and F1 score based on model predictions.

    Args:
        eval_pred (tuple): A tuple (logits, labels).

    Returns:
        dict: A dictionary with keys "accuracy" and "f1".
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

def get_trainer(
    model: torch.nn.Module,
    tokenizer,
    train_dataset,
    eval_dataset,
    custom_optimizer,
    training_config: Dict[str, Any]
) -> Trainer:
    """
    Initializes the Trainer object.

    Parameters:
      model (torch.nn.Module): The model to train.
      tokenizer: The tokenizer used for processing data.
      train_dataset: Training dataset.
      eval_dataset: Evaluation dataset.
      custom_optimizer: Pre-built optimizer to update trainable parameters.
      training_config (dict): Dictionary containing training hyperparameters.

    Returns:
      Trainer: A Hugging Face Trainer instance.
    """
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        evaluation_strategy="epoch",
        save_strategy=training_config["save_strategy"],
        learning_rate=float(training_config["learning_rate"]),
        logging_strategy="epoch",
        report_to="none",
        seed=training_config["seed"],
        warmup_steps=training_config.get("warmup_steps", 0),
        save_total_limit=training_config.get("save_total_limit", None),
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        overwrite_output_dir=training_config.get("overwrite_output_dir", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4)
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(custom_optimizer, None)
    )
    return trainer