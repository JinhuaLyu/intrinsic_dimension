# training/trainer_utils.py
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, Any

def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """
    Compute accuracy and F1 score based on model predictions.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

class MyTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        # Use safe_serialization=False to avoid issues with shared tensors.
        self.model.save_pretrained(output_dir, safe_serialization=False)
        # Also save tokenizer if available.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def get_trainer(
    model: torch.nn.Module,
    tokenizer,
    train_dataset,
    eval_dataset,
    custom_optimizer,
    training_config: Dict[str, Any]
) -> MyTrainer:
    """
    Initializes the Trainer object.
    """
    print("--------------------Debug----------------------")
    print(training_config.get("load_best_model_at_end", False))
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
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_accuracy"),
        greater_is_better=training_config.get("greater_is_better", False),
        overwrite_output_dir=training_config.get("overwrite_output_dir", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4)
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set early stopping callback: stop training if no improvement for 5 evaluations.
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=training_config.get("early_stop", 5))

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(custom_optimizer, None),
        callbacks=[early_stopping_callback]
    )
    return trainer