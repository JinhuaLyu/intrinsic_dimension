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
import functions
import matplotlib.pyplot as plt
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 0) Hyperparameters
learning_rates = [1e-4, 3e-4]   # Learning rates to try
num_epochs = 10                             # Total number of epochs

save_dir = "./results/plots"
os.makedirs(save_dir, exist_ok=True)
csv_dir = "./results/csv"
os.makedirs(csv_dir, exist_ok=True)


# 1) Load dataset
raw_dataset = load_dataset("wmt14", "fr-en")
train_dataset = raw_dataset['train']
validation_dataset = raw_dataset['validation']
test_dataset = raw_dataset['test'] # Not used in this example, but you can use it for testing
# Shuffle datasets
train_dataset = train_dataset.shuffle(seed=42)
validation_dataset = validation_dataset.shuffle(seed=42)
# Select smaller subsets for faster training
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
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
peft_model = get_peft_model(base_model, peft_config)

trainable_params, total_params = functions.count_trainable_parameters(peft_model)
print(f"Trainable Parameters: {trainable_params}")
print(f"Total Parameters: {total_params}")
print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")

# Calculate steps per epoch and for 0.5 epoch
train_dataset_size = len(processed_train)  # Size of the training dataset
batch_size = 4                              # Per-device training batch size
steps_per_epoch = train_dataset_size // batch_size
save_steps = int(steps_per_epoch * 0.5)




# 6) Running loops
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    training_args = TrainingArguments(
        output_dir=f"./lora-t5-translation-checkpoints_{lr}",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",  # Evaluate every eval_steps
        eval_steps=save_steps,        # Evaluate every 0.5 epoch
        save_steps=save_steps,        # Save every 0.5 epoch
        logging_steps=500,
        fp16=False,
        learning_rate=lr,
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

    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_steps, eval_losses   = [], []

    # Extract training/evaluation losses
    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    ###############################################################################
    #             Save Training Loss to a File (CSV) Including LR in Filename     #
    ###############################################################################
    # Save training loss
    csv_filename_train = os.path.join(csv_dir, f"lora_training_loss_lr_{lr}.csv")
    with open(csv_filename_train, "w") as f:
        f.write("step,training_loss\n")
        for step, loss in zip(train_steps, train_losses):
            f.write(f"{step},{loss}\n")

    # Save evaluation loss
    csv_filename_eval = os.path.join(csv_dir, f"lora_evaluation_loss_lr_{lr}.csv")
    with open(csv_filename_eval, "w") as f:
        f.write("step,evaluation_loss\n")
        for step, loss in zip(eval_steps, eval_losses):
            f.write(f"{step},{loss}\n")

    ###############################################################################
    #            Plot Training and Evaluation Loss Including Learning Rate        #
    ###############################################################################
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Training Loss")
    plt.plot(eval_steps, eval_losses, label="Evaluation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Lora Training and Evaluation Loss (LR={lr})")
    plt.legend()
    plt.grid(True)

    # Save the figure to a PNG file (with LR in the filename)
    plot_filename = os.path.join(save_dir,f"loss_curves_lr_{lr}.png")
    plt.savefig(plot_filename)
    plt.show()


    # 8) Evaluate
    peft_model.eval()