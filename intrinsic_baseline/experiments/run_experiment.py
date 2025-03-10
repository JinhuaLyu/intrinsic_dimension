# experiments/run_experiment.py
import os
import yaml
import torch
from transformers import AdamW
from utils.reproducibility import set_seed, get_logger
from data.data_utils import load_and_preprocess_dataset
from models.model_utils import build_model, replace_with_low_dim_params
from training.trainer_utils import get_trainer

# Create a logger for the experiment
logger = get_logger(__name__)

def main():
    # Load configuration file from the configs directory
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set the random seed for reproducibility
    seed = config["training"]["seed"]
    set_seed(seed)

    # Create the output directory to store checkpoints, logs, and results
    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Build the pre-trained model and corresponding tokenizer
    model, tokenizer = build_model(
        config["model"]["model_name_or_path"], 
        num_labels=config["model"]["num_labels"]
    )

    # Replace selected layers of the model with low-dimensional parameterizations
    model = replace_with_low_dim_params(
        model,
        trainable_layers=config["training"]["trainable_layers"],
        train_mode=config["training"]["train_mode"],
        d=config["training"]["d"],
        seed=seed,
        projection=config["training"].get("projection", "linear")  # Allow projection selection via config
    )

    # Print parameter statistics for debugging and comparison
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Load and preprocess the dataset based on the specified GLUE task (e.g., MRPC)
    tokenized_datasets = load_and_preprocess_dataset(
        config["data"]["task_name"],
        tokenizer,
        max_length=config["data"]["max_length"],
        seed=seed
    )
    # Select dataset splits
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # Define the optimizer to update only trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["training"]["learning_rate"]
    )

    # Initialize the Trainer with the model, datasets, optimizer, and training hyperparameters
    trainer = get_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        optimizer,
        training_config={
            "output_dir": output_dir,
            "num_train_epochs": config["training"]["num_train_epochs"],
            "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
            "per_device_eval_batch_size": config["training"]["per_device_eval_batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "save_strategy": config["training"]["save_strategy"],
            "seed": seed
        }
    )

    # Ensure the model is on the correct device (GPU if available) before training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer.train()

    # Evaluate the model on the evaluation and test datasets after training
    eval_metrics = trainer.evaluate(eval_dataset)
    test_metrics = trainer.evaluate(test_dataset)
    logger.info(f"Evaluation Metrics: {eval_metrics}")
    logger.info(f"Test Metrics: {test_metrics}")

    # Save the evaluation and test results to a YAML file in the output directory
    results_file = os.path.join(output_dir, "results.yaml")
    with open(results_file, "w") as f:
        yaml.dump({"eval": eval_metrics, "test": test_metrics}, f)

if __name__ == "__main__":
    main()