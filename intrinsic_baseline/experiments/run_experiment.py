# experiments/run_experiment.py
import os
import yaml
import torch
from transformers import AdamW
from utils.reproducibility import set_seed, get_logger
from data.data_utils import load_and_preprocess_dataset
from models.model_utils import build_model, replace_with_low_dim_params
from training.trainer_utils import get_trainer
import argparse

logger = get_logger(__name__)

def main():
    # Parse command-line arguments to allow a custom config file
    parser = argparse.ArgumentParser(description="Run experiment with specified config.")
    parser.add_argument("--config_path", type=str, default=os.path.join(os.path.dirname(__file__), "../configs/config.yaml"),
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    print("Using config file:", args.config_path)
    # Load configuration file from the given path
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set the random seed for reproducibility
    seed = config["training"]["seed"]
    set_seed(seed)
    d = config["training"]["global_intrinsic_dimension"]
    lr = config["training"]["learning_rate"]

    # Create the output directory to store checkpoints, logs, and results
    output_dir = config["output"]["output_dir"]
    print("output_dir:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build the pre-trained model and corresponding tokenizer
    model, tokenizer = build_model(
        config["model"]["model_name_or_path"], 
        num_labels=config["model"]["num_labels"]
    )

    training_config = config["training"].copy()
    training_config["output_dir"] = output_dir

    # Decide whether to apply layerwise replacement or global intrinsic dimension reduction.
    intrinsic_mode = config["training"].get("intrinsic_mode", "layerwise")
    if intrinsic_mode == "global":
        global_intrinsic_dim = config["training"].get("global_intrinsic_dimension", None)
        if global_intrinsic_dim is None:
            raise ValueError("For global intrinsic mode, please set 'global_intrinsic_dimension' in your config.")
        from models.intrinsic_dimension import intrinsic_dimension
        model = intrinsic_dimension(
            model, 
            intrinsic_dimension=global_intrinsic_dim,
            output_dir=output_dir,
            training_config = training_config,
            projection="global",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Applied global intrinsic dimension reduction with dimension: {}".format(global_intrinsic_dim))
    else:
        model = replace_with_low_dim_params(
            model,
            trainable_layers=config["training"]["trainable_layers"],
            train_mode=config["training"]["train_mode"],
            d=config["training"]["d"],
            seed=seed,
            projection=config["training"].get("projection", "fastfood")
        )
        logger.info("Applied layerwise low-dimensional replacement with d: {}".format(config["training"]["d"]))

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
        lr=float(config["training"]["learning_rate"])
    )

    # Initialize the Trainer with the model, datasets, optimizer, and training hyperparameters

    trainer = get_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        optimizer,
        training_config=training_config
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
    results_file = os.path.join(output_dir, f"results_d_{d}_lr_{lr}.yaml")
    with open(results_file, "w") as f:
        yaml.dump({"eval": eval_metrics, "test": test_metrics}, f)

if __name__ == "__main__":
    main()