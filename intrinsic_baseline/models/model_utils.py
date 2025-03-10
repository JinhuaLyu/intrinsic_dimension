# models/model_utils.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .low_dim_projection import LowDimWeightWrapper
import numpy as np

def build_model(model_name: str, num_labels: int = 2):
    """
    Loads a pre-trained model and tokenizer for classification tasks.
    
    Args:
        model_name (str): Pre-trained model identifier.
        num_labels (int): Number of output labels.
    
    Returns:
        model: The pre-trained model.
        tokenizer: The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def replace_with_low_dim_params(model, trainable_layers, train_mode: str, d: int, seed: int, projection: str = "fastfood"):
    """
    Replaces the linear layers in specified transformer layers with low-dimensional parameterizations.
    This function performs layer-wise replacement, where each target layer's adjustable weight is replaced 
    by a wrapper that computes an update from a low-dimensional parameter vector using the specified projection method.
    
    Args:
        model: Transformer model.
        trainable_layers: List of layer indices to modify (e.g., [0, 2, 4]) or "ALL" for all layers.
        train_mode (str): Specifies which parts to replace ("all", "attention_matrices", "ffn_matrices").
        d (int): Dimension of the low-dimensional space for each replaced layer.
        seed (int): Random seed for reproducibility.
        projection (str): Projection method to use ("linear" or "fastfood").
    
    Returns:
        model: Modified model with selected layers replaced by low-dimensional wrappers.
    """
    # Freeze all parameters in the model.
    for param in model.parameters():
        param.requires_grad = False

    # Identify the transformer backbone by checking common attribute names.
    transformer_backbone = None
    for name in ["bert", "roberta", "distilbert", "electra", "xlm"]:
        if hasattr(model, name):
            transformer_backbone = getattr(model, name)
            break
    if transformer_backbone is None:
        raise ValueError("No supported transformer backbone found in the model.")

    # Locate the encoder layers.
    if hasattr(transformer_backbone, "encoder"):
        layers = transformer_backbone.encoder.layer
    elif hasattr(transformer_backbone, "transformer"):
        layers = transformer_backbone.transformer.layer
    else:
        raise ValueError("Unable to find encoder layers in the transformer backbone.")

    # If trainable_layers is "ALL", select all layers.
    if trainable_layers == "ALL":
        trainable_layers = list(range(len(layers)))

    # Replace specified layers with low-dimensional wrappers.
    for layer_idx in trainable_layers:
        layer = layers[layer_idx]

        if train_mode in ["all", "attention_matrices"]:
            # Replace attention-related linear layers.
            layer.attention.self.query = LowDimWeightWrapper(
                layer.attention.self.query, d, seed + layer_idx * 10 + 1, projection=projection)
            layer.attention.self.key = LowDimWeightWrapper(
                layer.attention.self.key, d, seed + layer_idx * 10 + 2, projection=projection)
            layer.attention.self.value = LowDimWeightWrapper(
                layer.attention.self.value, d, seed + layer_idx * 10 + 3, projection=projection)
            layer.attention.output.dense = LowDimWeightWrapper(
                layer.attention.output.dense, d, seed + layer_idx * 10 + 4, projection=projection)

        if train_mode in ["all", "ffn_matrices"]:
            # Replace feed-forward network (FFN) linear layers.
            layer.intermediate.dense = LowDimWeightWrapper(
                layer.intermediate.dense, d, seed + layer_idx * 10 + 5, projection=projection)
            layer.output.dense = LowDimWeightWrapper(
                layer.output.dense, d, seed + layer_idx * 10 + 6, projection=projection)

    return model

def apply_intrinsic_dimension(model, intrinsic_dimension: int, output_dir: str, str_filter=None, projection: str = "fastfood", device: str = "cuda"):
    """
    Applies a global intrinsic dimension reduction on the model's adjustable parameters.
    
    Instead of replacing each layer individually with its own low-dimensional parameterization, 
    this function extracts all trainable parameters (adjustable parameters) across the model,
    concatenates them into a single vector, and then uses a projection (e.g., FastFood transform)
    to compute an update from a global low-dimensional intrinsic vector of size `intrinsic_dimension`.
    This global update is then split and applied to the respective parameters.
    
    Args:
        model: The transformer model.
        intrinsic_dimension (int): The overall low-dimensional space dimension for the entire model.
        output_dir (str): Directory to save any related files (if needed).
        str_filter (set, optional): A set of strings to filter which parameters to include; if empty, all adjustable parameters are used.
        projection (str): Projection method to use ("fastfood" or "random").
        device (str): Device to use ("cuda" or "cpu").
    
    Returns:
        model: The modified model with a global intrinsic dimension hook applied.
    """
    # Import the intrinsic dimension hook from another module.
    # Ensure you have a corresponding file (e.g., models/intrinsic_dimension.py) implementing this functionality.
    from .intrinsic_dimension import intrinsic_dimension as apply_intrinsic_dimension_hook
    if str_filter is None:
        str_filter = set()
    model = apply_intrinsic_dimension_hook(model, intrinsic_dimension, output_dir, str_filter, projection, device)
    return model