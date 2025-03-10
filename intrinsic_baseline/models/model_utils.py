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
        model: Pre-trained model.
        tokenizer: Corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def replace_with_low_dim_params(model, trainable_layers, train_mode: str, d: int, seed: int, projection: str = "linear"):
    """
    Replaces the linear layers in specified transformer layers with a low-dimensional parameterization.
    
    Args:
        model: Transformer model.
        trainable_layers: List of layer indices to modify (e.g., [0, 2, 4]) or "ALL" for all layers.
        train_mode (str): Specifies which parts to replace ("all", "attention_matrices", "ffn_matrices").
        d (int): Dimension of the low-dimensional space.
        seed (int): Random seed for reproducibility.
        projection (str): Projection method, either "linear" (default) or "fastfood".
    
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