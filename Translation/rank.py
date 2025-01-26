import torch
from transformers import T5ForConditionalGeneration

# Step 1: Load the "before-training" model
model_name = "t5-small"
model_before = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 2: Load the "after-training" model
output_dir = "./t5-translation-checkpoints/checkpoint-7500"
model_after = T5ForConditionalGeneration.from_pretrained(output_dir)

# Threshold to zero out small entries in the difference
threshold = 1e-5

# A list to store information about matrices with rank < 200
low_rank_matrices = []

# ------------------------------------------------------
# Compare Q, K, V, O projection matrices in all Encoder blocks
# ------------------------------------------------------
for layer_idx, (layer_before, layer_after) in enumerate(zip(model_before.encoder.block,
                                                           model_after.encoder.block)):
    # Each T5 block has a SelfAttention sub-layer at index 0
    self_attn_before = layer_before.layer[0].SelfAttention
    self_attn_after  = layer_after.layer[0].SelfAttention
    
    # Compare Q, K, V, O
    for proj_name in ["q", "k", "v", "o"]:
        # Get weights before training
        weight_before = getattr(self_attn_before, proj_name).weight.data.clone()
        # Get weights after training
        weight_after  = getattr(self_attn_after, proj_name).weight.data.clone()
        
        # Compute the difference (before - after)
        diff = weight_before - weight_after
        
        # Zero out entries with absolute value smaller than threshold
        diff[diff.abs() < threshold] = 0
        
        # Compute the rank of the difference matrix
        rank = torch.linalg.matrix_rank(diff).item()
        
        # Store if rank < 200
        if rank < 200:
            low_rank_matrices.append((f"encoder_block_{layer_idx}_{proj_name}", diff, rank))

# ------------------------------------------------------
# Compare Q, K, V, O projection matrices in all Decoder blocks
# ------------------------------------------------------
for layer_idx, (layer_before, layer_after) in enumerate(zip(model_before.decoder.block,
                                                           model_after.decoder.block)):
    # 1) Self-Attention sub-layer is at index 0 in T5 decoder block
    self_attn_before = layer_before.layer[0].SelfAttention
    self_attn_after  = layer_after.layer[0].SelfAttention
    
    for proj_name in ["q", "k", "v", "o"]:
        weight_before = getattr(self_attn_before, proj_name).weight.data.clone()
        weight_after  = getattr(self_attn_after, proj_name).weight.data.clone()
        diff = weight_before - weight_after
        diff[diff.abs() < threshold] = 0
        rank = torch.linalg.matrix_rank(diff).item()
        
        if rank < 512:
            low_rank_matrices.append((f"decoder_block_{layer_idx}_self_{proj_name}", diff, rank))
    
    # 2) Cross-Attention (EncDecAttention) sub-layer is typically at index 1 in T5 decoder block
    cross_attn_before = layer_before.layer[1].EncDecAttention
    cross_attn_after  = layer_after.layer[1].EncDecAttention
    
    for proj_name in ["q", "k", "v", "o"]:
        weight_before = getattr(cross_attn_before, proj_name).weight.data.clone()
        weight_after  = getattr(cross_attn_after, proj_name).weight.data.clone()
        diff = weight_before - weight_after
        diff[diff.abs() < threshold] = 0
        rank = torch.linalg.matrix_rank(diff).item()
        
        if rank < 512:
            low_rank_matrices.append((f"decoder_block_{layer_idx}_cross_{proj_name}", diff, rank))

# Print all matrices that have rank < 200
for name, mat, rank in low_rank_matrices:
    print(f"{name}: rank = {rank}, shape = {mat.shape}")