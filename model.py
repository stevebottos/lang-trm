from transformers import GPT2LMHeadModel
import torch
from torch import nn
from typing import List, Dict, Tuple
import layers
from dataclasses import dataclass

# This is the original just for posterity
# model_config = {
#     "batch_size": 1,
#     "seq_len": 256,
#     "vocab_size": 11,
#     "num_puzzle_identifiers": 0,  # All Sudoku puzzles share same embedding
#     "puzzle_emb_ndim": 0,  # no puzzle stuff
#     "H_cycles": 3,
#     "L_cycles": 6,
#     "H_layers": 0,
#     "L_layers": 2,
#     "hidden_size": 768,
#     "num_heads": 8,
#     "expansion": 4,
#     "pos_encodings": "rope",  # Just always use rope
#     "halt_max_steps": 8,
#     "halt_exploration_prob": 0.1,
#     "forward_dtype": "bfloat16",
#     "mlp_t": False,
#     "puzzle_emb_len": 0,  # No puzzle stuff
#     "no_ACT_continue": True,
# }

# These are fixed, taken from the values in the paper
HIDDEN_SIZE = 768
NUM_HEADS = 8
CAUSAL = False  # TODO: Correct? I think so.
SEQ_LEN = 256
ROPE_THETA = 10000.0
EXPANSION = 4
HALT_MAX_STEPS = 8
HALT_EXPLORATION_PROB = 0.1

H_CYCLES = 3
L_CYCLES = 6
H_LAYERS = 0  # They don't use h model in trm, just keeping this here to avoid confusion
L_LAYERS = 2


class LangTRM(nn.Module):
    def __init__(self):
        super().__init__()

        # Get the input embedding layer from gpt2 and tie weights. We
        # can later experiment with additional models.
        pretrained_language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.input_embedding = pretrained_language_model.transformer.wte

        # Freeze the pretrained embeddings
        self.input_embedding.weight.requires_grad = False

        n_vocab, dim = self.input_embedding.weight.shape
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.lm_head.weight = self.input_embedding.weight

        # Since weights are frozen and tied, lm_head shouldn't update them either
        self.lm_head.weight.requires_grad = False

        # Normalize before lm_head to match expected scale of pretrained weights
        # GPT-2's final layer produces std~7, so we scale LayerNorm output accordingly
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.final_scale = nn.Parameter(torch.ones(1) * 3)

        self.q_head = layers.CastedLinear(HIDDEN_SIZE, 2, bias=True)

        # Use learned positional embeddings instead of RoPE
        self.pos_embedding = nn.Embedding(SEQ_LEN, HIDDEN_SIZE)
        # Initialize with truncated normal
        with torch.no_grad():
            layers.trunc_normal_init_(self.pos_embedding.weight, std=0.02)

        # Reasoning Layers - use PyTorch's built-in TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_HEADS,
            dim_feedforward=int(HIDDEN_SIZE * EXPANSION),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-normalization (normalizes input to each layer)
        )
        self.L_level = nn.TransformerEncoder(encoder_layer, num_layers=L_LAYERS)

        self.learned_L_starter = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.learned_H_starter = nn.Parameter(torch.zeros(HIDDEN_SIZE))

    def forward(self, z_H, z_L, input, input_is_embedding=False):
        if not input_is_embedding:
            input_embeddings = self.input_embedding(input)
            # Add learned positional embeddings
            _, seq_len = input.shape
            positions = torch.arange(seq_len, device=input_embeddings.device)
            pos_emb = self.pos_embedding(positions).unsqueeze(
                0
            )  # (1, seq_len, hidden_size)
            input_embeddings = input_embeddings + pos_emb
        else:
            input_embeddings = input

        with torch.no_grad():
            for _ in range(H_CYCLES - 1):
                for _ in range(L_CYCLES):
                    # L processes reasoning with input from H and the original input
                    z_L = self.L_level(z_L + input_embeddings)
                # H processes the high-level state with input from L
                z_H = self.L_level(z_H + z_L)

        # 1 with grad
        for _ in range(L_CYCLES):
            z_L = self.L_level(z_L + input_embeddings)
        z_H = self.L_level(z_H + z_L)

        # LM Outputs - normalize and scale to match pretrained weight expectations
        # GPT-2 expects inputs with std~7, not std~1 from LayerNorm
        embeddings = self.final_norm(z_H) * self.final_scale
        output = self.lm_head(embeddings)
        return output, z_H, z_L, embeddings
