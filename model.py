from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import nn
from typing import List
import layers

# This is
model_config = {
    "batch_size": 1,
    "seq_len": 256,
    "vocab_size": 11,
    "num_puzzle_identifiers": 0,  # All Sudoku puzzles share same embedding
    "puzzle_emb_ndim": 0,  # no puzzle stuff
    "H_cycles": 3,
    "L_cycles": 6,
    "H_layers": 0,
    "L_layers": 2,
    "hidden_size": 768,
    "num_heads": 8,
    "expansion": 4,
    "pos_encodings": "rope",  # Just always use rope
    "halt_max_steps": 8,
    "halt_exploration_prob": 0.1,
    "forward_dtype": "bfloat16",
    "mlp_t": False,
    "puzzle_emb_len": 0,  # No puzzle stuff
    "no_ACT_continue": True,
}

# These are fixed, taken from the values in the paper
HIDDEN_SIZE = 768
NUM_HEADS = 8
CAUSAL = False  # TODO: Correct?
SEQ_LEN = 256
ROPE_THETA = 10000.0
EXPANSION = 4
L_LAYERS = 2
K_STEPS = 8  # Explicitly say that we want k-steps, no early stopping, can be really easily added later


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = (
            layers.Attention(
                hidden_size=HIDDEN_SIZE,
                head_dim=HIDDEN_SIZE // NUM_HEADS,
                num_heads=NUM_HEADS,
                num_key_value_heads=NUM_HEADS,
                causal=CAUSAL,
            ),
        )

        self.mlp = layers.SwiGLU(
            hidden_size=HIDDEN_SIZE,
            expansion=EXPANSION,
        )

        self.norm_eps = 1e-5

    def forward(
        self, cos_sin: layers.CosSin, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = layers.rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),  # type: ignore
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = layers.rms_norm(
            hidden_states + out, variance_epsilon=self.norm_eps
        )
        return hidden_states


class MainStack(nn.Module):
    def __init__(self, layers: List[BasicBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class LangTRM(nn.Module):
    def __init__(self):
        super().__init__()

        # Get the input embedding layer from gpt2 and tie weights. We
        # can later experiment with additional models.
        pretrained_language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.input_embedding = pretrained_language_model.transformer.wte

        n_vocab, dim = self.input_embedding.weight.shape
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.lm_head.weight = self.input_embedding.weight

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rotary_emb = layers.RotaryEmbedding(
            dim=HIDDEN_SIZE // NUM_HEADS,
            max_position_embeddings=SEQ_LEN,
            base=ROPE_THETA,
        )

        # Reasoning Layers
        self.L_level = MainStack(layers=[BasicBlock() for _ in range(L_LAYERS)])

        # Initial states
        self.H_init = nn.Buffer(
            layers.trunc_normal_init_(torch.empty(HIDDEN_SIZE), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            layers.trunc_normal_init_(torch.empty(HIDDEN_SIZE), std=1),
            persistent=True,
        )

    def forward(self, input_text_ids): ...
