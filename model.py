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


@dataclass
class States:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class Carry:
    inner_carry: States
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = layers.Attention(
            hidden_size=HIDDEN_SIZE,
            head_dim=HIDDEN_SIZE // NUM_HEADS,
            num_heads=NUM_HEADS,
            num_key_value_heads=NUM_HEADS,
            causal=CAUSAL,
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


class LangTRMInnerModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Get the input embedding layer from gpt2 and tie weights. We
        # can later experiment with additional models.
        pretrained_language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.input_embedding = pretrained_language_model.transformer.wte

        n_vocab, dim = self.input_embedding.weight.shape
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.lm_head.weight = self.input_embedding.weight

        self.q_head = layers.CastedLinear(HIDDEN_SIZE, 2, bias=True)

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

    def empty_carry(self, batch_size: int, device):
        return States(
            z_H=torch.empty(
                batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
            ),
            z_L=torch.empty(
                batch_size, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: States,
    ):
        return States(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: States,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        States,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        seq_info = dict(cos_sin=self.rotary_emb())

        # This is the pseudocode in the paper
        input_embeddings = self.input_embedding(batch["inputs"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(H_CYCLES - 1):
                for _ in range(L_CYCLES):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # 1 with grad
        for _ in range(L_CYCLES):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        print(z_H.shape)

        # LM Outputs
        new_carry = States(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class LangTRM(nn.Module):
    """ACT wrapper."""

    def __init__(self):
        super().__init__()
        self.inner = LangTRMInnerModule()

    def initial_carry(self, batch: Dict[str, torch.Tensor], device):
        batch_size = batch["inputs"].shape[0]

        return Carry(
            inner_carry=self.inner.empty_carry(
                batch_size, device
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones(
                (batch_size,), dtype=torch.bool, device=device
            ),  # Default to halted
            current_data={
                k: torch.empty_like(v, device=device) for k, v in batch.items()
            },
        )

    def forward(
        self,
        carry: Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= HALT_MAX_STEPS

            halted = is_last_step

            # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
            if self.training and (HALT_MAX_STEPS > 1):
                # In the original there's a conditional here, but we always want to
                # handle the halting logic like this
                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < HALT_EXPLORATION_PROB
                ) * torch.randint_like(new_steps, low=2, high=HALT_MAX_STEPS + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
