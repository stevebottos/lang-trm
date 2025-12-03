from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from dataset import SynthDataset
from dataset_dolly import DollyDataset
from model import LangTRM
import torch
import mlflow
from copy import deepcopy


def prepare_targets_for_loss(
    targets: torch.Tensor, pad_token_id: int, ignore_index: int = -100
) -> torch.Tensor:
    """
    Replaces all but the first pad_token_id in the target tensor with the ignore_index.
    """
    pad_mask = targets == pad_token_id
    targets_modified = targets.masked_fill(pad_mask, ignore_index)

    return targets_modified


def calculate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    """
    Calculate accuracy where predicted IDs match target IDs, excluding ignore_index positions.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) - may contain ignore_index for positions to skip
        ignore_index: token ID to exclude from accuracy calculation

    Returns:
        accuracy as a float between 0 and 1
    """
    predictions = logits.argmax(dim=-1)

    # Create mask for valid (non-ignored) tokens
    valid_mask = targets != ignore_index

    # Calculate accuracy only on valid tokens
    correct = (predictions == targets) & valid_mask
    accuracy = correct.sum().item() / valid_mask.sum().item()

    return accuracy


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


def tokenize_batch(samples):
    queries = samples["query"]
    n = len(queries)
    targets = samples["target"]
    tokenized = tokenizer(
        queries + targets,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )["input_ids"].to(device)

    return tokenized[:n, :], tokenized[n:, :]


device = torch.device("cuda")

model = LangTRM().to(device)

# Print model parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'=' * 80}")
print(f"Model Parameters:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"{'=' * 80}\n")

ds = DollyDataset()
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
criterion = torch.nn.CrossEntropyLoss()

# Initialize
mlflow.start_run()

global_step = 0
for step in range(10000):
    for global_step, samples in enumerate(dl):
        inputs, targets = tokenize_batch(samples)
        original_inputs = deepcopy(inputs)
        input_is_embedding = False
        # Prepare targets: replace all but the first EOS token with -100 (ignore index)
        targets = prepare_targets_for_loss(targets, tokenizer.pad_token_id)

        batch_size, seq_len = inputs.shape

        # Initialize h and l with proper shapes
        h = (
            model.learned_H_starter.view(1, 1, -1)
            .expand(batch_size, seq_len, -1)
            .clone()
        )
        l = (
            model.learned_L_starter.view(1, 1, -1)
            .expand(batch_size, seq_len, -1)
            .clone()
        )

        optimizer.zero_grad()
        total_loss = 0
        for deep_supervision_step in range(8):
            # Use automatic mixed precision with bfloat16
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, h, l, embeddings = model(h, l, inputs, input_is_embedding)

                # Deep supervision: compute loss at each step
                step_loss = criterion(logits.permute(0, 2, 1), targets)
                total_loss = total_loss + step_loss

                inputs = embeddings
                input_is_embedding = True

        # Average the losses across all steps
        loss = total_loss / 8
        loss.backward()
        optimizer.step()

        # Calculate metrics
        loss_value = loss.item()
        accuracy = calculate_accuracy(logits, targets, ignore_index=-100)

        # Log to MLflow
        mlflow.log_metric("loss", loss_value, step=global_step)
        mlflow.log_metric("accuracy", accuracy, step=global_step)

        print(f"Step {global_step}: loss={loss_value:.4f}, accuracy={accuracy:.4f}")

        # Periodically show predictions vs targets
        if global_step % 100 == 0:
            print("\n" + "=" * 80)
            print(f"Example predictions at step {global_step}:")
            print("=" * 80)

            # Get predictions
            predictions = logits.argmax(dim=-1)

            # Show first 2 examples from the batch
            for i in range(min(2, predictions.shape[0])):
                input_tokens = original_inputs[i].tolist()
                pred_tokens = predictions[i].tolist()
                target_tokens = targets[i].tolist()

                # Filter out ignore_index (-100) from targets for decoding
                target_tokens_filtered = [t for t in target_tokens if t != -100]

                # Decode to text and strip whitespace/newlines, skip special tokens
                input_text = tokenizer.decode(
                    input_tokens, skip_special_tokens=True
                ).strip()
                pred_text = tokenizer.decode(
                    pred_tokens, skip_special_tokens=True
                ).strip()
                target_text = tokenizer.decode(
                    target_tokens_filtered, skip_special_tokens=True
                ).strip()

                print(f"\nExample {i + 1}:")
                print(f"  Input:      {input_text}")
                print(f"  Target:     {target_text}")
                print(f"  Predicted:  {pred_text}")

            print("=" * 80 + "\n")


mlflow.end_run()
