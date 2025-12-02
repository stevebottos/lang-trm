from transformers import GPT2Tokenizer
from dataset import SynthDataset
from model import LangTRM
import torch
import mlflow


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
    logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int
) -> float:
    """
    Calculate accuracy where predicted IDs match target IDs, excluding padding tokens.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        pad_token_id: token ID to exclude from accuracy calculation

    Returns:
        accuracy as a float between 0 and 1
    """
    predictions = logits.argmax(dim=-1)

    # Create mask for non-padding tokens
    non_pad_mask = targets != pad_token_id

    # Calculate accuracy only on non-padding tokens
    correct = (predictions == targets) & non_pad_mask
    accuracy = correct.sum().item() / non_pad_mask.sum().item()

    return accuracy


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


def tokenize_batch(samples):
    n = len(samples)
    queries = [s.query for s in samples]
    targets = [s.target for s in samples]
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

ds = SynthDataset()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
criterion = torch.nn.CrossEntropyLoss()

# Initialize
mlflow.start_run()

global_step = 0
for step in range(10000):
    samples = ds.take(16)
    queries, targets = tokenize_batch(samples)
    batch = {"inputs": queries}

    carry = model.initial_carry(batch, device)
    for deep_supervision_step in range(8):
        optimizer.zero_grad()
        carry, logits = model(carry, batch)
        logits = logits["logits"]
        loss = criterion(logits.permute(0, 2, 1), targets)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        loss_value = loss.item()
        accuracy = calculate_accuracy(logits, targets, tokenizer.pad_token_id)

        # Log to MLflow
        mlflow.log_metric("loss", loss_value, step=global_step)
        mlflow.log_metric("accuracy", accuracy, step=global_step)

        print(f"Step {global_step}: loss={loss_value:.4f}, accuracy={accuracy:.4f}")

        # Periodically show predictions vs targets
        if global_step % 10 == 0:
            print("\n" + "=" * 80)
            print(f"Example predictions at step {global_step}:")
            print("=" * 80)

            # Get predictions
            predictions = logits.argmax(dim=-1)

            # Show first 2 examples from the batch
            for i in range(min(2, predictions.shape[0])):
                pred_tokens = predictions[i].tolist()
                target_tokens = targets[i].tolist()

                # Decode to text
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
                target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)

                print(f"\nExample {i + 1}:")
                print(f"  Target:     {target_text}")
                print(f"  Predicted:  {pred_text}")

            print("=" * 80 + "\n")

        global_step += 1
    # print(new_carry.new_current_data)

mlflow.end_run()
