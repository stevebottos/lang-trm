from transformers import GPT2Tokenizer
from dataset import SynthDataset
from model import LangTRM
import torch


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


device = torch.device("cuda")

model = LangTRM().to(device)

ds = SynthDataset()
samples = ds.take(16)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


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


# Initialize
queries, targets = tokenize_batch(samples)
batch = {"inputs": queries}
carry = model.initial_carry(batch, device)

for step in range(10000):
    out = model(carry, batch)
