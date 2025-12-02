from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
    print()
    tokenized = tokenizer(
        queries + targets,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )["input_ids"].to(device)

    return tokenized[:n, :], tokenized[n:, :]


queries, targets = tokenize_batch(samples)
print(queries, targets)
batch = {"inputs": queries}
# print(queries.shape, targets.shape)
carry = model.dummy_carry(batch, device)
print(carry)

out = model(carry, batch)
print(out)
