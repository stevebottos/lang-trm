from torch.utils.data import Dataset


from dotenv import load_dotenv
import sqlite3
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import jsonlines

load_dotenv(".env")


class DollyDataset(Dataset):
    def __init__(self):
        data_json = Path(os.getenv("DS_OUT_DIR")) / "dolly/databricks-dolly-15k.jsonl"
        self.data = []
        with jsonlines.open(str(data_json), "r") as f:
            for line in f:
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {"query": sample["instruction"], "target": sample["response"]}


if __name__ == "__main__":
    ds = DollyDataset()
    print(ds[0])
