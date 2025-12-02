from dotenv import load_dotenv
import sqlite3
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import torch


load_dotenv(".env")


@dataclass
class DeepSupervisionTextExample:
    query: str
    target: str
    inner_carry: Any
    current_data: Dict[str, torch.Tensor]
    halted: bool = False
    steps_carried: int = 0


class SynthDataset:
    def __init__(self, db_path=None):
        if db_path is None:
            db_output_dir = os.getenv("DS_OUT_DIR")
            if not db_output_dir:
                raise ValueError("DS_OUT_DIR environment variable not set")
            db_path = Path(db_output_dir) / "filtered_data.db"
        self.db_path = db_path
        self.current_rowid = 1
        self.epoch_count = 0
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM synth_data")
        self.dataset_size = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT MIN(rowid), MAX(rowid) FROM synth_data")
        self.min_rowid, self.max_rowid = self.cursor.fetchone()
        self.current_rowid = self.min_rowid

    def samples_remaining(self) -> int:
        return self.max_rowid - self.current_rowid + 1

    def __len__(self):
        return self.dataset_size

    def take(self, k) -> tuple[List[DeepSupervisionTextExample], bool]:
        epoch_wrapped = False

        if self.current_rowid > self.max_rowid:
            self.current_rowid = self.min_rowid
            self.epoch_count += 1
            epoch_wrapped = True

        self.cursor.execute(
            "SELECT query, synthetic_reasoning, synthetic_answer FROM synth_data WHERE rowid >= ? ORDER BY rowid LIMIT ?",
            (self.current_rowid, k),
        )
        rows = self.cursor.fetchall()
        actual_samples = len(rows)
        self.current_rowid += actual_samples

        samples = []
        for query, reasoning, answer in rows:
            example = DeepSupervisionTextExample(
                query=query,
                target=answer,
                inner_carry=None,
                current_data={"query": query, "target": answer, "reasoning": reasoning},
                halted=False,
                steps_carried=0,
            )
            samples.append(example)

        if actual_samples < k:
            remaining_needed = k - actual_samples
            self.current_rowid = self.min_rowid
            self.epoch_count += 1
            epoch_wrapped = True
            additional_samples, _ = self.take(remaining_needed)
            samples.extend(additional_samples)

        return samples


if __name__ == "__main__":
    import time
    from tqdm import tqdm

    ds = SynthDataset()

    n_iter = (len(ds) // 16) + 1
    _t = time.time()
    for i in tqdm(range(n_iter)):
        samples, wrapped = ds.take(16)
    print(time.time() - _t)
