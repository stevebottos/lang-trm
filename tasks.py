from invoke.tasks import task
from dotenv import load_dotenv
import os

load_dotenv(".env")


@task
def get_synth_dataset(c):
    output = os.getenv("DS_OUT_DIR")
    assert os.path.exists(output)  # type: ignore

    from huggingface_hub import snapshot_download

    snapshot_download(  # type: ignore
        repo_id="PleIAs/SYNTH",
        repo_type="dataset",
        local_dir=os.path.join(output, "synth"),
        local_dir_use_symlinks=False,
        allow_patterns="*.parquet",  # Download only Parquet files
    )


@task
def process_synth_dataset(c):
    import os
    from pathlib import Path
    import polars as pl
    import sqlite3
    from transformers import AutoTokenizer

    MAX_TOKENS = 256

    output = Path(os.getenv("DS_OUT_DIR")) / "synth"
    db_path = Path(os.getenv("DS_OUT_DIR")) / "filtered_data.db"
    parquets = list(output.glob("*.parquet"))
    n_entries = 0
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Setup SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS synth_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        synthetic_reasoning TEXT, 
        synthetic_answer TEXT
    )
    """)
    # Clear existing data to prevent duplicates on subsequent runs
    cursor.execute("DELETE FROM synth_data")
    conn.commit()

    for i, p in enumerate(parquets):
        df = pl.read_parquet(p)
        print(f"Original DataFrame length for file {i}: {len(df)}")

        df = df.with_columns(
            [
                pl.col("query")
                .map_elements(lambda x: len(tokenizer.encode(x)), return_dtype=pl.Int64)
                .alias("query_counts"),
                pl.col("synthetic_answer")
                .map_elements(lambda x: len(tokenizer.encode(x)), return_dtype=pl.Int64)
                .alias("target_counts"),
                pl.col("synthetic_reasoning")
                .map_elements(lambda x: len(tokenizer.encode(x)), return_dtype=pl.Int64)
                .alias("reasoning_counts"),
            ]
        )

        filtered_df = df.filter(
            (pl.col("query_counts") < MAX_TOKENS)
            & (pl.col("reasoning_counts") < MAX_TOKENS)
            & (pl.col("target_counts") < MAX_TOKENS)
            & (pl.col("language") == "en")
        )

        # -- Debugging --
        # Original 'processed file' print statement already shows rows in filtered_df
        # We will clarify the output of that statement.
        # -- End Debugging --

        # Select columns and insert into SQLite
        to_insert = filtered_df.select(
            ["query", "synthetic_reasoning", "synthetic_answer"]
        )
        records = to_insert.iter_rows()

        cursor.executemany(
            "INSERT INTO synth_data (query, synthetic_reasoning, synthetic_answer) VALUES (?, ?, ?)",
            records,
        )
        conn.commit()

        n_entries += len(filtered_df)
        print(
            f"Processed file {i}, added {len(filtered_df)} entries. Total entries: {n_entries}"
        )

    conn.close()
    print(f"Finished processing. Data saved to {db_path}")
