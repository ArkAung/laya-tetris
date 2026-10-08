#!/usr/bin/env python3
"""Turn the training file (one jsonl with a "split" column) into the layout the Hugging Face dataset page expects.

  python scripts/split_for_hf.py tetris_data hf_dataset

Writes hf_dataset/data/{train,validation,test}.jsonl, hf_dataset/tetris_decisions_all.jsonl (the exact file used
for training, with its split column) and hf_dataset/questions.json. The splits are by game, never by row.
"""
import json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(f"{dst}/data", exist_ok=True)
names = {"train": "train", "val": "validation", "test": "test"}
out = {k: open(f"{dst}/data/{v}.jsonl", "w") for k, v in names.items()}
counts = dict.fromkeys(names, 0)
for line in open(f"{src}/train.jsonl"):
    row = json.loads(line); out[row["split"]].write(line); counts[row["split"]] += 1
for f in out.values(): f.close()
shutil.copy(f"{src}/train.jsonl", f"{dst}/tetris_decisions_all.jsonl")
shutil.copy(f"{src}/questions.json", f"{dst}/questions.json")
print("rows per split:", {names[k]: v for k, v in counts.items()}, "-> fill these into hf/dataset_card.md")
