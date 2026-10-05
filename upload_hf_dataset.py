#!/usr/bin/env python3
"""Upload the Review Lottery dataset to Hugging Face Hub.

Requires: `huggingface-cli login` (or HF_TOKEN env) with write access.
Run:  .venv/bin/python scripts_hf/upload_hf_dataset.py
Creates: public dataset repo `EDGAhab/review-lottery-data` (change REPO_ID as needed).
"""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "EDGAhab/review-lottery-data"
BASE = os.path.expanduser("~/workspace/review-lottery")

api = HfApi()
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

# README (dataset card) = release/data_card.md with HF front-matter
card = open(f"{BASE}/release/data_card.md").read()
front = ("---\nlicense: cc0\ntask_categories:\n- tabular-classification\n"
         "tags:\n- peer-review\n- iclr\n- metascience\n---\n\n")
open("/tmp/hf_dataset_README.md", "w").write(front + card)
api.upload_file(path_or_fileobj="/tmp/hf_dataset_README.md",
                path_in_repo="README.md", repo_id=REPO_ID, repo_type="dataset")

# processed tables
for f in ["papers.parquet", "reviews.parquet", "peersum_reviews.parquet"]:
    api.upload_file(path_or_fileobj=f"{BASE}/data/processed/{f}",
                    path_in_repo=f"data/{f}", repo_id=REPO_ID, repo_type="dataset")
    print("uploaded", f)

# notes (tables behind every number in the paper)
for d in ["phase2_notes", "phase3_notes"]:
    api.upload_folder(folder_path=f"{BASE}/{d}", path_in_repo=d,
                      repo_id=REPO_ID, repo_type="dataset")
    print("uploaded", d)

# posterior draws (~1.1 GB; needed to re-run the B=1000 simulation exactly)
api.upload_folder(folder_path=f"{BASE}/posteriors", path_in_repo="posteriors",
                  repo_id=REPO_ID, repo_type="dataset")
print("uploaded posteriors")

print("DONE: https://huggingface.co/datasets/" + REPO_ID)
