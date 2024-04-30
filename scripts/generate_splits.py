from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
# import rootutils
from omegaconf import DictConfig
from pathlib import Path
import json
import numpy as np
import pandas as pd

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# # ------------------------------------------------------------------------------------ #
# # the setup_root above is equivalent to:
# # - adding project root dir to PYTHONPATH
# #       (so you don't need to force user to install project as a package)
# #       (necessary before importing any local modules e.g. `from src import utils`)
# # - setting up PROJECT_ROOT environment variable
# #       (which is used as a base for paths in "configs/paths/default.yaml")
# #       (this way all filepaths are the same no matter where you run the code)
# # - loading environment variables from ".env" in root dir
# #
# # you can remove it if you:
# # 1. either install project as a package or move entry files to project root dir
# # 2. set `root_dir` to "." in "configs/paths/default.yaml"
# #
# # more info: https://github.com/ashleve/rootutils
# # ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="data_preparation")
def generate_splits(cfg: DictConfig) -> Optional[float]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    data_path = Path(cfg.output_dir)

    # Load the JSON file containing object names
    with open(data_path / 'uid_to_name.json', 'r') as file:
        uid_to_name = json.load(file)

    # Get all unique IDs from the uid_to_name dictionary
    all_uids = list(uid_to_name.keys())

    # Shuffle the list of UIDs to ensure randomness
    np.random.shuffle(all_uids)

    # Calculate the number of samples for each split
    num_total = len(all_uids)

    assert cfg.train_split_size + cfg.val_split_size + cfg.test_split_size == 1, "train/val/test split cannot be larger than one"
    num_train = int(num_total * cfg.train_split_size)
    num_val = int(num_total * cfg.val_split_size)

    # Split the UIDs into train, validation, and test sets
    train_uids = all_uids[:num_train]
    val_uids = all_uids[num_train:num_train + num_val]
    test_uids = all_uids[num_train + num_val:]

    # Create DataFrames for train, validation, and test sets
    train_df = pd.DataFrame(train_uids, columns=['uid'])
    val_df = pd.DataFrame(val_uids, columns=['uid'])
    test_df = pd.DataFrame(test_uids, columns=['uid'])

    # Save DataFrames to CSV files
    train_df.to_csv(data_path / 'train.csv', index=False)
    val_df.to_csv(data_path / 'val.csv', index=False)
    test_df.to_csv(data_path / 'test.csv', index=False)

    # Create a DataFrame for overfitting with only a single ID
    train_overfit_df = pd.DataFrame(train_uids[:1], columns=['uid'])

    # Save the DataFrame to a CSV file
    train_overfit_df.to_csv(data_path / 'train_overfit.csv', index=False)

    # Create a DataFrame for a small batch training with 8 IDs
    train_batch_df = pd.DataFrame(train_uids[:8], columns=['uid'])

    # Save the DataFrame to a CSV file
    train_batch_df.to_csv(data_path / 'train_batch.csv', index=False)

    log.info("CSV files have been saved.")

if __name__ == "__main__":
    generate_splits()
