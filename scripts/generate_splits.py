from pathlib import Path
import hydra
from hydra import initialize, compose

from omegaconf import DictConfig
import logging
import subprocess
from lightning import seed_everything

import json
import pandas as pd
import numpy as np
np.random.seed(42)

from pathlib import Path

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def generate_splits(cfg: DictConfig):
    seed_everything(cfg.seed)

    data_path = Path(cfg.data_dir)

    # Load the JSON file containing object names
    with open(data_path / 'uid_to_name.json', 'r') as file:
        uid_to_name = json.load(file)

    # Get all unique IDs from the uid_to_name dictionary
    all_uids = list(uid_to_name.keys())

    # Shuffle the list of UIDs to ensure randomness
    np.random.shuffle(all_uids)

    # Calculate the number of samples for each split
    num_total = len(all_uids)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    num_test = num_total - num_train - num_val

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
    train_overfit_df.to_csv(data_path / '/train_overfit.csv', index=False)

    # Create a DataFrame for a small batch training with 8 IDs
    train_batch_df = pd.DataFrame(train_uids[:8], columns=['uid'])

    # Save the DataFrame to a CSV file
    train_batch_df.to_csv(data_path / 'train_batch.csv', index=False)

    print("CSV files for overfitting and small batch training have been saved.")

if __name__ == "__main__":
    generate_splits()
