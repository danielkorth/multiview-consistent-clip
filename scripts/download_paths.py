import json
from pathlib import Path
import hydra
from hydra import initialize, compose
import tqdm

from omegaconf import DictConfig
import logging
import subprocess

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def render_data(cfg: DictConfig):
    
    command = (
        f"python3 {cfg.local.objaverse_dir}/download_objaverse.py --start_i 0 --end_i {cfg.num_objects}"
    )
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    render_data()