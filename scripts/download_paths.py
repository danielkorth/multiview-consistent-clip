from pathlib import Path
import hydra
from hydra import initialize, compose

from omegaconf import DictConfig
import logging
import subprocess
from lightning import seed_everything

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def download_paths(cfg: DictConfig):
    seed_everything(cfg.seed)
    command = (
        # change python or python3
        f"python {cfg.local.base_dir}/objaverse/download_objaverse.py --count {cfg.num_objects} --save_json_path {cfg.output_dir}"
    )
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    download_paths()