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
        f"python {cfg.local.base_dir}/objaverse/download_objaverse.py "
        f"--count {cfg.num_objects} "
        f"--save_json_path {cfg.output_dir} "
        f"{'--small' if cfg.use_small_dataset else ''}"
    )
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    download_paths()