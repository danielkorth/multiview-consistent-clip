from pathlib import Path
import hydra
from hydra import initialize, compose

from omegaconf import DictConfig
import logging
import subprocess

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def render_data(cfg: DictConfig):
    
    command = (
        f"python3 {cfg.local.base_dir}/objaverse/download_objaverse.py --count {cfg.num_objects} --save_json_path {cfg.output_dir}"
    )
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    render_data()