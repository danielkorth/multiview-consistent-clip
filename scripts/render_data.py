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
    import time
    start_time = time.time()
    with open(cfg.input_models_path, "r") as f:
        model_paths = json.load(f)
    for model_path in tqdm.tqdm(model_paths):
        command = (
            f"export DISPLAY=:0 && export SSL_CERT_DIR=/etc/ssl/certs &&"
            f"~/blender-3.2.2-linux-x64/blender -b -P {cfg.local.base_dir}/objaverse/blender_script.py --"
            f" --object_path {model_path}"
            f" --output_dir {cfg.output_dir}/renderings"
        )
        subprocess.run(command, shell=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generation completed in {elapsed_time} seconds.")
    
if __name__ == "__main__":
    render_data()
