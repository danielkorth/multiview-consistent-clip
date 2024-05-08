import json
import time
import hydra
import tqdm
import os

from omegaconf import DictConfig
import logging
import subprocess

from lightning import seed_everything


logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def render_data(cfg: DictConfig):
    seed_everything(cfg.seed)
    # TODO update the json files by removing the ids that had a timeout during rendering
    start_time = time.time()
    with open(cfg.input_models_path, "r") as f:
        model_paths = json.load(f)
    for model_path in tqdm.tqdm(model_paths):
        command = (
            # for linux
            # f"export DISPLAY=:0 && export SSL_CERT_DIR=/etc/ssl/certs &&"
            # for windows
            f"blender -b -P {cfg.local.base_dir}/objaverse/blender_script.py --"
            f" --object_path {model_path}"
            f" --output_dir {cfg.output_dir}/renderings"
        )
        process = subprocess.Popen(command, shell=True)
        start_command_time = time.time()
        while True:
            if process.poll() is not None:
                break
            current_time = time.time()
            if current_time - start_command_time > 5*60:
                process.kill()
                logger.warning("Command execution exceeded 5 minute. Terminating process.")
                break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generation completed in {elapsed_time} seconds.")

    # post processing and remove all unrednreded stuff
    with open(f'{cfg.output_dir}/uid_to_name.json', 'r') as f:
        uid_to_name = json.load(f)

    folder_names = [name for name in os.listdir(f'{cfg.output_dir}/renderings') if os.path.isdir(os.path.join(f'{cfg.output_dir}/renderings', name))]
    uid_to_name = {k: v for k, v in uid_to_name.items() if k in folder_names}
    new_input_models_path = list()

    for path in model_paths:
        for idx in folder_names:
            if idx in path:
                new_input_models_path.append(path)

    with open(f'{cfg.output_dir}/uid_to_name.json', 'w') as f:
        json.dump(uid_to_name, f)

    with open(f'{cfg.output_dir}/input_models_path.json', 'w') as f:
        json.dump(model_paths, f)
    
    print(f"Done with postprocessing")
    
if __name__ == "__main__":
    render_data()
