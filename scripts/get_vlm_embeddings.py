from pathlib import Path
import tqdm

from omegaconf import DictConfig
import logging
import torch 
import hydra

from src.models.vlm import VLM
from src.data.renderdataset import RenderDataset
from torch.utils.data import DataLoader

from lightning import seed_everything

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="data_preparation")
def get_vlm_embeddings(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RenderDataset()
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    model = VLM(vlm_name = cfg.vlm_name).to(device)

    output_dir = Path(cfg.output_dir) / "renderings"
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, data in enumerate(tqdm(dataloader, desc="Processing images")):
        info, images = data
        images = images.to(device)
        image_embeddings = model.forward_image(images)

        for one_info, one_embed in zip(info, image_embeddings):
            one_info = Path(one_info)
            view_id = one_info.name[0:3]
        
            # Save embeddings to the output directory
            torch.save(image_embeddings, output_dir / one_info / f"clip_embed_{view_id}.pt")

_name__ == "__main__":
    get_vlm_embeddings()
