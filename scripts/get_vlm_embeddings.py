from pathlib import Path
import tqdm

from omegaconf import DictConfig
import logging
import torch 
import hydra

from src.models.vlm import VLM
from src.data.dataset.rendered_images import RenderedImagesDataset
from torch.utils.data import DataLoader

from lightning import seed_everything

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="get_vlm_embeddings")
def get_vlm_embeddings(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RenderedImagesDataset(cfg.output_dir)
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=4)
    model = VLM(vlm_name = cfg.vlm_name, device=device)

    output_dir = Path(cfg.output_dir) / "renderings"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    for batch_idx, data in enumerate(tqdm.tqdm(dataloader, desc="Processing images")):
        info, images = data['path'], data['image']
        images = images.to(device)
        image_embeddings = model.forward_image(images)

        for one_info, one_embed in zip(info, image_embeddings):
            one_info = Path(one_info)
            view_id = one_info.name[0:3]
        
            # Save embeddings to the output directory
            torch.save(one_embed, output_dir / one_info.parent / f"clip_embed_{view_id}.pt")


if __name__ == "__main__":
    get_vlm_embeddings()