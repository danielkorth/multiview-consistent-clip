from pathlib import Path
import tqdm

from omegaconf import DictConfig
import logging
import torch 
import hydra
import os

from src.models.vlm import VLM
from src.data.dataset.rendered_images import RenderedImagesDataset
from src.data.dataset.captions import CaptionsDataset
from torch.utils.data import DataLoader

from lightning import seed_everything

logger = logging.getLogger(__name__)

@hydra.main(version_base= None, config_path="../configs", config_name="get_vlm_embeddings")
def get_vlm_embeddings(cfg: DictConfig):
    # assert False == True, "need to fix TODO"
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    dataset_images = RenderedImagesDataset(cfg.output_dir)
    dataset_captions = CaptionsDataset(cfg.output_dir)
    dataloader_images = DataLoader(dataset_images, batch_size=cfg.batch_size_images)
    dataloader_captions = DataLoader(dataset_captions, batch_size=cfg.batch_size_text)

    output_dir = Path(cfg.output_dir)
    renderings_dir = output_dir / "renderings"
    renderings_dir.mkdir(parents=True, exist_ok=True)

    model = VLM(vlm_name = cfg.vlm_name, device=device)
    model = model.to(device)

    # get text embeddings
    for batch_idx, data in enumerate(tqdm.tqdm(dataloader_captions, desc="Processing text")):
        uids, captions = data['uid'], data['captions']
        caption_embeddings = model.forward_text(captions)

        for uid, caption in zip(uids, caption_embeddings):
            # dirty fix for following problem fix 
            # actually: TODO filteer the uid_to_name.json based on what was rendered and what was not rendered
            if os.path.exists(renderings_dir / uid):
                torch.save(caption.detach().cpu(), renderings_dir / uid / f"{cfg.vlm_name}_text_embed.pt")

    # get vision embeddings
    for batch_idx, data in enumerate(tqdm.tqdm(dataloader_images, desc="Processing images")):
        info, images = data['path'], data['image']
        images = images.to(device)
        image_embeddings = model.forward_image(images)

        for one_info, one_embed in zip(info, image_embeddings):
            one_info = Path(one_info)
            view_id = one_info.name[0:3]
        
            # Save embeddings to the output directory
            torch.save(one_embed.detach().cpu(), renderings_dir / one_info.parent / f"{cfg.vlm_name}_embed_{view_id}.pt")
    

if __name__ == "__main__":
    get_vlm_embeddings()