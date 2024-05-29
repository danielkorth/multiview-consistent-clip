from torch.utils.data import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
from nptyping import NDArray
from typing import Tuple
from pathlib import Path
from PIL import Image
import numpy as np


class RenderedImagesDataset(Dataset):
    def __init__(self, objaverse_dir: str) -> None:
        super().__init__()
        self.renderings_dir: str = os.path.join(objaverse_dir, "renderings")
        if not os.path.exists(self.renderings_dir):
            os.makedirs(self.renderings_dir)

        self.image_relative_paths: list[str] = []

        sub_folders = os.listdir(self.renderings_dir)
        for folder in sub_folders:
            image_paths = os.listdir(os.path.join(self.renderings_dir, folder))
            for image_path in image_paths:
                if image_path.endswith('.png'):
                    self.image_relative_paths.append(os.path.join(folder,image_path))
 
        self.size = len(self.image_relative_paths)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[NDArray, str]: #TODO specify input shape, and reshape
        assert self.image_relative_paths, "Relative image paths not loaded."

        image_relative_path = self.image_relative_paths[idx]
        # alpha compositing
        png = Image.open(Path(self.renderings_dir) / image_relative_path).convert("RGBA")
        background = Image.new('RGBA', png.size, (255, 255, 255))
        img = Image.alpha_composite(background, png)
        image =np.asarray(img)[:, :, :3]

        return {
           'image': image,
           'path': str(image_relative_path)
        }