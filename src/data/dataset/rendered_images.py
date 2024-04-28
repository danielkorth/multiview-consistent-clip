from torch.utils.data import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
from nptyping import NDArray
from typing import Tuple


class RenderedImagesDataset(Dataset):
    def __init__(self, objaverse_dir: str) -> None:
        super.__init__()
        self.renderings_dir: str = os.path.join(objaverse_dir, "renderings")

        self.image_relative_paths: list[str] = []
        self.size: int|None = None #num datapoints

        self.expected_size = 36*2041 # 36 images per object. TODO: read from some metadata file

    def load_relative_image_paths(self)->None:

        sub_folders = os.listdir(self.renderings_dir)
        for folder in sub_folders:
            image_paths = os.listdir(os.path.join(self.renderings_dir, folder))
            for image_path in image_paths:
                self.image_relative_paths.append(os.path.join(folder,image_path))
        
        assert len(self.image_relative_paths) == self.expected_size, "Number of images is not as expected." 
        self.size = len(self.image_relative_paths)    

    def __len__(self) -> int:
        assert self.size is not None, "Relative image paths not loaded."
        return self.size

    def __getitem__(self, idx: int) -> Tuple[NDArray, str]: #TODO specify input shape, and reshape
        assert self.image_relative_paths, "Relative image paths not loaded."

        image_relative_path = self.image_relative_paths[idx]
        image = plt.imread(os.path.join(self.renderings_dir, image_relative_path))[:, :, :3]

        return (image, image_relative_path)
    

