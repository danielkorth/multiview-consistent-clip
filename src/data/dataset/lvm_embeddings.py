from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import json
from typing import Dict, List


class LVMEmbeddingsDataset(Dataset):
    def __init__(self, data_dir: str, objects_list_csv_path: str):
        super().__init__()

        self.embeddings_dir: str = os.path.join(data_dir, "renderings") #TODO: fix ugly naming
        self.hashes = pd.read_csv(objects_list_csv_path)["uid"].tolist()

        #TODO read from some metadata file
        self.num_encodings_per_object = 36 
        self.num_datapoints_per_object = 6
        self.num_images_per_data_point = 6

    def __len__(self) -> int:
        return len(self.hashes) * self.num_encodings_per_object

    # def __getitem__(self, idx: int) -> Dict[torch.tensor, torch.tensor]:
    #     """Return image and text prompt embedding pair."""
    #     hash_idx = idx // self.num_encodings_per_object
    #     embedding_idx = idx % self.num_encodings_per_object

    #     prompt_embedding_path = os.path.join(self.embeddings_dir, self.hashes[hash_idx], "prompt_embedding.pt")
    #     image_embedding_path = os.path.join(self.embeddings_dir, self.hashes[hash_idx], 'img' + str(embedding_idx).zfill(3) + ".pt")

    #     image_embedding = torch.load(image_embedding_path)
    #     prompt_embedding = torch.load(prompt_embedding_path)

    #     return {'prompt_embedding': prompt_embedding, 'image_embedding': image_embedding}
    
    def __getitem__(self, idx: int) -> Dict[torch.tensor, List[torch.tensor]]:
        """Text prompt embedding with n corresponding image embeddings."""

        hash_idx = idx // self.num_datapoints_per_object
        first_embedding_idx = (idx % self.num_datapoints_per_object)*self.num_images_per_data_point

        prompt_embedding_path = os.path.join(self.embeddings_dir, self.hashes[hash_idx], "prompt_embedding.pt")
        image_embedding_paths = \
            [os.path.join(self.embeddings_dir, self.hashes[hash_idx], str(embedding_idx).zfill(3) + ".pt") \
              for embedding_idx in range(first_embedding_idx, first_embedding_idx + self.num_images_per_data_point)]

        prompt_embedding = torch.load(prompt_embedding_path)
        image_embeddings = [torch.load(image_embedding_path) for image_embedding_path in image_embedding_paths]

        return {'prompt_embedding': prompt_embedding, 'image_embeddings': image_embeddings}
    

