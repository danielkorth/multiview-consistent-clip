from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import json
from typing import Dict, List
import random


class LVMEmbeddingsDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            objects_list_csv_path: str, 
            num_encodings_per_object: int = 36, 
            datapoint_size: int = 2):
        super().__init__()

        self.embeddings_dir: str = os.path.join(data_dir, "renderings") #TODO: fix ugly naming
        self.hashes = pd.read_csv(objects_list_csv_path)["uid"].tolist()

        self.num_encodings_per_object = num_encodings_per_object
        self.datapoint_size = datapoint_size

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
    
    def __getitem__(self, idx: int) -> Dict[torch.tensor, torch.tensor, torch.tensor]:
        """
        Return text prompt embedding with datapoint_size corresponding image embeddings and distances.
        
        batch: dict{ 
            prompt_embeddings: torch.tensor (embedding size), 
            original_img_embeddings: torch.tensor (data points size, embedding size)}
            distances between text and image embedding: torch.tensor (data points size)
        """
        image_indices = random.sample(range(self.num_encodings_per_object), self.datapoint_size)

        prompt_embedding_path = os.path.join(self.embeddings_dir, self.hashes[idx], "prompt_embedding.pt")
        image_embedding_paths = \
            [os.path.join(self.embeddings_dir, self.hashes[idx], 'img' + str(embedding_idx).zfill(3) + ".pt") \
              for embedding_idx in image_indices]

        prompt_embedding = torch.load(prompt_embedding_path)
        image_embeddings_list = [torch.load(image_embedding_path) for image_embedding_path in image_embedding_paths]
        distances_list = [torch.norm(prompt_embedding - image_embedding) for image_embedding in image_embeddings]

        image_embeddings = torch.stack(image_embeddings_list)
        distances = torch.stack(distances_list)

        return {'prompt_embedding': prompt_embedding, 'image_embeddings': image_embeddings, 'distances': distances}
    

