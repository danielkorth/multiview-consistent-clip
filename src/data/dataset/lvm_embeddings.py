from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import json


class LVMEmbeddingsDataset(Dataset):
    def __init__(self, data_dir: str, objects_list_csv_path: str):
        super().__init__()

        self.uid_to_name_dict: dict = json.load(open(os.path.join(data_dir, "uid_to_name.json"), "r"))
        self.embeddings_dir: str = os.path.join(data_dir, "renderings") #TODO: fix ugly naming
        self.hashes = pd.read_csv(objects_list_csv_path)["uid"].tolist()

        self.num_encodings_per_object = 36 #TODO read from some metadata file

    def __len__(self) -> int:
        return len(self.hashes) * self.num_encodings_per_object

    def __getitem__(self, idx: int) -> dict:
        #TODO: we dont really need the text prompt do we? only text promt embedding

        hash_idx = idx // self.num_encodings_per_object
        embedding_idx = idx % self.num_encodings_per_object
        image_embedding_path = os.path.join(self.embeddings_dir, self.hashes[hash_idx], str(embedding_idx).zfill(3) + ".pt")

        image_embedding = torch.load(image_embedding_path)
        text_prompt = self.uid_to_name_dict[self.hashes[hash_idx]]

        return {'image_embedding': image_embedding, 'text_prompt': text_prompt}
    

