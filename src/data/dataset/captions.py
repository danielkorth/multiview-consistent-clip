from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from nptyping import NDArray
from typing import Tuple
from pathlib import Path
import json


class CaptionsDataset(Dataset):
    def __init__(self, objaverse_dir: str) -> None:
        super().__init__()
        self.renderings_dir: str = os.path.join(objaverse_dir, "renderings")
        uid_to_name = Path(objaverse_dir) / "uid_to_name.json"
        with open(uid_to_name, 'r') as f:
            uid_to_name_dict = json.load(f)
        self.uid_list = list(uid_to_name_dict.keys())
        self.name_list = list(uid_to_name_dict.values())

    def __len__(self) -> int:
        return len(self.name_list)

    def __getitem__(self, idx: int) -> Tuple[NDArray, str]: 
        return {
            'uid': self.uid_list[idx],
            'captions': self.name_list[idx]
        }
