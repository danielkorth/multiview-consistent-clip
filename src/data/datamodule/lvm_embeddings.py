import pytorch_lightning as pl 
from torch.utils.data import DataLoader 
from src.data.dataset.lvm_embeddings import LVMEmbeddingsDataset
import os

class LVMEmbeddingsDataModule(pl.LightningDataModule): 
    def __init__(
            self,  
            data_dir: str,
            **kwargs) -> None:
        super().__init__()

        self.data_dir = data_dir
        # TODO: alter so that is matches file structure
        self.path_hashes_training = os.path.join(data_dir, 'hashes_training.csv')
        self.path_hashes_validation = os.path.join(data_dir, 'hashes_validation.csv')
        self.path_hashes_test = os.path.join(data_dir, 'hashes_test.csv')

    def setup(self, split: str = 'train') -> None: 
        self.train_data = LVMEmbeddingsDataset(self.data_dir, self.path_hashes_training)
        self.val_data = LVMEmbeddingsDataset(self.data_dir, self.path_hashes_validation)
        self.test_data = LVMEmbeddingsDataset(self.data_dir, self.path_hashes_test)
  
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.train_data, batch_size=32, shuffle=True) 
  
    def val_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_data, batch_size=32, shuffle=True) 
    
    def test_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_data, batch_size=32, shuffle=True)