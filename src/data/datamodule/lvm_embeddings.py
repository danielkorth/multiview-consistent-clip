import pytorch_lightning as pl 
from torch.utils.data import DataLoader 
from src.data.dataset.lvm_embeddings import LVMEmbeddingsDataset
import os

class LVMEmbeddingsDataModule(pl.LightningDataModule): 
    def __init__(
            self,  
            data_dir: str,
            batch_size: int,
            shuffle: bool = True,
            num_workers: int = 1,
            pin_memory: bool = False
            **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None: 
        if stage == ['fit', 'all']:
            self.train_csv = os.path.join(self.hparams['data_dir'], 'train.csv')
            self.train_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.train_csv)
        if stage in ['validate', 'fit', 'all']:
            self.val_csv = os.path.join(self.hparams['data_dir'], 'val.csv')
            self.val_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.val_csv)
        if stage in ['test', 'all']:
            self.test_csv = os.path.join(self.hparams['data_dir'], 'test.csv')
            self.test_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.test_csv)
  
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.train_data, batch_size=self.hparams['batch_size'], shuffle=self.hparams['shuffle'], num_workers=self.hparams['num_workers'], self.pin_memory=self.hparams['pin_memory']) 
  
    def val_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_data, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.hparams['num_workers'], self.pin_memory=self.hparams['pin_memory'])
    
    def test_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_data, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.hparams['num_workers'], self.pin_memory=self.hparams['pin_memory'])