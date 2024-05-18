import lightning as L
from torch.utils.data import DataLoader 
from src.data.dataset.lvm_embeddings import LVMEmbeddingsDataset
import os

class LVMEmbeddingsDataModule(L.LightningDataModule): 
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None: 
        if stage in ['fit', 'all']:
            self.train_csv =  self.hparams['train_split']
            self.train_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.train_csv)
        if stage in ['validate', 'fit', 'all']:
            self.val_csv = self.hparams['val_split']
            self.val_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.val_csv, mode='val')
        if stage in ['test', 'all']:
            self.test_csv = self.hparams['test_split']
            self.test_dataset = LVMEmbeddingsDataset(self.hparams['data_dir'], self.test_csv, mode='test')
  
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=self.hparams['shuffle'], num_workers=self.hparams['num_workers'], pin_memory=self.hparams['pin_memory']) 
  
    def val_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.hparams['num_workers'], pin_memory=self.hparams['pin_memory'])
    
    def test_dataloader(self) -> DataLoader: 
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.hparams['num_workers'], pin_memory=self.hparams['pin_memory'])