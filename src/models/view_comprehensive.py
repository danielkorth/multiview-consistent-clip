from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.vlm_autoencoder import VLMAutoencoder
from src.models.components.losses import LossAutoencoder

class ViewComprehensiveEmbeddingModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss = loss

    def model_step(
        self, batch: Dict[str, torch.tensor]
    ) -> torch.Tensor:
        
        """
        batch: dict{ 
            prompt_embeddings: torch.tensor (batch size, embedding size), 
            original_img_embeddings: torch.tensor (batch size, data points size, embedding size)}
            distances between text and image embedding: torch.tensor (batch size, data points size)
        """
    
        original_img_embeddings = batch["image_embeddings"]
        view_comprehensive_decodings, view_independent_encodings = self.net.forward(original_img_embeddings)

        return self.loss(original_img_embeddings, view_comprehensive_decodings, view_independent_encodings)

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:

        loss_dict = self.model_step(batch)

        self.log("train/loss", loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auto_score", loss_dict['auto_score_normalized'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/vi_score", loss_dict['vi_score_normalized'], on_step=False, on_epoch=True, prog_bar=False)

        return loss_dict['loss']

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss_dict = self.model_step(batch)
        
        self.log("val/loss", loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auto_score", loss_dict['auto_score_normalized'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/vi_score", loss_dict['vi_score_normalized'], on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss_dict = self.model_step(batch)

        self.log("test/loss", loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.hparams.optimizer(params=self.net.paramters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
