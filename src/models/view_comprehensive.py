from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


from src.models.components.vlm_autoencoder import VLMAutoencoder
from src.models.components.loss_functions import loss_autoencoder_embedding

class ViewComprehensiveEmbeddingModule(LightningModule):

    def __init__(
        self,
        net: VLMAutoencoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


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

        return loss_autoencoder_embedding(original_img_embeddings, view_comprehensive_decodings, view_independent_encodings)

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:

        loss, loss_auto, loss_vi = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_auto", loss_auto, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_vi", loss_vi, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss, loss_auto, loss_vi = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_auto", loss_auto, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_vi", loss_vi, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss= self.model_step(batch)

        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

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
