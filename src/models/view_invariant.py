from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import matplotlib.pyplot as plt

def save_cov_images(sim):
    fig, ax = plt.subplots()
    im = ax.imshow(sim)
    fig.colorbar(im)
    plt.imsave("temp2.png", sim)
    return plt.imread("temp2.png")

class ViewInvariantEmbeddingModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        temperature: int = 1
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net 
        self.loss = loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, img_embeddings: torch.Tensor) -> torch.Tensor:
        """ img_embeddings: torch.tensor (batch size, data point size, embedding size)"""

        # Reshape the input tensor to (batch size * data points size, embedding size)
        original_shape = img_embeddings.shape
        img_embeddings = img_embeddings.view(-1, original_shape[-1])
        predictions = self.net.forward(img_embeddings)
        return predictions.view(*original_shape)

    def model_step(
        self, batch: Dict[str, torch.tensor]
    ) -> dict:
        
        """
        batch: dict{ 
            prompt_embeddings: torch.tensor (batch size, embedding size), 
            img_embeddings: torch.tensor (batch size, data points size, embedding size)}
            distances between text and image embedding: torch.tensor (batch size, data points size)
        """

        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.forward(original_img_embeddings)
        
        return self.loss(text_embeddings, predicted_image_embeddings)

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self.model_step(batch)
        loss = loss_dict['loss']
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        sim_score = loss_dict.get('sim_score')
        if sim_score is not None:
            self.log("train/sim_score", sim_score, on_step=True, on_epoch=True, prog_bar=False)
        dissim_score = loss_dict.get('dissim_score')
        if dissim_score is not None:
            self.log("train/dissim_score", dissim_score, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:

        loss_dict = self.model_step(batch)
        loss = loss_dict['loss']
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        sim_score = loss_dict.get('sim_score')
        if sim_score is not None:
            self.log("val/sim_score", sim_score, on_step=True, on_epoch=True, prog_bar=False)
        dissim_score = loss_dict.get('dissim_score')
        if dissim_score is not None:
            self.log("val/dissim_score", dissim_score, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss_dict = self.model_step(batch)
        loss = loss_dict['loss']
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

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
        optimizer = self.hparams.optimizer(params=self.net.parameters())
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
    