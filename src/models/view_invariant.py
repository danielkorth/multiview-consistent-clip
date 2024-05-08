from typing import Any, Dict, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


from src.models.components.loss_functions import loss_distance_between_image_pairs

class ViewInvariantEmbeddingModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net 

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
    ) -> torch.Tensor:
        
        """
        batch: dict{ 
            prompt_embeddings: torch.tensor (batch size, embedding size), 
            img_embeddings: torch.tensor (batch size, data points size, embedding size)}
            distances between text and image embedding: torch.tensor (batch size, data points size)
        """

        text_prompt_embeddings = batch["prompt_embedding"]
        distances = batch["distances"]

        original_img_embeddings = batch["image_embeddings"]
        predicted_img_embeddings = self.forward(original_img_embeddings)

        #Assuming datapoint size = 2, aka using 2 images per object.
        # sim = pairwise_cosine_similarity(predicted_img_embeddings, reduction='sum')
        sim = loss_distance_between_image_pairs(predicted_img_embeddings)
        return sim

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:

        loss = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        original_img_embeddings = batch["image_embeddings"]
        predicted_img_embeddings = self.forward(original_img_embeddings)
        sim = pairwise_cosine_similarity(predicted_img_embeddings[0], predicted_img_embeddings[0])
        self.logger.log_image(key='heatmap', images=[sim])
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # def on_train_epoch_end(self) -> None:
    #     "Lightning hook that is called when a training epoch ends."
    #     pass

    # def on_train_start(self) -> None:
    #     "Lightning hook that is called when training starts."
    #     pass

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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