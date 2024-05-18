from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from src.models.components.loss_functions import loss_contrastive


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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        temperature: int = 1
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


        # references: 
        ### CLIP paper (has a dummy implementation!!)
        # https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
        ### 

        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.forward(original_img_embeddings)

        return loss_contrastive(text_embeddings, predicted_image_embeddings)

        # ----- Daniel's implementation -----

        # # repeat text_embeddings to fix size of image embeddings
        # text_embeddings = text_embeddings.unsqueeze(1).repeat(1, image_embeddings.shape[1], 1)

        # # reshape
        # text_embeddings = text_embeddings.reshape(-1, text_size[1])
        # image_embeddings = image_embeddings.reshape(-1, text_size[1])

        
        # # Calculating the Loss
        # logits = (text_embeddings @ image_embeddings.T) / self.hparams['temperature']
        # cross_entropy = nn.CrossEntropyLoss(reduction='none')
        # texts_loss = cross_entropy(logits, targets)
        # images_loss = cross_entropy(logits.T, targets.T)
        # loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # return loss.mean()

        # ----- End of Daniel's implementation -----

        # #Assuming datapoint size = 2, aka using 2 images per object.
        # sim = pairwise_cosine_similarity(predicted_img_embeddings, reduction='mean')
        # sim = loss_distance_between_image_pairs(predicted_img_embeddings)
        # return sim, predicted_img_embeddings

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, loss_sim, loss_dissim = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_sim", loss_sim, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_dissim", loss_dissim, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:

        original_img_embeddings = batch["image_embeddings"]
        batch_size, data_points_size, embedding_size = original_img_embeddings.shape
        predicted_image_embeddings = self.forward(original_img_embeddings)

        loss_similarity = 0
        for b in range(batch_size):
            sim =  pairwise_cosine_similarity(predicted_image_embeddings[b])
            loss_similarity += torch.triu(sim, diagonal=1).sum() / (data_points_size * (data_points_size - 1) / 2)
        mean_loss_similarity = loss_similarity / batch_size

        loss_dissimilarity = 0
        for batch_idx in range(batch_size):
            for nested_batch_idx in range (batch_idx, batch_size):
                sim = pairwise_cosine_similarity(predicted_image_embeddings[batch_idx], predicted_image_embeddings[nested_batch_idx])
                loss_dissimilarity += sim[1:].sum() / (data_points_size**2 + data_points_size)

        mean_loss_dissimilarity = loss_dissimilarity / (batch_size * (batch_size + 1) / 2)

        self.log("val/mean_loss_similarity", mean_loss_similarity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/mean_loss_dissimilarity", mean_loss_dissimilarity, on_step=True, on_epoch=True, prog_bar=True)

        loss, loss_similarity, loss_dissimilarity = self.model_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # TODO fix and uncomment

        # sim = pairwise_cosine_similarity(predicted_img_embeddings[0], predicted_img_embeddings[0])
        # self.logger.log_image(key='heatmap', images=[sim])

        # should go down (learn mv consistency)
        # self.log("val/mean_cossim", sim.mean(), on_step=True, on_epoch=True, prog_bar=False)
        # self.log("val/std_cossim", sim.std(), on_step=True, on_epoch=True, prog_bar=True)
        
        # should not go down (dont unlearn)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:
        loss, loss_sim, loss_dissim = self.model_step(batch)
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
    