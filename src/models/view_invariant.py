from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from src.models.components.loss_functions import loss_contrastive

import matplotlib.pyplot as plt

def save_matrix_png(sim, path, type: str = 'mean'):
    plt.figure(figsize=(10, 10))
    plt.imshow(sim, cmap='hot', interpolation='nearest')
    if type == 'mean':
        plt.title('Mean Cosine Similarity')
        plt.clim(0.5, 1)
    else: 
        plt.title('Std Cosine Similarity')
        plt.clim(0, 0.4) 
    plt.colorbar()
    plt.savefig(path)

class ViewInvariantEmbeddingModule(LightningModule):
    def __init__(
        self,

        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        cfg: Dict[str, Any] = None,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'cfg'])

        self.net = net 
        self.cfg = cfg

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

        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.forward(original_img_embeddings)

        # return ((predicted_image_embeddings - original_img_embeddings).mean(), 0, 0)

        return loss_contrastive(text_embeddings, predicted_image_embeddings)

    def training_step(
        self, batch: Dict[str, torch.tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, sim_score, dissim_score = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/sim_score", sim_score, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/dissim_score", dissim_score, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:

        loss, sim_score, dissim_score = self.model_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/sim_score", sim_score, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/dissim_score", dissim_score, on_step=True, on_epoch=True, prog_bar=False)

        # TODO fix and uncomment

        # sim = pairwise_cosine_similarity(predicted_img_embeddings[0], predicted_img_embeddings[0])
        # self.logger.log_image(key='heatmap', images=[sim])

        # should go down (learn mv consistency)
        # self.log("val/mean_cossim", sim.mean(), on_step=True, on_epoch=True, prog_bar=False)
        # self.log("val/std_cossim", sim.std(), on_step=True, on_epoch=True, prog_bar=True)
        
        # should not go down (dont unlearn)
    
    # def on_test_epoch_start(self) -> None:
    #     super().on_validation_epoch_start()
    #     self.test_output_list = []
    #     return 

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int) -> None:

        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.forward(original_img_embeddings) # [batch_size, data_points_size, embedding_size]
        predicted_image_embeddings = original_img_embeddings
        batch_size, data_points_size, embedding_size = predicted_image_embeddings.shape

        # calculate pairwise cosine similarity between predicted image embeddings
        sim = pairwise_cosine_similarity(predicted_image_embeddings.view(-1, embedding_size), predicted_image_embeddings.view(-1, embedding_size))
        similarity_mat = torch.stack([sim[i:i+data_points_size, i:i+data_points_size] for i in range(0, sim.shape[0], data_points_size)]).squeeze()

        # calculate loss
        loss, _, _ = loss_contrastive(text_embeddings, predicted_image_embeddings)

        # calculate text to images similarity
        temp = pairwise_cosine_similarity(text_embeddings, predicted_image_embeddings.view(-1, embedding_size))
        t2i_similarity = torch.stack([temp[i, (i*data_points_size):(i*data_points_size)+data_points_size] for i in range(0, batch_size)])
        mean_t2i = t2i_similarity.mean()
        mean_t2i_per_object = t2i_similarity.mean(dim=0)
        std_t2i = t2i_similarity.std()
        std_t2i_per_object = t2i_similarity.std(dim=0)

        save_matrix_png(similarity_mat.mean(dim=0), Path(self.cfg.paths.output_dir) / "sim_mean.png", type='mean')
        save_matrix_png(similarity_mat.std(dim=0), Path(self.cfg.paths.output_dir) / "sim_std.png", type='std')

        # capture all metrics in a dictionary
        metrics = {
            "loss": loss.item(),
            "mean_t2i": mean_t2i.item(),
            "mean_t2i_per_object": mean_t2i_per_object.tolist(),
            "std_t2i": std_t2i.item(),
            "std_t2i_per_object": std_t2i_per_object.tolist()
        }
        with open(Path(self.cfg.paths.output_dir) / 'metrics.csv', 'w') as f:
            for key in metrics.keys():
                f.write("%s,%s\n"%(key,metrics[key])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # self.test_output_list.append({"loss": loss, "similarity_matrix": similarity_mat})
    
    # def on_test_epoch_end(self) -> None:
    #     # compose the test_output_list
    #     outputs = {}
    #     for key in self.test_output_list[0].keys():
    #         outputs[key] = [d[key] for d in self.test_output_list]

    #     loss = torch.stack(outputs['loss'])
    #     sim = torch.stack(outputs['similarity_matrix']).squeeze()

    #     # save thes metrics
    #     save_matrix_png(sim.mean(dim=0), Path(self.cfg.paths.output_dir) / "sim_mean.png", type='mean')
    #     save_matrix_png(sim.std(dim=0), Path(self.cfg.paths.output_dir) / "sim_std.png", type='std')
    #     with open(Path(self.cfg.paths.output_dir) / "loss.txt", "w") as f:
    #         f.write(str(float(loss.item())))


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
    