from typing import Any, Dict
from pathlib import Path

import torch
from lightning import LightningModule
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from src.utils.visualize import plot_object_similarity, plot_all_similarity 


class VLMHead(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        cfg: Dict[str, Any] = None,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net 
        self.cfg = cfg

        self.loss = loss

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
        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.forward(original_img_embeddings) # [batch_size, data_points_size, embedding_size]
        batch_size, data_points_size, embedding_size = predicted_image_embeddings.shape

        # calculate loss 
        loss = self.loss(text_embeddings, predicted_image_embeddings)
        loss = loss['loss']

        # calculate text to images similarity
        temp = pairwise_cosine_similarity(text_embeddings, predicted_image_embeddings.view(-1, embedding_size))
        t2i_similarity = torch.stack([temp[i, (i*data_points_size):(i*data_points_size)+data_points_size] for i in range(0, batch_size)])
        mean_t2i = t2i_similarity.mean()
        std_t2i = t2i_similarity.std()

        # calculate similarity matrix for single object
        sim = pairwise_cosine_similarity(predicted_image_embeddings.view(-1, embedding_size), predicted_image_embeddings.view(-1, embedding_size))
        similarity_mat = torch.stack([sim[i:i+data_points_size, i:i+data_points_size] for i in range(0, sim.shape[0], data_points_size)]).squeeze()
        plot_object_similarity(x=similarity_mat.mean(dim=0).cpu(),
                            path=Path(self.cfg.paths.output_dir),
                            title="Mean Cosine Similarity between single object embeddings",
                            type='mean')
        plot_object_similarity(x=similarity_mat.std(dim=0).cpu(),
                            path=Path(self.cfg.paths.output_dir),
                            title="Std Cosine Similarity between single object embeddings",
                            type='mean')

        # calculate similarity matrix for all (+ text)
        a = torch.cat((text_embeddings, predicted_image_embeddings.view(-1, embedding_size)))
        b = pairwise_cosine_similarity(a)
        plot_all_similarity(x=b.cpu(),
                            path=Path(self.cfg.paths.output_dir),
                            title="Cosine Similarity between all objects & text", 
                            type='mean')

        # calculate inter and intra object similarity
        intra_object_mask = torch.ones_like(sim).bool()
        for i in range(0, sim.shape[0], data_points_size):
            intra_object_mask[i:i+data_points_size, i:i+data_points_size] = 0
        intra_object_similarity = sim[intra_object_mask].mean()
        intra_object_similarity_std = sim[intra_object_mask].std()

        inter_object_mask = (~intra_object_mask) & (~torch.eye(intra_object_mask.shape[0]).to(intra_object_mask.device).bool())
        inter_object_similarity = sim[inter_object_mask].mean()
        inter_object_similarity_std = sim[inter_object_mask].std()

        # capture all metrics in a dictionary
        metrics = {
            "loss": loss.cpu().item(),
            "mean_t2i": mean_t2i.cpu().item(),
            "std_t2i": std_t2i.cpu().item(),
            "intra_object_similarity": intra_object_similarity.cpu().item(),
            "intra_object_similarity_std": intra_object_similarity_std.cpu().item(),
            "inter_object_similarity": inter_object_similarity.cpu().item(),
            "inter_object_similarity_std": inter_object_similarity_std.cpu().item(),
        }
        with open(Path(self.cfg.paths.output_dir) / 'metrics.csv', 'w') as f:
            for key in metrics.keys():
                f.write("%s,%s\n"%(key,metrics[key]))


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
    