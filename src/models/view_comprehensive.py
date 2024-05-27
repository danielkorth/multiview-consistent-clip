from typing import Any, Dict 
from pathlib import Path

import torch
from lightning import LightningModule
from src.utils.visualize import save_matrix_png
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class ViewComprehensiveEmbeddingModule(LightningModule):

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
        self.save_hyperparameters(logger=False, ignore=["net", "cfg"])

        self.net = net
        self.loss = loss
        self.cfg = cfg

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
        out = self.net.forward(original_img_embeddings)

        return self.loss(original_img_embeddings, out['decoded'], out['vi_encoding'], out['vd_decoding'])

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

        # calculate the regular loss
        loss = self.model_step(batch)['loss']

        text_embeddings = batch["prompt_embedding"]
        original_img_embeddings = batch["image_embeddings"]
        predicted_image_embeddings = self.net.forward_view_independent(original_img_embeddings)['decoded'] # [batch_size, data_points_size, embedding_size]
        # predicted_image_embeddings = original_img_embeddings
        batch_size, data_points_size, embedding_size = predicted_image_embeddings.shape

        # calculate pairwise cosine similarity between predicted image embeddings
        sim = pairwise_cosine_similarity(predicted_image_embeddings.view(-1, embedding_size), predicted_image_embeddings.view(-1, embedding_size))
        similarity_mat = torch.stack([sim[i:i+data_points_size, i:i+data_points_size] for i in range(0, sim.shape[0], data_points_size)]).squeeze()

        # calculate text to images similarity
        temp = pairwise_cosine_similarity(text_embeddings, predicted_image_embeddings.view(-1, embedding_size))
        t2i_similarity = torch.stack([temp[i, (i*data_points_size):(i*data_points_size)+data_points_size] for i in range(0, batch_size)])
        mean_t2i = t2i_similarity.mean()
        mean_t2i_per_object = t2i_similarity.mean(dim=0)
        std_t2i = t2i_similarity.std()
        std_t2i_per_object = t2i_similarity.std(dim=0)

        save_matrix_png(similarity_mat.mean(dim=0).cpu(), Path(self.cfg.paths.output_dir) / "sim_mean.png", type='mean')
        save_matrix_png(similarity_mat.std(dim=0).cpu(), Path(self.cfg.paths.output_dir) / "sim_std.png", type='std')

        # capture all metrics in a dictionary
        metrics = {
            "loss": loss.cpu().item(),
            "mean_t2i": mean_t2i.cpu().item(),
            "mean_t2i_per_object": mean_t2i_per_object.cpu().tolist(),
            "std_t2i": std_t2i.cpu().item(),
            "std_t2i_per_object": std_t2i_per_object.cpu().tolist()
        }
        with open(Path(self.cfg.paths.output_dir) / 'metrics.csv', 'w') as f:
            for key in metrics.keys():
                f.write("%s,%s\n"%(key,metrics[key]))
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
