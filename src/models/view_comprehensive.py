from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


from src.models.components.vlm_head import VLMHead
from src.models.components.loss_functions import loss_distance_between_image_pairs, loss_autoencoder_embedding

class ViewComprehensiveEmbeddingModule(LightningModule):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.view_invariant_head = VLMHead()
        self.view_dependent_head = VLMHead()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, img_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ img_embeddings: torch.tensor (batch size, data points size, embedding size)"""
        #TODO perform reshaping
        vi_embedding = self.view_invariant_head(img_embeddings)
        vd_embedding = self.view_dependent_head(img_embeddings)

        view_comprehensive_embedding = vi_embedding + vd_embedding

        return view_comprehensive_embedding, vi_embedding

    def model_step(
        self, batch: Dict[torch.tensor, torch.tensor, torch.tensor]
    ) -> torch.Tensor:
        
        """
        batch: dict{ 
            prompt_embeddings: torch.tensor (batch size, embedding size), 
            original_img_embeddings: torch.tensor (batch size, data points size, embedding size)}
            distances between text and image embedding: torch.tensor (batch size, data points size)
        """
        
        # text_prompt_embeddings = batch["text_prompt_embedding"]
        # distances = batch["distances"]
        # batch_size, data_points_size, embedding_size = original_img_embeddings.size()

        original_img_embeddings = batch["original_img_embeddings"]
        view_comprehensive_decodings, view_independent_encodings = self.forward(original_img_embeddings)

        loss_auto = loss_autoencoder_embedding(original_img_embeddings, view_comprehensive_decodings)
        loss_vi = loss_distance_between_image_pairs(view_independent_encodings)
        
        return loss_auto + loss_vi

    def training_step(
        self, batch: Dict[torch.tensor, List[torch.tensor]], batch_idx: int
    ) -> torch.Tensor:

        loss = self.model_step(batch)
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[torch.tensor, List[torch.tensor]], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.val_loss.update(loss)
        self.log("val/loss", self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[torch.tensor, List[torch.tensor]], batch_idx: int) -> None:
        loss= self.model_step(batch)

        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_train_start(self) -> None:
        "Lightning hook that is called when training starts."
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
