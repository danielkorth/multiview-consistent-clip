from typing import Any, Dict, List

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


from src.models.components.vlm_head import VLMHead
from src.models.components.loss_functions import loss_view_invariant_embedding

class ViewInvariantEmbeddingModule(LightningModule):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = VLMHead()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, img_embedding: torch.Tensor) -> torch.Tensor:
        return self.net.forward(img_embedding)

    def model_step(
        self, batch: Dict[torch.tensor, List[torch.tensor]]
    ) -> torch.Tensor:

        text_prompt_embedding = batch["text_prompt_embedding"]
        original_img_embeddings = batch["original_img_embeddings"]

        #TODO run inference on larger batches?
        predicted_img_embeddings = self.forward(torch.stack(original_img_embeddings))
        predicted_img_embeddings_list = list(torch.unbind(predicted_img_embeddings))

        return loss_view_invariant_embedding(
            text_prompt_embedding,
            original_img_embeddings,
            predicted_img_embeddings_list)

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


if __name__ == "__main__":
    _ = ViewInvariantEmbeddingModule()
