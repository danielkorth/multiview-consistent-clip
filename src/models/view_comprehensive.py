from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric


from src.models.components.vlm_embedding_encoder import VLMEmbeddingEncoder
from src.models.components.vlm_embedding_decoder import VLMEmbeddingDecoder

class ViewComprehensiveEmbeddingModule(LightningModule):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.view_invariant_encoder = VLMEmbeddingEncoder()
        self.view_invariant_decoder = VLMEmbeddingDecoder()
        self.view_dependent_encoder = VLMEmbeddingEncoder()
        self.view_dependent_decoder = VLMEmbeddingDecoder()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, img_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vi_encoding = self.view_invariant_encoder(img_embedding)
        vi_decoding = self.view_invariant_decoder(vi_encoding)

        vd_encoding = self.view_dependent_encoder(img_embedding)
        vd_decoding = self.view_dependent_decoder(vd_encoding)

        view_comprehensive_decoding = vi_decoding + vd_decoding

        return view_comprehensive_decoding, vi_encoding


    def _loss(
            self, 
            text_prompt_embedding: torch.tensor,
            original_img_embeddings: List[torch.tensor],
            decoded_img_embeddings: List[torch.tensor],
            view_independent_encodings: List[torch.tensor]
        ) -> torch.tensor:

        # Autoencoder loss - dist between original and decoded img embeddings
        

        # View independent loss - ...

        raise NotImplementedError

    def model_step(
        self, batch: Dict[torch.tensor, List[torch.tensor]]
    ) -> torch.Tensor:
        
        text_prompt_embedding = batch["text_prompt_embedding"]
        original_img_embeddings = batch["original_img_embeddings"]

        #TODO run inference on larger batches?
        view_comprehensive_decodings, view_independent_encodings = self.forward(torch.stack(original_img_embeddings))
        
        return self._loss(
            text_prompt_embedding,
            original_img_embeddings,
            list(torch.unbind(view_comprehensive_decodings)), 
            list(torch.unbind(view_independent_encodings)))

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
    _ = ViewComprehensiveEmbeddingModule()
