import torch
from torch import nn
from typing import Tuple

from src.models.components.vlm_head import VLMHead


class VLMAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_input_size: int = 512,
        encoder_hidden_size: int = 128,
        encoder_n_hidden_layers: int = 2,
        encoder_output_size: int = 128,
        decoder_input_size: int = 128,
        decoder_hidden_size: int = 128,
        decoder_n_hidden_layers: int = 2,
        decoder_output_size: int = 512,
        act_fct: nn.Module = nn.ReLU(),
    ) -> None:
        
        super().__init__()
        self.view_invariant_encoder = VLMHead(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoder_output_size,
            act_fct=act_fct
        )
        
        self.view_dependent_encoder = VLMHead(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoder_output_size,
            act_fct=act_fct
        )
        
        self.view_invariant_decoder = VLMHead(
            input_size=decoder_input_size,
            hidden_size=decoder_hidden_size,
            n_hidden_layers=decoder_n_hidden_layers,
            output_size=decoder_output_size,
            act_fct=act_fct
        )
        
        self.view_dependent_decoder = VLMHead(
            input_size=decoder_input_size,
            hidden_size=decoder_hidden_size,
            n_hidden_layers=decoder_n_hidden_layers,
            output_size=decoder_output_size,
            act_fct=act_fct
        )

    def forward(self, img_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ img_embeddings: torch.tensor (batch size, data points size, embedding size)"""
        
        batch_size, datapoint_size, vlm_embedding_size = img_embeddings.shape
        img_embeddings = img_embeddings.view(-1, vlm_embedding_size)

        vi_encoding = self.view_invariant_encoder(img_embeddings)
        vd_encoding = self.view_dependent_encoder(img_embeddings)

        vi_decoding = self.view_invariant_decoder(vi_encoding)
        vd_decoding = self.view_dependent_decoder(vd_encoding)

        view_comprehensive_decoding = vi_decoding + vd_decoding

        # TODO IF TIME PERMITS ;)
        # vd_decoding = F.normalize(vd_decoding, p=2, dim=1)

        return (
            view_comprehensive_decoding.view((batch_size, datapoint_size, vlm_embedding_size)), 
            vi_encoding.view(*(batch_size, datapoint_size, -1))
        )
    #TODO check if this works. used in config optimizer.
    def paramters(self, recurse: bool = True):
        for net in [self.view_invariant_encoder, self.view_dependent_encoder, self.view_invariant_decoder, self.view_dependent_decoder]:
            for name, param in net.named_parameters(recurse=recurse):
                yield param
    
    def forward_view_independent(self, img_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ img_embeddings: torch.tensor (batch size, data points size, embedding size)"""
        
        batch_size, datapoint_size, vlm_embedding_size = img_embeddings.shape
        img_embeddings = img_embeddings.view(-1, vlm_embedding_size)

        vi_encoding = self.view_invariant_encoder(img_embeddings)
        vi_decoding = self.view_invariant_decoder(vi_encoding)

        view_comprehensive_decoding = vi_decoding

        return view_comprehensive_decoding.view((batch_size, datapoint_size, vlm_embedding_size))