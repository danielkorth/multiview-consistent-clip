import torch
from torch import nn
from typing import Tuple
import numpy as np

from src.models.components.vlm_head import VLMHead


class VLMAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_input_size: int = 512,
        encoder_n_hidden_layers: int = 2,
        encoding_size: int = 128,
        act_fct: nn.Module = nn.ReLU(),
    ) -> None:
        
        super().__init__()
        self.view_invariant_encoder = VLMCoder(
            input_size=encoder_input_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoding_size,
            act_fct=act_fct, 
            is_encoder=True
        )
        
        self.view_dependent_encoder = VLMCoder(
            input_size=encoder_input_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoding_size,
            act_fct=act_fct, 
            is_encoder=True
        )
        
        self.view_invariant_decoder = VLMCoder(
            input_size=encoding_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoder_input_size,
            act_fct=act_fct, 
            is_encoder=False
        )
        
        self.view_dependent_decoder = VLMCoder(
            input_size=encoding_size,
            n_hidden_layers=encoder_n_hidden_layers,
            output_size=encoder_input_size,
            act_fct=act_fct, 
            is_encoder=False
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

        return {
            "decoded": view_comprehensive_decoding.view((batch_size, datapoint_size, vlm_embedding_size)),
            "vi_encoding": vi_encoding.view(*(batch_size, datapoint_size, -1)),
            "vd_decoding": vd_decoding.view(*(batch_size, datapoint_size, -1))
        }
    
    def forward_view_independent(self, img_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ img_embeddings: torch.tensor (batch size, data points size, embedding size)"""
        
        batch_size, datapoint_size, vlm_embedding_size = img_embeddings.shape
        img_embeddings = img_embeddings.view(-1, vlm_embedding_size)

        vi_encoding = self.view_invariant_encoder(img_embeddings)
        vi_decoding = self.view_invariant_decoder(vi_encoding)

        view_comprehensive_decoding = vi_decoding

        return {
            "decoded": view_comprehensive_decoding.view((batch_size, datapoint_size, vlm_embedding_size)),
            "vi_encoding": vi_encoding.view(*(batch_size, datapoint_size, -1)),
        }

    def paramters(self, recurse: bool = True):
        for net in [self.view_invariant_encoder, self.view_dependent_encoder, self.view_invariant_decoder, self.view_dependent_decoder]:
            for name, param in net.named_parameters(recurse=recurse):
                yield param



class VLMCoder(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        n_hidden_layers: int = 2,
        output_size: int = 128,
        dropout: float = 0.0,
        act_fct: nn.Module = nn.ReLU(),
        is_encoder: bool = True
    ) -> None:
        
        super().__init__()


        linear_sizes = [
            int(input_size - i * (input_size - output_size) / (n_hidden_layers + 1))
            for i in range(n_hidden_layers + 2)
        ]
        layer_sizes = [1 << (size - 1).bit_length() for size in linear_sizes]

        self.model = nn.Sequential()
        for i in range(n_hidden_layers+1):
            self.model.add_module('linear' + str(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.model.add_module('activation' + str(i), act_fct)
            if dropout > 0.0:
                self.model.add_module('dropout' + str(i), nn.Dropout(dropout))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
