#create simple endocer

import torch
from torch import nn

class VLMHead(nn.Module):

    def __init__(
        self,
        input_size: int = 512,
        lin1_size: int = 512,
        lin2_size: int = 512,
        output_size: int = 512,
    ) -> None:
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)