import torch
from torch import nn

class VLMHeadModel(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 512,
        n_hidden_layers: int = 2, # number of hidden layers
        output_size: int = 512,
        dropout: float = 0.0,
        act_fct: nn.Module = nn.ReLU(),
    ) -> None:
        
        super().__init__()

        self.model = nn.Sequential()
        for i in range(n_hidden_layers+1):
            if i == 0:
                self.model.add_module('linear' + str(i), nn.Linear(input_size, hidden_size))
                self.model.add_module('activation' + str(i), act_fct)
                if dropout > 0.0:
                    self.model.add_module('dropout' + str(i), nn.Dropout(dropout))
            elif i == n_hidden_layers:
                self.model.add_module('linear' + str(i), nn.Linear(hidden_size, output_size))
            else:
                self.model.add_module('linear' + str(i), nn.Linear(hidden_size, hidden_size))
                self.model.add_module('activation' + str(i), act_fct)
                if dropout > 0.0:
                    self.model.add_module('dropout' + str(i), nn.Dropout(dropout))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)