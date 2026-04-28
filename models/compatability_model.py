from __future__ import annotations
import torch 
import torch.nn as nn 

class CompatabilityModel(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        input_dim = 2 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), 
            nn.Dropout(0.1), 

            nn.Linear(256, 128), 
            nn.ReLU(), 

            nn.Dropout(0.1), 
            nn.Linear(128, 1)
        )
        
    def forward(self, p:torch.Tensor, z_c:torch.Tensor)-> torch.Tensor:
        """
        Args: 
            p : page embedding 
            z_c : candidate card embedding 
        Returns:
            score 

        """
        x = torch.cat([p, z_c], dim=-1)
        out = self.mlp(x) 
        return out 