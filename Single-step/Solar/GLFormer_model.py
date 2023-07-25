""""
This script defines the GLFormer and its spatial Positional Encoding module.

The main components of the script are:

1. LearnedPositionalEncoding: A class defining the learned positional encoding, which is used for spatial embedding in the GLFormer.

2. GLFormer: The main class defining the GLFormer module.

"""


# Import necessary library
import torch.nn as nn
# Import light transformer components
from transformer import LightformerLayer, Lightformer

# Define the Learned Positional Encoding, which is used for spatial embedding in GLformers
class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, max_len = 137):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = 0.1)
    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        # Add the embedding vector to the feature maps
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class GLFormer(nn.Module):
    def __init__(self, d_model, nglf):
        super(GLFormer, self).__init__()
        # Define the GLFormer module
        self.layers = nglf
        self.hid_dim =d_model
        self.attention_layer = LightformerLayer(self.hid_dim, 8, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.Former = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim)

    def forward(self,input):
        x = input.permute(1,0,2)
        # Apply the learned positional encoding for spatial embedding
        x = self.lpos(x)
        # Send the embedded feature maps with the spatial mask to the GLFormer module
        output = self.Former(x)
        return output.permute(1,0,2)
