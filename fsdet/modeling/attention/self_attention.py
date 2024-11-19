import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Flatten height and width into a single dimension: (B, C, H, W) -> (H*W, B, C)
        x_flat = x.view(batch_size, channels, -1).permute(2, 0, 1)  # Shape: (H*W, B, C)
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        # Residual connection + layer normalization
        attn_output = self.norm(x_flat + attn_output)
        # Reshape back: (H*W, B, C) -> (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        return attn_output
