import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.bn = nn.BatchNorm1d(channels)

        self.pos_proj = nn.Linear(3, channels)

    def forward(self, x, pos):
        B, C, N = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        pos_emb = self.pos_proj(pos)
        pos_emb = pos_emb.view(B, N, self.num_heads, self.head_dim)
        pos_emb = pos_emb.transpose(1, 2)

        q = q + pos_emb
        k = k + pos_emb

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out = attn_out.transpose(1, 2).reshape(B, C, N)
        out = self.proj(out)
        out = self.bn(out)
        return out

class PointTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.attn = LocalAttention(channels, num_heads)
        self.norm2 = nn.BatchNorm1d(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv1d(channels * 4, channels, 1),
        )

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x

class PointTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg["model"]["point_transformer"]["embed_dim"]
        depths = cfg["model"]["point_transformer"]["depth"]
        num_heads = cfg["model"]["point_transformer"]["num_heads"]

        self.stem = nn.Sequential(
            nn.Conv1d(3, embed_dim // 2, 1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
        )

        self.blocks = nn.ModuleList()
        cur_dim = embed_dim
        for i, depth in enumerate(depths):
            for _ in range(depth):
                self.blocks.append(PointTransformerBlock(cur_dim, num_heads[i]))
            if i < len(depths) - 1:
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(cur_dim, cur_dim * 2, 1),
                    nn.BatchNorm1d(cur_dim * 2),
                    nn.GELU(),
                ))
                cur_dim *= 2

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(cur_dim, cfg["model"]["fusion"]["hidden_dim"])

    def forward(self, points):
        x = points.permute(0, 2, 1)
        pos = points

        x = self.stem(x)

        for block in self.blocks:
            if isinstance(block, PointTransformerBlock):
                x = block(x, pos)
            else:
                x = block(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x