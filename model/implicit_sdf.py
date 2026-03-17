import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitSDF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_controls = cfg["model"]["implicit_sdf"]["control_points"]
        self.hidden_dim = cfg["model"]["implicit_sdf"]["hidden_dim"]
        self.num_layers = cfg["model"]["implicit_sdf"]["num_layers"]
        self.pe_L = 10

        self.control_points = nn.Parameter(torch.randn(self.num_controls, 3) * 0.1)

        in_dim = 3 + 3 + 6 * self.pe_L

        layers = []
        cur_dim = in_dim
        for i in range(self.num_layers):
            out_dim = self.hidden_dim if i < self.num_layers - 1 else 1
            layers.append(nn.Linear(cur_dim, out_dim))
            if i < self.num_layers - 1:
                layers.append(nn.SiLU())
            cur_dim = self.hidden_dim
        self.mlp = nn.Sequential(*layers)

    def positional_encoding(self, x, L=10):
        device = x.device
        freqs = 2.0 ** torch.linspace(0.0, L - 1, L, device=device)

        x_pe = x.unsqueeze(-1) * freqs.reshape(1, 1, 1, -1)

        pe = torch.cat([torch.sin(x_pe), torch.cos(x_pe)], dim=-1)
        pe = pe.reshape(x.shape[0], x.shape[1], -1)
        return pe

    def forward(self, points):
        B, N, _ = points.shape
        device = points.device

        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        scale = centered.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True)
        points_norm = centered / (scale + 1e-8)

        control_exp = self.control_points.unsqueeze(0).unsqueeze(0)
        dists = (points_norm.unsqueeze(2) - control_exp).norm(dim=-1)
        weights = F.softmin(dists * 10.0, dim=-1)
        influence = (weights.unsqueeze(-1) * self.control_points).sum(dim=2)

        pe = self.positional_encoding(points_norm, L=self.pe_L)

        x = torch.cat([points_norm, influence, pe], dim=-1)

        x = x.reshape(B * N, -1)
        sdf = self.mlp(x).reshape(B, N, 1)
        return sdf

    def get_gradient(self, points):
        points = points.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            sdf = self.forward(points)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=points,
                grad_outputs=torch.ones_like(sdf),
                create_graph=self.training,
                retain_graph=self.training,
                only_inputs=True
            )[0]
        return gradients