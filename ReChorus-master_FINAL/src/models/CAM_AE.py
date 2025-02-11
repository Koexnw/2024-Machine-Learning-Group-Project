import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CAM_AE(nn.Module):
    """
    CAM-AE: The neural network architecture for learning the data distribution in the reverse diffusion process.
    First-hop neighbors (direct neighbors) are to be integrated.
    """
    def __init__(self, d_model, num_heads, num_layers, in_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(CAM_AE, self).__init__()
        self.in_dims = in_dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.num_layers = num_layers

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip([d_model,d_model], [d_model,d_model])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip([d_model,d_model], [d_model,d_model])])
        self.forward_layers = nn.ModuleList([nn.Linear(d_model, d_model) \
                                         for i in range(num_layers)])
        self.dim_inters = 650
        self.first_hop_embedding = nn.Linear(1, d_model) # Expend dimension
        self.first_hop_decoding = nn.Linear(d_model, 1)
        self.second_hop_embedding = nn.Linear(1, d_model) # Expend dimension
        self.final_out = nn.Linear(self.dim_inters+emb_size, self.dim_inters)

        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(0.8)
        self.drop2 = nn.Dropout(dropout)

        self.encoder = nn.Linear(self.in_dims, self.dim_inters) # Get the encoded vector for user-item interactions
        self.decoder = nn.Linear(self.dim_inters+emb_size, self.in_dims)
        self.encoder2 = nn.Linear(self.in_dims, self.dim_inters)

        # Attention layer
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.5, batch_first=True)
            for i in range(num_layers)
        ])

        self.time_emb_dim = emb_size
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_sec_hop, timesteps):

        x = self.encoder(x)
        h_sec_hop = self.encoder(x_sec_hop)

        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        h = h.unsqueeze(-1)
        h = self.first_hop_embedding(h)

        h_sec_hop = torch.cat([h_sec_hop, emb], dim=-1)
        h_sec_hop = h_sec_hop.unsqueeze(-1)
        h_sec_hop = self.second_hop_embedding(h_sec_hop)

        for i in range(self.num_layers):

            attention_layer = self.self_attentions[i]
            attention, attn_output_weights = attention_layer(h_sec_hop, h, h)

            attention = self.drop1(attention)
            h = h + attention
            #h = self.norm1(h)
            h = self.drop2(h)
            forward_pass = self.forward_layers[i]
            h = forward_pass(h)

            if i != self.num_layers - 1:
                h = torch.tanh(h)

        h = self.first_hop_decoding(h)
        h = torch.squeeze(h)
        h = torch.tanh(h)
        h = self.decoder(h)

        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
