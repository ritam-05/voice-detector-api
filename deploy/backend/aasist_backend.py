import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for spectro-temporal graph attention.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.out_dim = out_dim
        
        if concat:
            assert out_dim % num_heads == 0
            self.head_dim = out_dim // num_heads
        else:
            self.head_dim = out_dim
            
        self.W = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h):
        """
        h: (B, T, in_dim)
        Returns: (B, T, out_dim) if concat else (B, T, head_dim)
        """
        B, T, _ = h.size()
        
        # Linear transformation: (B, T, head_dim * num_heads)
        Wh = self.W(h)
        Wh = Wh.view(B, T, self.num_heads, self.head_dim)  # (B, T, num_heads, head_dim)
        Wh = Wh.permute(0, 2, 1, 3)  # (B, num_heads, T, head_dim)
        
        # Compute attention scores using query-key mechanism
        # Split into query and key
        query = Wh  # (B, num_heads, T, head_dim)
        key = Wh    # (B, num_heads, T, head_dim)
        
        # Compute attention scores: (B, num_heads, T, T)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply leaky relu and softmax
        attention = F.softmax(self.leakyrelu(scores), dim=-1)  # (B, num_heads, T, T)
        
        # Apply attention to values
        # attention: (B, num_heads, T, T), Wh: (B, num_heads, T, head_dim)
        h_prime = torch.matmul(attention, Wh)  # (B, num_heads, T, head_dim)
        
        # Rearrange back
        h_prime = h_prime.permute(0, 2, 1, 3)  # (B, T, num_heads, head_dim)
        
        if self.concat:
            # Concatenate heads: (B, T, out_dim)
            return h_prime.reshape(B, T, -1)
        else:
            # Average heads: (B, T, head_dim)
            return h_prime.mean(dim=2)


class StackedGraphAttention(nn.Module):
    """
    Stacked Graph Attention layers with residual connections.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, num_heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
    def forward(self, x):
        """
        x: (B, T, input_dim)
        Returns: (B, T, hidden_dim)
        """
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h)
            h = norm(h_new + h if h.size(-1) == h_new.size(-1) else h_new)
        return h


class AASIST_Backend(nn.Module):
    """
    AASIST-inspired backend for Stage2Verifier.
    
    Takes Wav2Vec2 features (B, T, 768) and applies:
    - Spectro-temporal graph attention
    - Adaptive pooling
    - Classification head
    
    Returns: logits (B,) for binary classification
    """
    def __init__(self, input_dim=768, hidden_dim=128, num_graph_layers=2, num_heads=4):
        super().__init__()
        
        # Graph attention network for spectro-temporal modeling
        self.graph_attention = StackedGraphAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_graph_layers,
            num_heads=num_heads
        )
        
        # Adaptive pooling strategies
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 for mean + attention pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, features):
        """
        features: (B, T, input_dim) from Wav2Vec2 encoder
        Returns: logits (B,) for binary classification
        """
        # Apply graph attention
        h = self.graph_attention(features)  # (B, T, hidden_dim)
        
        # Mean pooling
        h_mean = h.mean(dim=1)  # (B, hidden_dim)
        
        # Attention pooling
        attn_weights = self.attention_pool(h)  # (B, T, 1)
        h_attn = (h * attn_weights).sum(dim=1)  # (B, hidden_dim)
        
        # Concatenate pooling strategies
        h_combined = torch.cat([h_mean, h_attn], dim=-1)  # (B, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(h_combined)  # (B, 1)
        
        return logits.squeeze(-1)  # (B,)
