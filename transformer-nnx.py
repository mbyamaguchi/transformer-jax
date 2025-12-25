from flax import nnx
import jax
import jax.numpy as jnp
from typing import Optional

# 1. Multi-Head Attention (スクラッチ実装)
class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        
        
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.o_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        B, L, _ = x.shape  # Batch, Length, D_model
        
        # 射影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ヘッド分割: (B, L, D) -> (B, L, H, HeadDim) -> (B, H, L, HeadDim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        # Attention Score: (B, H, L, HeadDim) @ (B, H, HeadDim, L) -> (B, H, L, L)
        scale = self.head_dim ** -0.5
        scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * scale

        if mask is not None:
            # mask shape: (B, 1, L, L) or similar broadcastable shape
            scores = jnp.where(mask, scores, -1e9)

        attn_weights = nnx.softmax(scores, axis=-1)

        # Output: (B, H, L, L) @ (B, H, L, HeadDim) -> (B, H, L, HeadDim)
        context = jnp.matmul(attn_weights, v)

        # ヘッド結合: (B, H, L, HeadDim) -> (B, L, H, HeadDim) -> (B, L, D)
        context = context.transpose((0, 2, 1, 3)).reshape(B, L, self.d_model)

        return self.o_proj(context)

# 2. Feed Forward Network
class FeedForward(nnx.Module):
    def __init__(self, d_model: int, expansion_factor: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(d_model, d_model * expansion_factor, rngs=rngs)
        self.linear2 = nnx.Linear(d_model * expansion_factor, d_model, rngs=rngs)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.gelu(x)
        return self.linear2(x)

# 3. Transformer Block
class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs):
        self.attention = MultiHeadAttention(d_model, num_heads, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp = FeedForward(d_model, expansion_factor=4, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        # Pre-Norm
        h = self.norm1(x)
        x = x + self.attention(h, mask)
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

# 4. Main Transformer Model
class SimpleTransformer(nnx.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_len: int, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        # 固定の位置エンコーディング（学習パラメータではないため変数として保持するか、都度計算する）
        self.pos_emb = self.create_sinusoidal_embeddings(max_len, d_model)
        
        self.layers = [
            TransformerBlock(d_model, num_heads, rngs=rngs) for _ in range(num_layers)
        ]
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def create_sinusoidal_embeddings(self, max_len, d_model):
        pos = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
        return pe

    def __call__(self, x, mask=None):
        # x: (Batch, Seq_Len)
        B, L = x.shape
        
        # Embedding + Position
        x = self.embed(x) + self.pos_emb[:L, :]
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.head(x)