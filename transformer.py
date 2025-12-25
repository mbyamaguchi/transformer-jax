import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

# 乱数生成器
key = jax.random.PRNGKey(0)

class MultiHeadAttention(eqx.Module):
    num_heads: int
    head_dim: int
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(self, d_model: int, num_heads: int, key: jax.random.PRNGKey):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # パラメータの初期化用のキーを分割
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # 線形層の定義
        self.query_proj = eqx.nn.Linear(d_model, d_model, key=k1)
        self.key_proj = eqx.nn.Linear(d_model, d_model, key=k2)
        self.value_proj = eqx.nn.Linear(d_model, d_model, key=k3)
        self.output_proj = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        # x shape: (seq_len, d_model)
        seq_len, _ = x.shape

        # Q, K, V の射影
        # shape: (seq_len, num_heads, head_dim) に変形
        Q = self.query_proj(x).reshape(seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(x).reshape(seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(x).reshape(seq_len, self.num_heads, self.head_dim)

        # 軸の入れ替え: (num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (1, 0, 2))
        K = jnp.transpose(K, (1, 0, 2))
        V = jnp.transpose(V, (1, 0, 2))

        # Scaled Dot-Product Attention
        # attention_scores shape: (num_heads, seq_len, seq_len)
        attention_scores = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1))) / jnp.sqrt(self.head_dim)

        if mask is not None:
            # マスク位置に -inf を加算
            attention_scores = attention_scores + (mask * -1e9)

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # V との積算
        # output shape: (num_heads, seq_len, head_dim)
        attention_output = jnp.matmul(attention_weights, V)

        # 元の形に戻す (seq_len, num_heads, head_dim) -> (seq_len, d_model)
        attention_output = jnp.transpose(attention_output, (1, 0, 2))
        attention_output = attention_output.reshape(seq_len, -1)

        return self.output_proj(attention_output)
    

class FeedForward(eqx.Module):
    net: eqx.nn.Sequential

    def __init__(self, d_model: int, expansion_factor: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key, 2)
        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(d_model, d_model * expansion_factor, key=k1),
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Linear(d_model * expansion_factor, d_model, key=k2)
        ])

    def __call__(self, x):
        return self.net(x)
    

class TransformerBlock(eqx.Module):
    attention: MultiHeadAttention
    norm1: eqx.nn.LayerNorm
    mlp: FeedForward
    norm2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, num_heads: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key, 2)
        self.attention = MultiHeadAttention(d_model, num_heads, key=k1)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor=4, key=k2)
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        # Pre-Norm Architecture (現代的な標準)
        # x + Attention(Norm(x))
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask=mask)
        x = x + attn_out # 残差接続

        # x + MLP(Norm(x))
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out # 残差接続
        
        return x

class SimpleTransformer(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: list
    output_proj: eqx.nn.Linear
    positional_encodings: jnp.ndarray

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_len: int, key: jax.random.PRNGKey):
        k_emb, k_layers, k_out = jax.random.split(key, 3)
        
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=k_emb)
        
        # レイヤーのスタックを作成
        k_layers = jax.random.split(k_layers, num_layers)
        self.layers = [TransformerBlock(d_model, num_heads, key=k) for k in k_layers]
        
        self.output_proj = eqx.nn.Linear(d_model, vocab_size, key=k_out)
        
        # 固定の正弦波位置エンコーディングを作成
        pos = jnp.arange(max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
        self.positional_encodings = pe

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        # x shape: (seq_len,) -> token IDs
        seq_len = x.shape[0]
        
        # 埋め込み + 位置エンコーディング
        x = self.embedding(x) + self.positional_encodings[:seq_len, :]

        # Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask=mask)

        # 出力層
        return self.output_proj(x)
    
