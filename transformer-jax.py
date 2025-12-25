import jax
import jax.numpy as jnp
import numpy as np # 初期化やデータ作成用

# --- 1. ヘルパー関数: 線形層の初期化と計算 ---

def init_linear_params(key, in_dim, out_dim):
    """重みとバイアスを初期化して辞書で返す"""
    k_w, k_b = jax.random.split(key)
    # Xavier/Glorot Initialization
    limit = jnp.sqrt(6 / (in_dim + out_dim))
    w = jax.random.uniform(k_w, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return {'w': w, 'b': b}

def linear_forward(params, x):
    """y = xW + b"""
    return jnp.dot(x, params['w']) + params['b']

# --- 2. Attention Mechanism ---

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    # スコア計算: (seq_len, num_heads, head_dim)
    # matmulのために軸を入れ替えたりする実装が多いですが、
    # einsumを使うと直感的に書けます。
    # q: [seq_len_q, heads, dim], k: [seq_len_k, heads, dim] -> scores: [heads, seq_len_q, seq_len_k]
    scores = jnp.einsum('qhd,khd->hqk', q, k) / jnp.sqrt(d_k)
    
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # valuesとの積: weights: [heads, seq_q, seq_k], v: [seq_k, heads, dim] -> [seq_q, heads, dim]
    output = jnp.einsum('hqk,khd->qhd', attn_weights, v)
    return output

def multi_head_attention_forward(params, x, mask=None):
    # paramsは {'W_q': ..., 'W_k': ..., 'W_v': ..., 'W_o': ...} を想定
    seq_len, d_model = x.shape
    num_heads = params['num_heads']
    head_dim = d_model // num_heads

    # Q, K, V の射影
    q = linear_forward(params['W_q'], x)
    k = linear_forward(params['W_k'], x)
    v = linear_forward(params['W_v'], x)

    # ヘッド分割: (seq_len, d_model) -> (seq_len, num_heads, head_dim)
    q = q.reshape(seq_len, num_heads, head_dim)
    k = k.reshape(seq_len, num_heads, head_dim)
    v = v.reshape(seq_len, num_heads, head_dim)

    # Attention計算
    attn_out = scaled_dot_product_attention(q, k, v, mask)

    # 結合して出力射影
    attn_out = attn_out.reshape(seq_len, d_model)
    return linear_forward(params['W_o'], attn_out)

# --- 3. Transformer Block & 全体モデル ---

def init_transformer_params(key, d_model, num_heads, d_ff, vocab_size, max_len, num_layers):
    """モデル全体のパラメータ辞書を作成"""
    params = {}
    keys = jax.random.split(key, num_layers + 3)
    
    # Embedding
    params['embedding'] = jax.random.normal(keys[0], (vocab_size, d_model)) * 0.02
    
    # Positional Encoding (学習可能なパラメータとして実装する例)
    params['pos_embedding'] = jax.random.normal(keys[1], (max_len, d_model)) * 0.02

    # Layers
    params['layers'] = []
    for i in range(num_layers):
        layer_key = keys[2+i]
        ks = jax.random.split(layer_key, 6)
        layer_params = {
            # Attention Params
            'attn': {
                'num_heads': num_heads,
                'W_q': init_linear_params(ks[0], d_model, d_model),
                'W_k': init_linear_params(ks[1], d_model, d_model),
                'W_v': init_linear_params(ks[2], d_model, d_model),
                'W_o': init_linear_params(ks[3], d_model, d_model),
            },
            # Feed Forward Params
            'ff': {
                'W1': init_linear_params(ks[4], d_model, d_ff),
                'W2': init_linear_params(ks[5], d_ff, d_model),
            },
            # Layer Norm (簡易版: scaleとbias)
            'ln1': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
            'ln2': {'scale': jnp.ones(d_model), 'bias': jnp.zeros(d_model)},
        }
        params['layers'].append(layer_params)
    
    # Output Head
    params['head'] = init_linear_params(keys[-1], d_model, vocab_size)
    return params

def layer_norm(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * params['scale'] + params['bias']

def transformer_forward(params, x, mask=None):
    """
    x: トークンIDの配列 (seq_len,)
    """
    seq_len = x.shape[0]
    
    # 1. Embedding + Positional Encoding
    h = params['embedding'][x] + params['pos_embedding'][:seq_len]

    # 2. Layers
    for layer_params in params['layers']:
        # --- Self-Attention Block ---
        normed_h = layer_norm(layer_params['ln1'], h)
        attn_out = multi_head_attention_forward(layer_params['attn'], normed_h, mask)
        h = h + attn_out # Residual connection
        
        # --- Feed Forward Block ---
        normed_h = layer_norm(layer_params['ln2'], h)
        # FF: Linear -> GELU -> Linear
        ff_out = linear_forward(layer_params['ff']['W1'], normed_h)
        ff_out = jax.nn.gelu(ff_out)
        ff_out = linear_forward(layer_params['ff']['W2'], ff_out)
        h = h + ff_out # Residual connection

    # 3. Output Head
    logits = linear_forward(params['head'], h)
    return logits

# --- 4. 実行テスト ---

# ハイパーパラメータ
config = {
    'd_model': 64,
    'num_heads': 4,
    'd_ff': 256,
    'vocab_size': 1000,
    'max_len': 50,
    'num_layers': 2
}

# 1. パラメータ初期化
key = jax.random.PRNGKey(42)
params = init_transformer_params(key, **config)

# 2. バッチデータの作成
batch_size = 5
seq_len = 10
# (5, 10) の入力データ
input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config['vocab_size'])

# 3. vmapによるバッチ化
# transformer_forwardは (seq_len,) を受け取る関数なので、
# vmapを使って (batch, seq_len) を受け取れるように変換する。
# in_axes=(None, 0, None): paramsは共有(None), xは0次元目を分割(0), maskは共有(None)
batched_forward = jax.vmap(transformer_forward, in_axes=(None, 0, None))

# 4. マスク作成 (例: 未来の単語を見ないCausal Mask)
mask = jnp.tril(jnp.ones((seq_len, seq_len))) # 下三角行列
mask = (mask == 0) # Trueの位置がマスクされる

# 5. 推論実行 (JITコンパイル付き)
jit_forward = jax.jit(batched_forward)
logits = jit_forward(params, input_ids, mask)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}") # (5, 10, 1000)


import jax
import jax.numpy as jnp

# --- 1. 損失関数の定義 (Cross Entropy Loss) ---

def compute_loss(params, x, targets, mask):
    """
    x: 入力 (Batch, Seq_Len)
    targets: 正解ラベル (Batch, Seq_Len)
    mask: パディング等を無視するためのマスク (Batch, Seq_Len)
    """
    # 1. 前向き計算 (前回の batched_forward を使用)
    #    推論用の関数をそのまま再利用します。
    #    logits shape: (Batch, Seq_Len, Vocab_Size)
    logits = batched_forward(params, x, mask)
    
    # 2. ロジットを確率（対数確率）に変換
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # 3. 正解ラベルに対応する確率を取り出す
    #    targets shape: (Batch, Seq_Len)
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    
    # sum(target * log_prob) -> 正解クラスのlog_probだけが残る
    # 軸: (Batch, Seq, Vocab) * (Batch, Seq, Vocab) -> sum over Vocab
    token_losses = -jnp.sum(targets_one_hot * log_probs, axis=-1)
    
    # 4. マスク処理（パディング部分のLossを0にするなど）
    loss = jnp.mean(token_losses)
    
    return loss

# --- 2. 1ステップの更新関数 (Update Step) ---

@jax.jit
def update_step(params, x, targets, mask, learning_rate):
    # jax.value_and_grad: 関数を実行し、その戻り値(loss)と勾配(grads)をペアで返す
    loss, grads = jax.value_and_grad(compute_loss)(params, x, targets, mask)
    
    # パラメータの更新: params = params - lr * grads
    # paramsは辞書のネスト構造なので、tree_mapを使って全要素に同じ計算を適用する
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, 
        params, 
        grads
    )
    
    return new_params, loss

# --- 3. 学習ループの実行 ---

# ハイパーパラメータ
learning_rate = 0.01
num_epochs = 100

# ダミーデータ作成: "Next Token Prediction" タスク
# 入力: [A, B, C, D] -> 正解: [B, C, D, E]
# input_ids は作成済み: shape (5, 10)
# targets は input_ids を1つずらして作成
inputs = input_ids[:, :-1]     # 最後の1文字を除く (Batch, 9)
targets = input_ids[:, 1:]     # 最初の1文字を除く (Batch, 9)

# Attentionマスクもサイズを合わせる (9x9)
seq_len = inputs.shape[1]
train_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
train_mask = (train_mask == 0)

print(f"Start Training... (Loss will be printed)")

loss_history = []

for epoch in range(num_epochs):
    # 更新実行
    params, loss = update_step(params, inputs, targets, train_mask, learning_rate)
    
    loss_history.append(loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

print(f"Final Loss: {loss:.4f}")



# Optax による
# Optaxを使う場合のイメージ（参考）
import optax

optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(params) # オプティマイザの状態（モーメンタム等）を初期化

@jax.jit
def update_step_optax(params, opt_state, x, y, mask):
    loss, grads = jax.value_and_grad(compute_loss)(params, x, y, mask)
    
    # 勾配から更新量を計算 (Adamのロジックなどがここで走る)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    # パラメータに更新量を適用 (これも内部でtree_mapを使っている)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss