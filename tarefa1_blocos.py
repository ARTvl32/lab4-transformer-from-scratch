"""
Laboratório 4 — Tarefa 1: Refatoração e Integração (Os Blocos de Montar)
=========================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Consolida os módulos construídos nos Labs 01-03 em funções genéricas
e reutilizáveis que serão importadas pelas Tarefas 2, 3 e 4.

Blocos implementados
--------------------
1. scaled_dot_product_attention(Q, K, V, mask=None)
       Attention(Q,K,V) = softmax( QK^T/sqrt(d_k) + M ) * V

2. feed_forward_network(x, W1, b1, W2, b2)
       FFN(x) = max(0, xW1 + b1) * W2 + b2
       Expansão: d_model -> d_ff (ex: 512->2048) com ReLU no meio.

3. add_and_norm(x, sublayer_out, eps=1e-6)
       Output = LayerNorm( x + sublayer_out )
       LayerNorm normaliza sobre o último eixo (por token).

4. positional_encoding(max_len, d_model)
       PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

5. create_causal_mask(seq_len)
       Máscara triangular superior com -1e9 (≈ -inf).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Softmax estável
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Softmax numericamente estável — subtrai o máximo antes de exp."""
    x_s = x - np.max(x, axis=axis, keepdims=True)
    e   = np.exp(x_s)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calcula o Scaled Dot-Product Attention.

    Parâmetros
    ----------
    Q    : (..., T_q, d_k)
    K    : (..., T_k, d_k)
    V    : (..., T_k, d_v)
    mask : (..., T_q, T_k) ou None
        Valores 0 = permitido, -1e9 = bloqueado.

    Retorna
    -------
    output  : (..., T_q, d_v)
    weights : (..., T_q, T_k)
    """
    d_k    = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (..., T_q, T_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


# ---------------------------------------------------------------------------
# 3. Feed-Forward Network (Position-wise)
# ---------------------------------------------------------------------------

def feed_forward_network(x, W1, b1, W2, b2):
    """
    Position-wise FFN aplicado independentemente a cada token.

    Fluxo: x (d_model) -> xW1+b1 (d_ff) -> ReLU -> *W2+b2 (d_model)

    Parâmetros
    ----------
    x        : (..., T, d_model)
    W1, b1   : pesos da 1ª camada linear  (d_model, d_ff) / (d_ff,)
    W2, b2   : pesos da 2ª camada linear  (d_ff, d_model) / (d_model,)

    Retorna
    -------
    out : (..., T, d_model)
    """
    hidden = np.maximum(0, x @ W1 + b1)   # ReLU  (..., T, d_ff)
    return hidden @ W2 + b2               #        (..., T, d_model)


# ---------------------------------------------------------------------------
# 4. Add & Norm
# ---------------------------------------------------------------------------

def add_and_norm(x, sublayer_out, gamma=None, beta=None, eps=1e-6):
    """
    Conexão residual seguida de Layer Normalization.

        Output = LayerNorm( x + sublayer_out )

    LayerNorm opera sobre o último eixo (dimensão d_model de cada token).

    Parâmetros
    ----------
    x            : (..., T, d_model)  — entrada original (skip connection)
    sublayer_out : (..., T, d_model)  — saída da sub-camada (Attention ou FFN)
    gamma        : (d_model,) ou None — escala aprendível (default: 1)
    beta         : (d_model,) ou None — deslocamento aprendível (default: 0)
    eps          : float              — estabilidade numérica

    Retorna
    -------
    out : (..., T, d_model)
    """
    residual = x + sublayer_out
    mean  = residual.mean(axis=-1, keepdims=True)
    var   = residual.var(axis=-1,  keepdims=True)
    x_hat = (residual - mean) / np.sqrt(var + eps)

    if gamma is not None:
        x_hat = gamma * x_hat
    if beta is not None:
        x_hat = x_hat + beta

    return x_hat


# ---------------------------------------------------------------------------
# 5. Positional Encoding
# ---------------------------------------------------------------------------

def positional_encoding(max_len, d_model):
    """
    Codificação posicional sinusoidal (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Retorna
    -------
    PE : (max_len, d_model)
    """
    PE  = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]                           # (max_len, 1)
    div = 10000 ** (2 * np.arange(d_model // 2)[None, :] / d_model)  # (1, d_model/2)

    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div)
    return PE


# ---------------------------------------------------------------------------
# 6. Máscara Causal
# ---------------------------------------------------------------------------

def create_causal_mask(seq_len):
    """
    Máscara triangular superior com -1e9 (≈ -inf).
    Triangular inferior + diagonal = 0 (permitido).
    Triangular superior             = -1e9 (bloqueado).

    Retorna
    -------
    mask : (seq_len, seq_len)
    """
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


# ---------------------------------------------------------------------------
# Inicializador de pesos
# ---------------------------------------------------------------------------

def init_weights(shape, seed=None, scale=0.01):
    """Inicializa uma matriz de pesos aleatória."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) * scale


# ---------------------------------------------------------------------------
# Demonstração / testes dos blocos
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 1 — Blocos Base (Refatoração e Integração)")
    print("=" * 60)

    np.random.seed(42)
    B, T, d_model, d_ff = 1, 5, 32, 128

    X = np.random.randn(B, T, d_model)

    # --- Attention ---
    Wq = init_weights((d_model, d_model), seed=0)
    Wk = init_weights((d_model, d_model), seed=1)
    Wv = init_weights((d_model, d_model), seed=2)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    attn_out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\n[Attention]  input={X.shape}  output={attn_out.shape}")
    print(f"             pesos somam 1 por linha: "
          f"{np.allclose(weights.sum(axis=-1), 1.0)}")

    # --- Attention com máscara causal ---
    mask = create_causal_mask(T)
    attn_masked, w_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    futuro_zerado = np.all(w_masked[0][np.triu(np.ones((T,T), dtype=bool), k=1)] == 0)
    print(f"[Mask]       futuro estritamente 0.0: {futuro_zerado}")

    # --- FFN ---
    W1 = init_weights((d_model, d_ff), seed=3)
    b1 = np.zeros(d_ff)
    W2 = init_weights((d_ff, d_model), seed=4)
    b2 = np.zeros(d_model)
    ffn_out = feed_forward_network(X, W1, b1, W2, b2)
    print(f"\n[FFN]        input={X.shape}  output={ffn_out.shape}")
    print(f"             expansão {d_model}→{d_ff}→{d_model}  ✓")

    # --- Add & Norm ---
    norm_out = add_and_norm(X, attn_out)
    mean_after = norm_out.mean(axis=-1)
    print(f"\n[Add & Norm] output={norm_out.shape}")
    print(f"             média ≈ 0: {np.allclose(mean_after, 0, atol=1e-5)}")

    # --- Positional Encoding ---
    PE = positional_encoding(T, d_model)
    print(f"\n[Pos Enc]    shape={PE.shape}  "
          f"valores únicos por posição: {not np.allclose(PE[0], PE[1])}")

    print("\n✓ Todos os blocos base instanciados e verificados.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
