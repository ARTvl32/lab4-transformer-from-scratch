"""
Laboratório 4 — Tarefa 2: Montando a Pilha do Encoder
======================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Implementa o EncoderBlock e a pilha completa do Encoder (N=6 camadas).

Fluxo exato de um bloco
------------------------
    X  (B, T, d_model)  ← entrada já somada com Positional Encoding
     │
     ├─ Self-Attention(Q=X, K=X, V=X)   ← sem máscara (bidirecional)
     │
     ├─ Add & Norm  →  X1
     │
     ├─ FFN(X1)
     │
     └─ Add & Norm  →  Z  (B, T, d_model)

O bloco é empilhável: a saída Z de uma camada alimenta X da próxima.
Após N=6 camadas, Z contém a matriz de memória contextualizada.
"""

import numpy as np
from tarefa1_blocos import (
    scaled_dot_product_attention,
    feed_forward_network,
    add_and_norm,
    positional_encoding,
    init_weights,
)


# ---------------------------------------------------------------------------
# Inicialização dos pesos de um bloco do Encoder
# ---------------------------------------------------------------------------

def init_encoder_weights(d_model, d_ff, seed_offset=0):
    """
    Inicializa todos os pesos necessários para um EncoderBlock.

    Retorna um dicionário com:
        Wq, Wk, Wv  — projeções de atenção  (d_model, d_model)
        W1, b1      — 1ª camada do FFN       (d_model, d_ff) / (d_ff,)
        W2, b2      — 2ª camada do FFN       (d_ff, d_model) / (d_model,)
    """
    s = seed_offset
    return {
        "Wq": init_weights((d_model, d_model), seed=s+0),
        "Wk": init_weights((d_model, d_model), seed=s+1),
        "Wv": init_weights((d_model, d_model), seed=s+2),
        "W1": init_weights((d_model, d_ff),    seed=s+3),
        "b1": np.zeros(d_ff),
        "W2": init_weights((d_ff, d_model),    seed=s+4),
        "b2": np.zeros(d_model),
    }


# ---------------------------------------------------------------------------
# EncoderBlock
# ---------------------------------------------------------------------------

def encoder_block(x, weights):
    """
    Executa um bloco do Encoder sobre o tensor de entrada x.

    Fluxo:
        x → Self-Attention → Add&Norm → FFN → Add&Norm → Z

    Parâmetros
    ----------
    x       : (B, T, d_model)  — entrada (embedding + positional encoding)
    weights : dict             — pesos do bloco (ver init_encoder_weights)

    Retorna
    -------
    Z : (B, T, d_model)  — representações contextualizadas
    """
    # --- Sub-camada 1: Self-Attention (Q, K, V todos derivados de x) ---
    Q = x @ weights["Wq"]
    K = x @ weights["Wk"]
    V = x @ weights["Wv"]

    # Sem máscara: o Encoder é bidirecional — cada token vê todos os outros
    attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=None)

    # Add & Norm após atenção
    x1 = add_and_norm(x, attn_out)

    # --- Sub-camada 2: FFN ---
    ffn_out = feed_forward_network(
        x1, weights["W1"], weights["b1"],
            weights["W2"], weights["b2"]
    )

    # Add & Norm após FFN
    Z = add_and_norm(x1, ffn_out)

    return Z


# ---------------------------------------------------------------------------
# Pilha do Encoder (N camadas)
# ---------------------------------------------------------------------------

def encoder_stack(x, all_weights):
    """
    Empilha N EncoderBlocks, passando a saída Z de cada camada para a próxima.

    Parâmetros
    ----------
    x           : (B, T, d_model)  — entrada com Positional Encoding somado
    all_weights : list[dict]       — lista de dicionários de pesos, um por camada

    Retorna
    -------
    Z : (B, T, d_model)  — matriz de memória final do Encoder
    """
    Z = x
    for i, weights in enumerate(all_weights):
        Z = encoder_block(Z, weights)
    return Z


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("TAREFA 2 — Pilha do Encoder")
    print("=" * 60)

    np.random.seed(42)

    # Hiperparâmetros (reduzidos para demo; originais: d_model=512, d_ff=2048)
    B       = 1
    T       = 2          # "Thinking Machines" → 2 tokens
    d_model = 64
    d_ff    = 256
    N       = 6          # número de camadas do Encoder

    # Simular embeddings para "Thinking" e "Machines"
    vocab_size = 100
    embedding_table = np.random.randn(vocab_size, d_model) * 0.1

    # IDs fictícios para os dois tokens
    token_ids = [10, 20]   # "Thinking"=10, "Machines"=20
    X_embed   = embedding_table[token_ids]          # (T, d_model)

    # Somar Positional Encoding
    PE = positional_encoding(T, d_model)
    X  = (X_embed + PE)[np.newaxis, :, :]           # (1, T, d_model)

    print(f"\nEntrada  X (embed + PE): {X.shape}")

    # Inicializar pesos independentes para cada uma das N camadas
    all_weights = [
        init_encoder_weights(d_model, d_ff, seed_offset=i * 10)
        for i in range(N)
    ]

    # Processar camada por camada, exibindo norma da saída
    print(f"\nFluxo pelas {N} camadas do Encoder:")
    Z = X
    for i, weights in enumerate(all_weights):
        Z_prev_norm = np.linalg.norm(Z)
        Z           = encoder_block(Z, weights)
        print(f"  Camada {i+1}: shape={Z.shape}  "
              f"norma={np.linalg.norm(Z):.4f}  "
              f"média≈0: {np.allclose(Z.mean(axis=-1), 0, atol=0.3)}")

    print(f"\nMatriz Z final (memória do Encoder): {Z.shape}")
    print(f"  Z['Thinking'] (primeiros 6 valores): "
          f"{Z[0, 0, :6].round(4)}")
    print(f"  Z['Machines'] (primeiros 6 valores): "
          f"{Z[0, 1, :6].round(4)}")
    print(f"\n✓ Encoder stack completo: (B,T,d_model) preservado em todas as camadas.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
