"""
Laboratório 4 — Tarefa 3: Montando a Pilha do Decoder
======================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Implementa o DecoderBlock e a pilha completa do Decoder (N=6 camadas).

Fluxo exato de um bloco
------------------------
    Y  (B, T_tgt, d_model)   ← tokens já gerados
    Z  (B, T_src, d_model)   ← memória do Encoder

    Sub-camada 1 — Masked Self-Attention
        Q, K, V ← Y
        máscara causal com -inf  (impede ver o futuro)
        → Add & Norm  →  Y1

    Sub-camada 2 — Cross-Attention (ponte Encoder-Decoder)
        Q  ← Y1   (o que o Decoder está gerando)
        K  ← Z    (índice da frase original)
        V  ← Z    (conteúdo semântico da frase original)
        sem máscara  (Decoder vê toda a entrada do Encoder)
        → Add & Norm  →  Y2

    Sub-camada 3 — FFN
        → Add & Norm  →  Y_out  (B, T_tgt, d_model)

    Cabeça de saída (apenas na última camada da pilha)
        Linear: d_model → vocab_size
        Softmax → distribuição de probabilidades
"""

import numpy as np
from tarefa1_blocos import (
    scaled_dot_product_attention,
    feed_forward_network,
    add_and_norm,
    create_causal_mask,
    softmax,
    init_weights,
)


# ---------------------------------------------------------------------------
# Inicialização dos pesos de um bloco do Decoder
# ---------------------------------------------------------------------------

def init_decoder_weights(d_model, d_ff, vocab_size, seed_offset=0):
    """
    Inicializa todos os pesos de um DecoderBlock.

    Retorna dict com:
        Wq1, Wk1, Wv1  — Masked Self-Attention
        Wq2, Wk2, Wv2  — Cross-Attention
        W1, b1, W2, b2 — FFN
        W_out, b_out   — projeção final d_model → vocab_size
    """
    s = seed_offset
    return {
        # Sub-camada 1: Masked Self-Attention
        "Wq1": init_weights((d_model, d_model), seed=s+0),
        "Wk1": init_weights((d_model, d_model), seed=s+1),
        "Wv1": init_weights((d_model, d_model), seed=s+2),
        # Sub-camada 2: Cross-Attention
        "Wq2": init_weights((d_model, d_model), seed=s+3),
        "Wk2": init_weights((d_model, d_model), seed=s+4),
        "Wv2": init_weights((d_model, d_model), seed=s+5),
        # Sub-camada 3: FFN
        "W1":  init_weights((d_model, d_ff),    seed=s+6),
        "b1":  np.zeros(d_ff),
        "W2":  init_weights((d_ff, d_model),    seed=s+7),
        "b2":  np.zeros(d_model),
        # Cabeça de saída
        "W_out": init_weights((d_model, vocab_size), seed=s+8),
        "b_out": np.zeros(vocab_size),
    }


# ---------------------------------------------------------------------------
# DecoderBlock
# ---------------------------------------------------------------------------

def decoder_block(y, Z, weights, return_logits=False):
    """
    Executa um bloco do Decoder.

    Parâmetros
    ----------
    y            : (B, T_tgt, d_model)  — tokens já gerados pelo Decoder
    Z            : (B, T_src, d_model)  — memória do Encoder
    weights      : dict                 — pesos do bloco
    return_logits: bool                 — se True, aplica Linear+Softmax no final

    Retorna
    -------
    y_out : (B, T_tgt, d_model)          se return_logits=False
    probs : (B, T_tgt, vocab_size)       se return_logits=True
    """
    T_tgt = y.shape[1]

    # ------------------------------------------------------------------
    # Sub-camada 1: Masked Self-Attention
    # Q, K, V derivam de y; máscara causal impede acesso ao futuro
    # ------------------------------------------------------------------
    Q1 = y @ weights["Wq1"]
    K1 = y @ weights["Wk1"]
    V1 = y @ weights["Wv1"]

    mask = create_causal_mask(T_tgt)          # (T_tgt, T_tgt)
    attn1_out, _ = scaled_dot_product_attention(Q1, K1, V1, mask=mask)

    y1 = add_and_norm(y, attn1_out)

    # ------------------------------------------------------------------
    # Sub-camada 2: Cross-Attention (ponte Encoder-Decoder)
    # Q vem de y1 (Decoder), K e V vêm de Z (Encoder)
    # Sem máscara: Decoder consulta toda a frase do Encoder
    # ------------------------------------------------------------------
    Q2 = y1 @ weights["Wq2"]
    K2 = Z  @ weights["Wk2"]
    V2 = Z  @ weights["Wv2"]

    attn2_out, _ = scaled_dot_product_attention(Q2, K2, V2, mask=None)

    y2 = add_and_norm(y1, attn2_out)

    # ------------------------------------------------------------------
    # Sub-camada 3: FFN
    # ------------------------------------------------------------------
    ffn_out = feed_forward_network(
        y2, weights["W1"], weights["b1"],
            weights["W2"], weights["b2"]
    )

    y_out = add_and_norm(y2, ffn_out)

    # ------------------------------------------------------------------
    # Cabeça de saída (opcional)
    # ------------------------------------------------------------------
    if return_logits:
        logits = y_out @ weights["W_out"] + weights["b_out"]   # (..., V)
        probs  = softmax(logits, axis=-1)
        return probs

    return y_out


# ---------------------------------------------------------------------------
# Pilha do Decoder (N camadas)
# ---------------------------------------------------------------------------

def decoder_stack(y, Z, all_weights):
    """
    Empilha N DecoderBlocks. A projeção final (Linear+Softmax) é aplicada
    somente na última camada.

    Parâmetros
    ----------
    y           : (B, T_tgt, d_model)
    Z           : (B, T_src, d_model)
    all_weights : list[dict]  — um dict por camada

    Retorna
    -------
    probs : (B, T_tgt, vocab_size)
    """
    out = y
    for i, weights in enumerate(all_weights):
        is_last = (i == len(all_weights) - 1)
        out = decoder_block(out, Z, weights, return_logits=is_last)
    return out   # probs na última camada


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("TAREFA 3 — Pilha do Decoder")
    print("=" * 65)

    np.random.seed(42)

    B, T_src, T_tgt = 1, 2, 3   # Encoder: 2 tokens; Decoder: 3 tokens gerados
    d_model  = 64
    d_ff     = 256
    N        = 6
    vocab_size = 50

    # Memória do Encoder (Z) — normalmente vinda da Tarefa 2
    Z = np.random.randn(B, T_src, d_model)

    # Tokens já gerados pelo Decoder: [<START>, token_a, token_b]
    Y = np.random.randn(B, T_tgt, d_model)

    print(f"\nEntradas:")
    print(f"  Z (Encoder memory) : {Z.shape}")
    print(f"  Y (Decoder input)  : {Y.shape}")
    print(f"  vocab_size={vocab_size}, N={N} camadas")

    # Inicializar pesos independentes para cada camada
    all_weights = [
        init_decoder_weights(d_model, d_ff, vocab_size, seed_offset=i * 20)
        for i in range(N)
    ]

    # Processar camadas intermediárias (sem logits) + última (com logits)
    print(f"\nFluxo pelas {N} camadas do Decoder:")
    out = Y
    for i, weights in enumerate(all_weights):
        is_last = (i == N - 1)
        out = decoder_block(out, Z, weights, return_logits=is_last)
        if not is_last:
            print(f"  Camada {i+1}: shape={out.shape}  "
                  f"norma={np.linalg.norm(out):.4f}")
        else:
            print(f"  Camada {i+1}: probs shape={out.shape}  "
                  f"(Linear + Softmax aplicados)")

    probs = out
    print(f"\nDistribuição de probabilidades: {probs.shape}")

    # Verificações
    somas = probs.sum(axis=-1)
    print(f"  Soma das probs por token (deve ser 1.0):")
    for t in range(T_tgt):
        print(f"    token[{t}]: {somas[0, t]:.8f}  "
              f"{'✓' if abs(somas[0,t]-1.0)<1e-6 else '✗'}")

    # Argmax no último token (o próximo a ser gerado)
    next_token_id = int(np.argmax(probs[0, -1]))
    print(f"\n  argmax do último token → ID={next_token_id}  "
          f"(prob={probs[0,-1,next_token_id]:.4f})")

    print(f"\n✓ Decoder stack completo com Cross-Attention e saída Softmax.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
