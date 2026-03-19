"""
Laboratório 4 — Tarefa 4: A Prova Final (Inferência Fim-a-Fim)
===============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Descrição
---------
Instancia o Transformer completo (Encoder-Decoder) e executa um teste de
tradução fim-a-fim com a frase de brinquedo "Thinking Machines".

Pipeline completo
-----------------
    1. encoder_input  → Embedding + Positional Encoding
    2. Encoder stack (N=6) → Z  (memória contextualizada)
    3. Decoder inicia com [<START>]
    4. Loop auto-regressivo:
          a. decoder_input → Embedding + Positional Encoding
          b. Decoder stack (N=6) → probs  (B, T_tgt, vocab_size)
          c. argmax(probs[:, -1, :]) → next_token_id
          d. Concatenar next_token ao decoder_input
          e. Repetir até <EOS> ou MAX_LEN

Vocabulário fictício (toy)
--------------------------
    <PAD>=0, <START>=1, <EOS>=2, Thinking=3, Machines=4,
    são=5, máquinas=6, pensantes=7, que=8, aprendem=9
    + tokens genéricos até vocab_size=200
"""

import numpy as np
from tarefa1_blocos  import positional_encoding, init_weights, softmax
from tarefa2_encoder import encoder_block, init_encoder_weights
from tarefa3_decoder import decoder_block, init_decoder_weights


# ---------------------------------------------------------------------------
# Vocabulário fictício
# ---------------------------------------------------------------------------

VOCAB = {
    "<PAD>"    : 0,
    "<START>"  : 1,
    "<EOS>"    : 2,
    "Thinking" : 3,
    "Machines" : 4,
    "são"      : 5,
    "máquinas" : 6,
    "pensantes": 7,
    "que"      : 8,
    "aprendem" : 9,
}
VOCAB_SIZE = 200
for _i in range(10, VOCAB_SIZE):
    VOCAB[f"tok{_i}"] = _i

ID2TOKEN = {v: k for k, v in VOCAB.items()}
START_ID = VOCAB["<START>"]
EOS_ID   = VOCAB["<EOS>"]
MAX_LEN  = 15


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------

class Transformer:
    """
    Transformer Encoder-Decoder completo instanciado com pesos aleatórios.

    Atributos
    ---------
    d_model    : dimensão dos embeddings (64 na demo)
    d_ff       : dimensão interna do FFN (256 na demo)
    N          : número de camadas do Encoder e do Decoder (6)
    vocab_size : tamanho do vocabulário
    """

    def __init__(self, d_model, d_ff, N, vocab_size, seed=0):
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.N          = N
        self.vocab_size = vocab_size

        np.random.seed(seed)

        # Tabela de embeddings compartilhada (Encoder e Decoder)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

        # Pesos do Encoder: N camadas independentes
        self.enc_weights = [
            init_encoder_weights(d_model, d_ff, seed_offset=i * 10)
            for i in range(N)
        ]

        # Pesos do Decoder: N camadas independentes
        self.dec_weights = [
            init_decoder_weights(d_model, d_ff, vocab_size, seed_offset=100 + i * 20)
            for i in range(N)
        ]

    # ------------------------------------------------------------------
    def encode(self, token_ids):
        """
        Processa a sequência de entrada pelo Encoder.

        Parâmetros
        ----------
        token_ids : list[int]  — IDs dos tokens da frase de entrada

        Retorna
        -------
        Z : (1, T_src, d_model)
        """
        T   = len(token_ids)
        PE  = positional_encoding(T, self.d_model)
        emb = self.embedding[token_ids]          # (T, d_model)
        X   = (emb + PE)[np.newaxis, :, :]       # (1, T, d_model)

        Z = X
        for weights in self.enc_weights:
            Z = encoder_block(Z, weights)
        return Z

    # ------------------------------------------------------------------
    def decode_step(self, decoder_ids, Z):
        """
        Executa um forward pass completo do Decoder para a sequência atual.

        Parâmetros
        ----------
        decoder_ids : list[int]          — tokens gerados até agora
        Z           : (1, T_src, d_model)

        Retorna
        -------
        probs : (vocab_size,)  — distribuição sobre o vocabulário para o
                                 próximo token (posição -1)
        """
        T   = len(decoder_ids)
        PE  = positional_encoding(T, self.d_model)
        emb = self.embedding[decoder_ids]        # (T, d_model)
        Y   = (emb + PE)[np.newaxis, :, :]       # (1, T, d_model)

        out = Y
        for i, weights in enumerate(self.dec_weights):
            is_last = (i == self.N - 1)
            out = decoder_block(out, Z, weights, return_logits=is_last)

        # out shape: (1, T, vocab_size) — pegar distribuição do último token
        return out[0, -1, :]    # (vocab_size,)


# ---------------------------------------------------------------------------
# Loop auto-regressivo
# ---------------------------------------------------------------------------

def autoregressive_loop(model, encoder_ids, verbose=True):
    """
    Executa a inferência fim-a-fim: Encoder → Decoder loop até <EOS>.

    Parâmetros
    ----------
    model       : Transformer
    encoder_ids : list[int]  — IDs da frase de entrada
    verbose     : bool

    Retorna
    -------
    output_tokens : list[str]  — tokens gerados (incluindo <START> e <EOS>)
    """
    # 1. Encoder processa a entrada UMA única vez
    Z = model.encode(encoder_ids)

    if verbose:
        print(f"\n  Encoder processou {len(encoder_ids)} tokens → "
              f"Z shape={Z.shape}")

    # 2. Decoder começa com <START>
    decoder_ids = [START_ID]

    if verbose:
        print(f"\n--- Loop auto-regressivo ---")
        src_str = " ".join(ID2TOKEN.get(i, f"[{i}]") for i in encoder_ids)
        print(f"  Entrada (Encoder): '{src_str}'")
        print(f"  Decoder inicia com: {[ID2TOKEN[i] for i in decoder_ids]}\n")

    step = 0
    while True:
        step += 1

        # 3. Forward pass do Decoder
        probs = model.decode_step(decoder_ids, Z)   # (vocab_size,)

        # 4. Selecionar token com maior probabilidade
        next_id    = int(np.argmax(probs))
        next_token = ID2TOKEN.get(next_id, f"tok{next_id}")
        next_prob  = probs[next_id]

        # 5. Concatenar à sequência do Decoder
        decoder_ids.append(next_id)

        if verbose:
            print(f"  Passo {step:2d}: argmax → ID={next_id:3d}  "
                  f"token='{next_token}'  prob={next_prob:.4f}")

        # 6. Condição de parada
        if next_id == EOS_ID:
            if verbose:
                print(f"\n  ✓ <EOS> gerado — encerrando loop.")
            break

        if step >= MAX_LEN:
            if verbose:
                print(f"\n  ⚠ MAX_LEN={MAX_LEN} atingido — encerrando loop.")
            break

    return [ID2TOKEN.get(i, f"tok{i}") for i in decoder_ids]


# ---------------------------------------------------------------------------
# Demonstração principal
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("TAREFA 4 — Prova Final: Transformer Completo (Inferência)")
    print("=" * 65)

    # Hiperparâmetros
    d_model    = 64
    d_ff       = 256
    N          = 6
    vocab_size = VOCAB_SIZE

    print(f"\nHiperparâmetros:")
    print(f"  d_model={d_model}, d_ff={d_ff}, N={N}, vocab_size={vocab_size}")

    # Instanciar o Transformer completo
    model = Transformer(d_model, d_ff, N, vocab_size, seed=42)
    print(f"\n✓ Transformer instanciado:")
    print(f"  Embedding : ({vocab_size}, {d_model})")
    print(f"  Encoder   : {N} camadas × (Wq,Wk,Wv + FFN)")
    print(f"  Decoder   : {N} camadas × (Wq1,Wk1,Wv1 + Wq2,Wk2,Wv2 + FFN + Linear)")

    # Frase de entrada: "Thinking Machines"
    encoder_input_str = ["Thinking", "Machines"]
    encoder_ids       = [VOCAB[t] for t in encoder_input_str]
    print(f"\nFrase de entrada (Encoder): {encoder_input_str}")
    print(f"  IDs: {encoder_ids}")

    # Executar inferência
    output_tokens = autoregressive_loop(model, encoder_ids, verbose=True)

    # Resultado final
    print("\n" + "=" * 65)
    print("RESULTADO DA TRADUÇÃO (toy sequence):")
    print(f"  Entrada : {' '.join(encoder_input_str)}")
    print(f"  Saída   : {' '.join(output_tokens)}")
    print("=" * 65)

    # Estatísticas
    n_gerados = len(output_tokens) - 2   # exclui <START> e <EOS>
    print(f"\nEstatísticas:")
    print(f"  Tokens gerados (sem marcadores) : {n_gerados}")
    print(f"  Sequência completa              : {len(output_tokens)} tokens")
    print(f"  Encoder rodou                   : 1 vez (Z fixo)")
    print(f"  Decoder rodou                   : {len(output_tokens)-1} vezes (auto-regressivo)")


if __name__ == "__main__":
    demo()
