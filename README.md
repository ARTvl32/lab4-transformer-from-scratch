# Laboratório 4 — O Transformer Completo "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial 2026.1  
**Instituição:** iCEV — Instituto de Ensino Superior  
**Professor:** Dimmy Magalhães  

> **Nota:** Partes geradas/complementadas com IA, revisadas por Arthur.

---

## Contexto

Nos laboratórios anteriores foram implementados os blocos individuais do
Transformer: Scaled Dot-Product Attention (Lab 01), Causal Masking e
Cross-Attention (Lab 03), e o fluxo do Encoder com Add & Norm e FFN (Lab 02).

Neste laboratório todos esses módulos são integrados em uma **arquitetura
Encoder-Decoder completa**, que executa um teste de tradução fim-a-fim
com uma *toy sequence*.

---

## Objetivos de Aprendizagem

1. Aplicar engenharia de software para integrar módulos de redes neurais separados em uma única topologia coerente
2. Garantir o fluxo correto de tensores passando pelas camadas **Add & Norm** e **Feed-Forward**
3. Acoplar o **Loop Auto-regressivo** de inferência na saída do Decoder

---

## Estrutura do Repositório

```
lab4-transformer/
│
├── tarefa1_blocos.py          # Blocos reutilizáveis: Attention, FFN, Add&Norm
├── tarefa2_encoder.py         # EncoderBlock empilhável → matriz Z
├── tarefa3_decoder.py         # DecoderBlock com Masked Attn + Cross-Attn
├── tarefa4_inferencia.py      # Transformer completo + loop auto-regressivo
└── README.md
```

---

## Tarefas

### Tarefa 1 — Refatoração e Integração (Os Blocos de Montar)

Reescreve e consolida os módulos dos laboratórios anteriores em funções
genéricas reutilizáveis:

| Função | Descrição |
|--------|-----------|
| `scaled_dot_product_attention(Q, K, V, mask)` | Atenção escalonada com máscara opcional |
| `feed_forward_network(x, W1, b1, W2, b2)` | FFN com expansão 512 → 2048 → 512 e ReLU |
| `add_and_norm(x, sublayer_out)` | Conexão residual + LayerNorm: `LayerNorm(x + sublayer(x))` |

---

### Tarefa 2 — Pilha do Encoder

`EncoderBlock(x, weights)` implementa o fluxo exato:

```
X (entrada + Positional Encoding)
  → Self-Attention(Q=X, K=X, V=X)
  → Add & Norm
  → FFN
  → Add & Norm
  → Z (matriz de memória contextualizada)
```

O bloco é empilhável: a saída Z de uma camada alimenta a entrada da próxima.

---

### Tarefa 3 — Pilha do Decoder

`DecoderBlock(y, Z, weights)` implementa o fluxo exato:

```
Y (tokens já gerados)
  → Masked Self-Attention (máscara causal com −∞)
  → Add & Norm
  → Cross-Attention (Q ← saída anterior, K/V ← Z do Encoder)
  → Add & Norm
  → FFN
  → Add & Norm
  → Linear (d_model → vocab_size)
  → Softmax → distribuição de probabilidades
```

---

### Tarefa 4 — Prova Final (Inferência)

Instancia o Transformer completo com tensores fictícios simulando a frase
**"Thinking Machines"** como entrada do Encoder.

Loop auto-regressivo:
1. Decoder inicia com `<START>`
2. A cada iteração: `argmax(softmax(logits))` seleciona o próximo token
3. O token é concatenado à entrada do Decoder
4. Loop encerra ao gerar `<EOS>`

---

## Como Executar

> Requer apenas **Python 3** e **NumPy**.

```bash
# Tarefa 1 — Blocos base
python tarefa1_blocos.py

# Tarefa 2 — Encoder
python tarefa2_encoder.py

# Tarefa 3 — Decoder
python tarefa3_decoder.py

# Tarefa 4 — Inferência fim-a-fim
python tarefa4_inferencia.py
```

---

## Arquitetura: Fluxo Completo de Tensores

```
encoder_input (B, T_src, d_model)
        │
   ┌────▼─────────────────────┐
   │      ENCODER (N=6)       │
   │  Self-Attn → Add&Norm    │
   │  FFN       → Add&Norm    │
   └────────────┬─────────────┘
                │  Z (B, T_src, d_model)
                │
decoder_input (B, T_tgt, d_model)
        │       │
   ┌────▼───────▼─────────────┐
   │      DECODER (N=6)       │
   │  Masked Self-Attn → A&N  │
   │  Cross-Attn(Z)    → A&N  │
   │  FFN              → A&N  │
   └────────────┬─────────────┘
                │
         Linear + Softmax
                │
         probs (B, T_tgt, V)
```

---

## Fundamentos Matemáticos

**Scaled Dot-Product Attention (com máscara):**

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**Add & Norm:**

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**FFN:**

$$\text{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2$$

---

## Referências

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
- Laboratórios 01, 02 e 03 — implementações base reutilizadas aqui

- *IAs generativas (claude e GPT) foram consultadas para o desenvolvimento deste laboratório*
