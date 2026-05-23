# LLM Exam Notes — Quick Revision

---

## 1. Transformer Architecture

- Has two parts: **Encoder** (understands input) + **Decoder** (generates output)
- **Positional Encoding** → tells the model the order of words
- **Multi-Head Attention** → looks at multiple parts of the sentence at once
- **Add & Norm** → stabilizes training (reduces covariance shift)
- **Feed Forward** → adds non-linearity
- **Masked Attention** (decoder only) → can't peek at future words
- Final **Softmax** → gives probability of next word

---

## 2. Layer Normalization

**Why not Batch Norm?** Sequences have varying lengths → small/unstable batches.

**Layer Norm** normalizes across the **features of one sample** (not across the batch).

$$\bar{a} = \frac{a - \mu}{\sigma} \cdot \gamma + \beta$$

> Think of it as: "normalize each token's values independently."

---

## 3. Three Architecture Types

| Type | Example | Good For | Can't Do |
|---|---|---|---|
| Decoder only | GPT | Text generation | See future context |
| Encoder only | BERT | Understanding tasks | Generate text |
| Encoder-Decoder | BART, T5 | Translation, summarization | — |

---

## 4. Auto-regressive Language Models

A language model predicts **P(X)** = probability of a sentence.

$$P(X) = \prod P(x_i \mid x_1, \dots, x_{i-1})$$

- Predict the **next token** given all previous tokens
- This is just **classification** over the vocabulary at each step
- Pipeline: embeddings → neural network → linear layer → softmax → next token

---

## 5. BERT

**Full name:** Bidirectional Encoder Representations from Transformers

**Key idea:** Use **both left and right context** to understand a word.

### Input
Token Embedding + Segment Embedding + Position Embedding

### Pretraining Task 1 — Masked LM (MLM)
- Mask **15%** of tokens, predict them
- Of those 15%:
  - **80%** → replace with `[MASK]`
  - **10%** → replace with a random word
  - **10%** → keep unchanged

### Pretraining Task 2 — Next Sentence Prediction (NSP)
- Given two sentences, predict: is B the actual next sentence?
- 50% real pairs (`IsNext`), 50% random pairs (`NotNext`)
- Helps with QA and NLI tasks

---

## 6. BART

**Full name:** Bidirectional and Auto-Regressive Transformer

**Structure:** Bidirectional Encoder + Autoregressive Decoder

**Idea:** Corrupt text → train model to reconstruct the original

### Corruption Strategies
| Method | What happens |
|---|---|
| Token Masking | Replace tokens with `[MASK]` |
| Token Deletion | Delete tokens entirely |
| Text Infilling | Mask whole spans (multiple tokens) |
| Sentence Permutation | Shuffle sentence order |
| Document Rotation | Start document from a random token |

### BART for Summarization
- Fine-tune on news article → summary
- Works well even with few examples

---

## 7. T5

**Full name:** Text-to-Text Transfer Transformer

**Key idea:** Every NLP task = text in, text out. Prefix tells the model what to do.

| Task | Input | Output |
|---|---|---|
| Translation | `translate English to German: Hello` | `Hallo` |
| Summarization | `summarize: long article...` | `Short summary` |
| Classification | `cola sentence: He runned fast.` | `not acceptable` |

**Pretraining:** Same denoising idea as BART — mask spans, predict the missing parts.

---

## 8. BERT vs. BART (Common Exam Q)

| | BERT | BART |
|---|---|---|
| Architecture | Encoder only | Encoder + Decoder |
| Pretraining | MLM + NSP | Denoising (text corruption) |
| Generates text? | ❌ | ✅ |
| Seq2Seq tasks? | ❌ | ✅ |

---

## Quick Cheat Sheet

| Model | Type | Pretraining | Best Use |
|---|---|---|---|
| BERT | Encoder | MLM + NSP | Classification, QA |
| GPT | Decoder | Causal LM | Text generation |
| BART | Enc-Dec | Denoising | Summarization |
| T5 | Enc-Dec | Text-to-text denoising | Any NLP task |
