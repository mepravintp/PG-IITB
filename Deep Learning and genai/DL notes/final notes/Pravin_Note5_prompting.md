# Lecture 8 — Prompt-based Learning
**Course:** ePGD Deep Learning & GenAI · IIT Bombay  
**Instructor:** Prof. Tanmoy Chakraborty, IIT Delhi  
**Reference:** *Pre-train, Prompt, and Predict* — Liu et al. (ACL Survey)

---

## 1. The Language Model Scaling Wars

The rapid growth of model size (parameters) has been one of the defining trends of modern NLP.

| Model | Architecture | Parameters | Training Tokens |
|-------|-------------|------------|-----------------|
| ELMo | 2-layer biLSTM | 93M | ~1B |
| BERT-base | 12-layer Transformer | 110M | 3.3B |
| BERT-large | 24-layer Transformer | 340M | 3.3B |
| RoBERTa | 24-layer Transformer | 340M | ~30B |
| GPT-3 (small) | 12-layer Transformer | 125M | — |
| GPT-3 (medium) | 24-layer Transformer | 350M | — |
| GPT-3 (175B / "GPT-3") | 96-layer Transformer | **175B** | 300B |
| Megatron-NLG | — | 530B | — |

**GPT-3 architecture details (full model):**  
`n_layers = 96`, `d_model = 12288`, `n_heads = 96`, `d_head = 128`, `batch = 3.2M`, `lr = 0.6×10⁻⁴`

> **Key takeaway:** Scaling up model size is one of the most important ingredients for achieving state-of-the-art performance. GLUE scores have risen steadily from ELMo (2017) → BERT → GPT-2 → T5 alongside parameter count growth.

### GPT-3 Training Data Mix

| Dataset | Quantity | Weight | Epochs (for 300B tokens) |
|---------|----------|--------|--------------------------|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

---

## 2. What Does Scaling Buy Us? — In-Context Learning

GPT-3 introduced a new paradigm: **Language Models are Few-Shot Learners** (Brown et al., 2020).

### The Three Prompting Modes

#### 2.1 Fine-Tuning (traditional — *not* used for GPT-3 style)
- The model is updated via repeated **gradient updates** using a large corpus of labelled examples.
- Each new task requires a **separate fine-tuned copy** of the model.
- Example: feed `sea otter => loutre de mer`, `peppermint => menthe poivrée`, ..., then update weights.

#### 2.2 Zero-Shot
- The model receives **only a natural language task description** — no examples, no gradient updates.
- Example prompt:
  ```
  Translate English to French:
  cheese =>
  ```
- The model must infer the task purely from the instruction.

#### 2.3 One-Shot
- Task description + **one demonstration example**.
  ```
  Translate English to French:
  sea otter => loutre de mer
  cheese =>
  ```

#### 2.4 Few-Shot (K-shot)
- Task description + **K demonstration examples** packed into the prefix (K = 10–100 typically).
  ```
  Translate English to French:
  sea otter => loutre de mer
  peppermint => menthe poivrée
  plush giraffe => girafe peluche
  cheese =>
  ```
- **No gradient updates** are performed in any of these modes — the learning is "in-context."

---

## 3. GPT-3 Performance vs Fine-Tuning

### TriviaQA Results (Accuracy vs Model Size)

- All three modes (zero/one/few-shot) improve monotonically with model scale.
- At **175B parameters**, GPT-3 few-shot (K=64) **matches or exceeds fine-tuned SOTA** on TriviaQA.
- Larger models benefit disproportionately from more examples (the gap between zero-shot and few-shot widens with scale).

### Translation (Multi-BLEU Scores)

| Setting | En→Fr | Fr→En | En→De | De→En |
|---------|-------|-------|-------|-------|
| SOTA (Supervised) | 45.6 | 35.0 | 41.2 | 40.2 |
| GPT-3 Zero-Shot | 25.2 | 21.2 | 24.6 | 27.2 |
| GPT-3 One-Shot | 28.3 | 33.7 | 26.2 | 30.4 |
| GPT-3 Few-Shot | 32.6 | **39.2** | 29.7 | **40.6** |

> Only **7% of GPT-3's training data** is in languages other than English, yet it achieves near-SOTA translation performance. Improvements had **not plateaued** at 175B.

### Reading Comprehension (Harder Datasets)

| Setting | CoQA | DROP | SQuADv2 |
|---------|------|------|---------|
| Fine-tuned SOTA | 90.7 | 89.1 | 93.0 |
| GPT-3 Zero-Shot | 81.5 | 23.6 | 59.5 |
| GPT-3 Few-Shot | 85.0 | 36.5 | 69.8 |

> GPT-3 **struggles on harder datasets** requiring deep reasoning (DROP, SQuADv2). Gap to fine-tuned SOTA is large — multi-step reasoning remains a weakness of pure in-context learning.

---

## 4. Practical Challenge — Cost of Large Models

Fine-tuning 11B/175B models for each task is **extremely expensive**:
- Requires separate **11B-parameter copy** per task (Task A Model, Task B Model, Task C Model…)
- Storage, inference, and distribution costs grow linearly with the number of tasks.

This motivates **parameter-efficient** approaches.

---

## 5. Hard vs Soft Prompts

### 5.1 Hard / Discrete Prompts
- Human-readable **natural language** instructions/task descriptions.
- Used in zero/one/few-shot in-context learning.
- **Problems:**
  - Require domain expertise to craft.
  - Performance **lags far behind** SOTA model tuning.
  - Extremely **sensitive** to exact wording.

> **Example of sensitivity** (Liu et al., 2021 — LAMA-TREx P17, BERT-base-cased):
> 
> | Prompt | P@1 |
> |--------|-----|
> | [X] is located in [Y]. *(original)* | 31.29 |
> | [X] is located in which country or state? [Y]. | 19.78 |
> | [X] is located in which country? [Y]. | 31.40 |
> | [X] is located in which country? In [Y]. | **51.08** |
> 
> A **single-word change** can cause a drastic difference — from 19.78 to 51.08!

### 5.2 Continuous / Soft Prompts
Instead of human-readable text, learn **continuous embedding vectors** as the prompt.

**Progress timeline:**
1. Manual prompt design (Brown et al., 2020; Schick & Schütze, 2021)
2. Mining & paraphrasing to augment prompt sets (Jiang et al., 2020)
3. Gradient-based search for improved discrete prompts (Shin et al., 2020 — AutoPrompt)
4. Automatic generation using a separate LM like T5 (Gao et al., 2020)
5. **Learning continuous/soft prompts** (Liu et al., 2021; Li & Liang, 2021; Lester et al., 2021)

---

## 6. Prompt Tuning (Lester et al., 2021)

### What is a Soft Prompt?
> A sequence of **additional task-specific tunable token embeddings** prepended to the input text.

```
[P1] [P2] [P3] [P4]  +  [task input tokens]  →  Frozen LLM  →  output
 ←— soft prompt ——→      ←— frozen ——————————→
```

- Only the **prompt embeddings** (m × d parameters) are trained via backpropagation.
- The **entire LLM is frozen** — no weight updates.
- Trainable parameters = **m × d** (e.g., 50 × 768 = **38,400 parameters**).

### Parameter Efficiency Comparison

| Method | What is trained | Typical parameters |
|--------|-----------------|-------------------|
| Full fine-tuning | All weights | Billions |
| Prompt tuning | Soft prompt embeddings only | ~tens of thousands |
| Prefix tuning | Prefix keys/values per layer | L × 2 × m × d |
| LoRA | Low-rank decompositions | 2 × L × r × d |

### Prompt Tuning Scales with Model Size
At smaller scales (< 1B), prompt tuning underperforms model tuning. At **~10B+ parameters**, prompt tuning matches model tuning on SuperGLUE — it **becomes competitive with scale**.

---

## 7. Prefix Tuning (Li & Liang, ACL 2021)

A variant of soft prompting where **trainable prefix vectors** are prepended to the **keys and values of every Transformer layer** (not just the input).

### Fine-tuning vs Prefix-tuning

| Aspect | Fine-tuning | Prefix-tuning |
|--------|------------|---------------|
| What changes | Entire model weights per task | Only prefix vectors per task |
| Storage | Full model copy per task (11B×N) | Tiny prefix per task (~0.1%) |
| Shared backbone | No — separate models | Yes — one pretrained model |
| Tasks served simultaneously | No | Yes — mix-task batching |

**Practical advantage:** One shared 11B pretrained model + tiny task-specific prefixes — enables mixed-task batches and drastically reduces storage.

---

## 8. Parameter-Efficient Fine-Tuning (PEFT) Taxonomy

From the lecture's handwritten notes, PEFT methods are categorized into three families:

```
PEFT Methods
├── Additive
│   ├── Adapters      — small bottleneck modules inserted between layers
│   └── Soft prompts  — LoRA, DoRA, AdaLoRA (reparametrization)
├── Selective         — train only a subset of existing parameters
└── Reparametrization — low-rank decomposition of weight matrices (LoRA family)
```

### LoRA (Low-Rank Adaptation)
- Freezes the pretrained weight matrix W.
- Adds a low-rank decomposition: ΔW = A × B, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d), r ≪ d.
- Only A and B are trained; merged into W at inference for zero overhead.

---

## 9. Problems With Soft Prompts

Despite their parameter efficiency, soft prompts have real limitations:

| Problem | Explanation |
|---------|------------|
| Requires separate training | Must train a new soft prompt for every task |
| Not universal | Cannot pre-compute prompts for all possible tasks/inputs |
| Not user-friendly | Non-expert users cannot create soft prompts on the fly |
| Not interpretable | Prompt embeddings have no human-readable meaning |

> **Conclusion:** Hard/discrete prompts remain the **default choice** for end-users interacting with LLMs in practice. Soft prompts are used mainly in deployment pipelines where the task is fixed.

---

## 10. Advanced Prompting Techniques

### 10.1 Chain-of-Thought Prompting (CoT) — Wei et al., 2022

Instead of asking for a direct answer, the few-shot examples include **intermediate reasoning steps**.

| Standard Prompting | Chain-of-Thought Prompting |
|--------------------|---------------------------|
| Model Input: "The cafeteria had 23 apples. Used 20 for lunch, bought 6 more. How many?" | Same question, but the example shows step-by-step reasoning |
| Model Output: **27 ✗** (wrong) | Model Output: "23 − 20 = 3. Then 3 + 6 = **9** ✓" |

**Why it works:** The reasoning trace forces the model to decompose multi-step problems, reducing arithmetic and logical errors.

**Trigger phrase:** Adding `"Let's think step by step"` to the prompt (zero-shot CoT) also induces chain-of-thought reasoning without examples.

### 10.2 Self-Consistency CoT (Wang et al., 2022)

Addresses the brittleness of a single CoT path by **sampling multiple diverse reasoning chains** and voting.

**Procedure:**
1. Add "think step-by-step" to the question.
2. Sample the model **n times** with different random seeds/temperatures.
3. Collect n reasoning chains + n answers.
4. Apply a **majority vote** to select the final answer.

> Significantly improves over single-chain CoT, especially on math and commonsense reasoning benchmarks.

### 10.3 Tree-of-Thought (ToT) — Yao et al., 2023

Generalises CoT from a linear chain to a **tree** — the model explores multiple paths at each step.

**Key components:**
- **Branching:** At each reasoning step, generate **multiple candidate thoughts** (not just one).
- **Scoring:** An LLM evaluator scores each thought as "sure / likely / impossible."
- **Backtracking:** Prune unproductive branches and return to a previous node.
- **Search:** Uses BFS or DFS over the thought tree.

**Use case example:** *Game of 24* — given 4 numbers, reach 24 using arithmetic.  
`Input: 4 9 10 13` → ToT explores `10 − 4 = 6`, `4 + 9 = 13`, etc., evaluates each, and backtracks dead-ends.

**LLM-as-a-Judge:** ToT can use the LLM itself as the evaluator/scorer, making it self-contained.

### 10.4 Graph-of-Thought (GoT) — Besta et al., 2023

Generalises further from a tree to an **arbitrary directed graph** structure.

**Operations:**
- **Refining:** Loops in the graph — revisit and improve an existing thought.
- **Aggregating:** Vertices with multiple incoming edges — combine several thoughts into a new, synthesised thought.
- **Backtracking:** Abandon a bad path and return to a prior node.

**Progression of reasoning paradigms:**

| Paradigm | Structure | Key novelty |
|----------|-----------|-------------|
| IO (standard) | Single step | Direct answer |
| CoT | Linear chain | Intermediate LLM thoughts |
| CoT-SC | Multiple independent chains | Vote over chains |
| ToT | Tree | Branch + backtrack from any node |
| GoT | Directed graph | Aggregate multiple chains; refine via loops |

---

## 11. Summary & Key Comparisons

### Prompting Paradigm Evolution

```
Pre-training only (GPT-1)
        ↓
Pre-train + Fine-tune (BERT, GPT-2)
        ↓
Pre-train + Prompt (GPT-3, in-context learning)  ← no gradient updates
        ↓
Pre-train + Soft Prompt Tune (prompt tuning, prefix tuning)  ← few parameters
        ↓
Pre-train + PEFT (LoRA, adapters)  ← small adapters, frozen backbone
```

### Hard vs Soft Prompts — Quick Reference

| Property | Hard Prompt | Soft Prompt |
|----------|------------|-------------|
| Human-readable | ✓ Yes | ✗ No |
| Gradient updates needed | ✗ No | ✓ Yes (for prompt params) |
| Sensitive to wording | ✓ Very sensitive | ✗ Optimised |
| User-friendly | ✓ Yes | ✗ No |
| Performance (large models) | Good | Matches fine-tuning |
| Default for end-users | ✓ Yes | ✗ No |

---

## 12. Exam-Relevant Key Points

1. **GPT-3 paper title:** *Language Models are Few-Shot Learners* (Brown et al., 2020).
2. **Zero-shot:** task description only, no examples, no gradient updates.
3. **One-shot:** task description + 1 example, no gradient updates.
4. **Few-shot (K-shot):** task description + K examples, no gradient updates. K ≤ context window.
5. **Prompt sensitivity:** a single-word change can shift accuracy by over 30 points (LAMA benchmark).
6. **Prompt tuning parameters:** m × d (m = soft prompt length, d = embedding dim).
7. **Prefix tuning** inserts learnable vectors at every layer (keys & values), not just input.
8. **CoT** requires the reasoning steps to be shown in the few-shot examples (or triggered by "Let's think step by step").
9. **Self-consistency** = multiple CoT samples + majority vote.
10. **ToT** adds branching + scoring + backtracking; **GoT** adds aggregation + loops.
11. **PEFT taxonomy:** Additive (adapters, soft prompts) / Selective / Reparametrisation (LoRA).
12. **Prompt tuning catches up** to model tuning only at very large scale (≥ 10B parameters).

---

## 13. Recommended Reading

- Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Liu et al. (2021) — *Pre-train, Prompt, and Predict: A Systematic Survey* — [arXiv:2107.13586](https://arxiv.org/abs/2107.13586)
- Lester et al. (2021) — *The Power of Scale for Parameter-Efficient Prompt Tuning* — [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
- Li & Liang (ACL 2021) — *Prefix-Tuning: Optimizing Continuous Prompts for Generation* — [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- Wei et al. (2022) — *Chain-of-Thought Prompting Elicits Reasoning in LLMs* — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Wang et al. (2022) — *Self-Consistency Improves CoT Reasoning* — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Yao et al. (2023) — *Tree of Thoughts: Deliberate Problem Solving with LLMs* — [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)

---

*Notes compiled from Lecture 8 slides by Prof. Tanmoy Chakraborty, IIT Delhi — ePGD Deep Learning & GenAI, IIT Bombay, 2026*
