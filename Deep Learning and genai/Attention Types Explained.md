# Attention Mechanisms Explained

This document explains attention in language models, with step-by-step descriptions and examples for each main attention type.

## What is Attention?

Attention is a mechanism that lets a model focus on the most relevant parts of an input sequence when producing an output. Instead of compressing all input information into a single fixed vector, attention calculates a weighted combination of input representations based on their relevance to the current output step.

### Core idea
- Query (`q`): the current decoder state or target token representation.
- Keys (`k`): representations of the input tokens.
- Values (`v`): the information associated with each input token.
- Attention scores measure how relevant each key is to the query.
- Softmax converts scores into normalized weights.
- The final output is a weighted sum of `v` values.

---

## 1. Additive Attention (Bahdanau)

### Step-by-step
1. Compute a score for each input token by combining the query and key with a learned feed-forward network.
2. Apply a nonlinearity (usually `tanh`).
3. Convert scores into weights using `softmax`.
4. Multiply each value by its weight and sum the results.

### Formula

<code>score(q, k_i) = v_a^T \tanh(W_q q + W_k k_i)</code>

<code>\alpha_i = \text{softmax}(score(q, k_i))</code>

<code>context = \sum_i \alpha_i \, v_i</code>

### How it solves
- It learns a flexible comparison between query and key using a small neural network.
- Works well when query and key dimensions differ.
- By using `tanh`, it can capture nonlinear relationships.

### Example
- Task: translate the English phrase "She reads books" into French.
- Query: decoder state at the step generating the French word.
- Keys/values: encoder representations for "She", "reads", "books".
- The attention layer scores each source word and assigns a high weight to "books" when producing "livres".

---

## 2. Dot-Product Attention (Luong)

### Step-by-step
1. Compute a score by taking the dot product of the query and each key.
2. Optionally scale the score by the square root of the key dimension.
3. Apply `softmax` to get weights.
4. Compute the weighted sum of values.

### Formula

<code>score(q, k_i) = q^T k_i</code>

<code>\alpha_i = \text{softmax}(score(q, k_i) / \sqrt{d_k})</code>

<code>context = \sum_i \alpha_i \, v_i</code>

### How it solves
- Uses a simple dot product for fast similarity computation.
- Efficient when query and key have the same dimension.
- Scaling by `\sqrt{d_k}` keeps gradients stable when dimension is large.

### Example
- Task: question answering over the sentence "The cat sat on the mat.".
- Query: embedding of the question token at the current step.
- Keys/values: embeddings of each sentence word.
- The model gives a larger weight to "cat" or "mat" depending on the question.

---

## 3. Self-Attention

### Step-by-step
1. Use the same sequence as source and target.
2. For each position, compute queries, keys, and values from the same token representations.
3. Compute attention scores between every pair of positions.
4. Use `softmax` to normalize scores across positions.
5. Combine values by weighted sum for each position.

### Formula

<code>Attention(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V</code>

### How it solves
- Allows each token to attend to every other token in the same sequence.
- Captures both short- and long-range dependencies directly.
- Enables parallel computation across sequence positions.

### Example
- Sentence: "The quick brown fox jumps.".
- At position of "jumps", self-attention may focus on "fox" to understand the subject.
- At position of "brown", it attends to "quick" and "fox" for descriptive context.

---

## 4. Multi-Head Attention

### Step-by-step
1. Project the input into multiple sets of queries, keys, and values.
2. Apply self-attention separately for each attention head.
3. Concatenate the outputs from all heads.
4. Project the concatenated result back to the model dimension.

### Formula

<code>head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)</code>

<code>MultiHead(Q,K,V) = \text{Concat}(head_1, \dots, head_h) W^O</code>

### How it solves
- Each head learns a different way to attend to the sequence.
- Some heads capture word order, others capture syntax or semantic links.
- Combines multiple perspectives for a richer representation.

### Example
- Input: "She finished her homework before dinner.".
- Head 1 may focus on subject-object relations: "She" and "homework".
- Head 2 may focus on temporal structure: "before" and "dinner".
- Head 3 may focus on noun modifiers: "her" with "homework".

---

## Summary

- Additive attention uses a learned feed-forward comparison and is flexible for mismatched dimensions.
- Dot-product attention is fast and efficient when query/key dimensions match.
- Self-attention computes relationships within the same sequence and enables parallel processing.
- Multi-head attention combines multiple attention patterns for stronger representation.

These attention variants are fundamental to modern transformer models and powerful for sequence modeling tasks like translation, summarization, and question answering.
