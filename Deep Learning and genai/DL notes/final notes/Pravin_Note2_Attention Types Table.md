# Attention Types Comparison Table

<table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #2c3e50; color: white; text-align: left;">
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Attention Type</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Definition</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Problem It Solves</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">How It Works</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Example</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ecf0f1;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Additive (Bahdanau)</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A mechanism that computes attention scores using a learned feed-forward network that combines the query and each key.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Allows flexible attention scoring when query and key dimensions are different or when nonlinear relationships are needed.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>score = v_a^T \tanh(W_q q + W_k k_i)</code><br><strong>Steps:</strong> (1) Apply linear projections to query and key, (2) Add them together, (3) Apply <code>tanh</code> nonlinearity, (4) Project to a scalar score with <code>v_a</code>, (5) Compute weights with softmax.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Translation:</strong> Translating "She reads books" to French. The model learns to give high attention weights to "books" when generating the French word "livres".</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Dot-Product (Luong)</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A simple attention mechanism that computes scores as the scaled dot product between query and key vectors.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Provides efficient, computationally fast attention when query and key have the same dimensionality. Scaling prevents gradient explosion.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>score = \frac{q^T k_i}{\sqrt{d_k}}</code><br><strong>Steps:</strong> (1) Compute dot product of query and each key, (2) Scale by <code>1/\sqrt{d_k}</code> for stability, (3) Apply softmax to get normalized weights, (4) Sum value vectors weighted by these scores.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Question Answering:</strong> Question: "Where did the cat sit?" over sentence "The cat sat on the mat." The model attends to "mat" and "sat" with high weights.</td>
    </tr>
    <tr style="background-color: #ecf0f1;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Self-Attention</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A mechanism where each token in a sequence attends to all other tokens (and itself) in the same sequence, using queries, keys, and values derived from the same input.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Captures both short- and long-range dependencies within a sequence without requiring separate encoder-decoder architecture. Enables parallel computation.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V</code><br><strong>Steps:</strong> (1) Project input into Q, K, V from same sequence, (2) Compute scores between all pairs, (3) Scale and apply softmax, (4) Weight each value vector by its score.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Language Understanding:</strong> In "The quick brown fox jumps", the word "jumps" attends to "fox" to understand the subject and to "quick" and "brown" for context.</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Multi-Head</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A mechanism that runs multiple attention operations in parallel, each with different learned projections, then concatenates and projects the results.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Allows the model to jointly attend to information from different representation subspaces. Captures multiple types of relationships simultaneously.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)</code><br><code>MultiHead = \text{Concat}(head_1, \ldots, head_h) W^O</code><br><strong>Steps:</strong> (1) Create <em>h</em> separate attention heads with independent projections, (2) Compute attention for each head, (3) Concatenate all outputs, (4) Project back to original dimension.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Sentence Analysis:</strong> "She finished her homework before dinner." Head 1 captures subject-object ("She"→"homework"), Head 2 captures temporal structure ("before"→"dinner"), Head 3 captures modifiers ("her"→"homework").</td>
    </tr>
  </tbody>
</table>

---

## Key Takeaways

| Aspect | Additive | Dot-Product | Self-Attention | Multi-Head |
| --- | --- | --- | --- | --- |
| **Complexity** | Higher (learned network) | Lower (simple dot product) | Medium | Higher (multiple heads) |
| **Best For** | Mismatched dimensions | Matched dimensions | Capturing dependencies | Complex relationships |
| **Computational Cost** | Moderate | Low | Medium | Higher |
| **Flexibility** | High (nonlinear) | Low (linear) | High | Very High |
