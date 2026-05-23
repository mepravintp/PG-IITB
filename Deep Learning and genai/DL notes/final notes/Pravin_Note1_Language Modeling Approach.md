# Comparison of Language Modeling Approaches

The following report outlines various concepts in language modeling, their definitions, the specific problems they address, their implementation methods, and their inherent limitations based on the provided source.

<table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #2c3e50; color: white; text-align: left;">
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Concept</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Definition</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Problem it Solves</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">How it Solves (Types, Formula, Example)</th>
      <th style="border: 1px solid #4d4d4d; padding: 12px;">Limitation</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ecf0f1;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>N-gram LM</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A statistical language model that predicts the next word based on the previous <em>n</em> words.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">It provides a simple method for estimating the conditional probabilities of word sequences.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>P(w_t \,|\, w_{t-n+1}, \dots, w_{t-1})</code><br><strong>How:</strong> It approximates the probability of the next word using counts of recent <em>n</em>-word histories in training data. More frequent histories get higher weight.<br><strong>Example:</strong> A bigram model predicts <em>"sat"</em> after <em>"cat"</em> when the sequence "cat sat" appears often.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Data sparsity for rare sequences, large storage requirements, and weak long-range dependency handling.</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Perplexity</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">An evaluation metric used to measure how effectively a language model predicts a sample.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">It quantifies the predictive performance of a language model.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>PP = 2^{-\left(\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i)\right)}</code><br><strong>How:</strong> It converts the average log-probability of a sequence into an effective branching factor. Lower values mean the model is less surprised by the actual data.<br><strong>Example:</strong> A model that assigns higher probability to the observed words will have lower perplexity.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A numeric score only; it does not explain why the model makes errors.</td>
    </tr>
    <tr style="background-color: #ecf0f1;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>RNN</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A neural network with recurrent connections designed to process sequences step by step.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Handles variable-length input and captures sequential dependencies.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>h_t = f\bigl( W x_t + U h_{t-1} \bigr)</code><br><strong>How:</strong> Each new token updates the hidden state using the previous state and the current input. This state carries contextual information forward through the sequence.<br><strong>Example:</strong> The previous words influence the prediction for the next word in a sentence.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Sequential processing is slow and it suffers from vanishing/exploding gradients and long-term dependency issues.</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>LSTM</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">An advanced RNN with memory cells and gating mechanisms for long-term state retention.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Solves the vanishing gradient problem and remembers long-term dependencies.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t</code><br><strong>How:</strong> Gates control what information is forgotten, what new input is stored, and what is exposed to the next step. This allows relevant signals to persist across many steps.<br><strong>Example:</strong> Retaining the meaning of <em>"book"</em> until later words like <em>"was amazing"</em> are processed.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">It remains sequential and is more computationally expensive than simpler RNNs.</td>
    </tr>
    <tr style="background-color: #ecf0f1;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Seq2Seq</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">An encoder-decoder architecture that maps an input sequence to an output sequence of different length.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Enables variable-length outputs for tasks like translation.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>How:</strong> The encoder reads the full input and compresses it into a context vector. The decoder then generates each output token using that vector plus its own hidden state, allowing output length to differ from input length.<br><strong>Example:</strong> Translating an English sentence into French token by token.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Information bottleneck from compressing all input into one vector may lose detail for long sentences.</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><strong>Attention</strong></td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">A mechanism to focus on relevant encoder states rather than relying on a fixed context vector.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Improves long-range dependency handling and reduces Seq2Seq bottlenecks.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;"><code>Attention(q, k, v) = \mathrm{softmax}\left( \frac{q \, k^\top}{\sqrt{d_k}} \right) v</code><br><strong>How:</strong> It computes similarity scores between a query and each key, then uses those scores to weight the value vectors. This directs the model’s focus to the most relevant source tokens for each output step.<br><strong>Types:</strong> Additive (Bahdanau), Dot-product (Luong), Self-attention, Multi-head.<br><strong>Example:</strong> Focusing on <em>"yesterday"</em> when generating <em>"hier"</em>.</td>
      <td style="border: 1px solid #bdc3c7; padding: 12px;">Can be expensive for large inputs and models due to quadratic attention cost.</td>
    </tr>
  </tbody>
</table>
