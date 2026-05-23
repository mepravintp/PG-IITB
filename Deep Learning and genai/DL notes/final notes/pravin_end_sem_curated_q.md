# Let's create the polished Markdown study guide based on the provided text
markdown_content = """# Deep Learning & GenAI
## End-Semester Open-Book Exam Prep Study Guide
**IIT Bombay ePGD | May 2026**

---

### Course Content Overview
| Section | Topics | Question Range |
| :---: | :--- | :---: |
| **A** | Language Modelling: N-grams, Perplexity, Entropy | Q1 – Q5 |
| **B** | Neural Networks: Backprop, Activations, CNNs | Q6 – Q11 |
| **C** | RNNs, LSTMs, BPTT | Q12 – Q15 |
| **D** | Transformers: Attention, Masking, Parameters | Q16 – Q21 |
| **E** | Pre-training, Fine-tuning, PEFT / LoRA | Q22 – Q25 |
| **F** | RLHF: Bradley-Terry, PPO-CLIP, DPO | Q26 – Q29 |
| **G** | GenAI: Seq2Seq, Decoding, RAG | Q30 |

---

## SECTION A: Language Modelling (N-grams, Perplexity, Entropy)

### Q1. Bigram MLE Probability
**Question:** A corpus contains the following sentences (with sentence markers):  
`I am Sam Sam I am I do not like eggs.`  
Find the unsmoothed $P(\text{am} \mid \text{I})$ and the Add-1 smoothed $P(\text{am} \mid \text{I})$.

**Given:**
* $\text{count(I am)} = 2$
* $\text{count(I)} = 3$
* Vocabulary $\mathcal{V} = \{\text{I, am, Sam, do, not, like, eggs}, </S>\}$
* Vocabulary size $|\mathcal{V}| = 8$

**Formula:**
* **Unsmoothed:** $P(w_2 \mid w_1) = \frac{\text{count}(w_1 \, w_2)}{\text{count}(w_1)}$
* **Add-1 Smoothed:** $P(w_2 \mid w_1) = \frac{\text{count}(w_1 \, w_2) + 1}{\text{count}(w_1) + |\mathcal{V}|}$

**Step-by-Step Solution:**
1.  **Step 1 (Unsmoothed):** $$P(\text{am} \mid \text{I}) = \frac{2}{3} \approx 0.667$$
2.  **Step 2 (Add-1 Smoothed):** $$P(\text{am} \mid \text{I}) = \frac{2 + 1}{3 + 8} = \frac{3}{11} \approx 0.273$$

**Final Answer:**
* **Unsmoothed Probability:** $0.667$
* **Add-1 Smoothed Probability:** $0.273$

> 💡 **Note:** Smoothing pulls the probability down because it redistributes probability mass to unseen bigrams. Zero probability for unseen n-grams makes perplexity infinite; smoothing prevents this catastrophic failure.

---

### Q2. Cross-Entropy and Perplexity from Model Probabilities
**Question:**
A bigram model scores a 4-token test sequence. The probability assigned to each actual next token is given below. Find (a) the per-token cross-entropy $H$ in bits, and (b) the perplexity ($PPL$).

**Given:**
* Token 1 (`the`): $q = \frac{1}{2}$
* Token 2 (`cat`): $q = \frac{1}{4}$
* Token 3 (`sat`): $q = \frac{1}{8}$
* Token 4 (`down`): $q = \frac{1}{8}$
* Total tokens $N = 4$

**Formula:**
* **Cross-Entropy:** $H = -\frac{1}{N} \sum_{t=1}^{N} \log_2(q(w_t))$
* **Perplexity:** $PPL = 2^H$

**Step-by-Step Solution:**
1.  **Step 1:** Compute the $\log_2$ probabilities for each token:
    * $\log_2\left(\frac{1}{2}\right) = -1$
    * $\log_2\left(\frac{1}{4}\right) = -2$
    * $\log_2\left(\frac{1}{8}\right) = -3$
    * $\log_2\left(\frac{1}{8}\right) = -3$
2.  **Step 2:** Sum the log values:
    $$\text{Sum} = (-1) + (-2) + (-3) + (-3) = -9$$
3.  **Step 3:** Calculate per-token cross-entropy $H$:
    $$H = -\frac{1}{4} \times (-9) = \frac{9}{4} = 2.25 \text{ bits/token}$$
4.  **Step 4:** Calculate Perplexity $PPL$:
    $$PPL = 2^{2.25} = 2^2 \times 2^{0.25} = 4 \times 1.1892 = 4.757$$

**Final Answer:**
* **Cross-Entropy ($H$):** $2.25 \text{ bits/token}$
* **Perplexity ($PPL$):** $4.76$

> 🔍 **Sanity Check:** > $$PPL = \left(\frac{1}{2} \times \frac{1}{4} \times \frac{1}{8} \times \frac{1}{8}\right)^{-\frac{1}{4}} = \left(\frac{1}{512}\right)^{-\frac{1}{4}} = 512^{\frac{1}{4}} \approx 4.757 \quad \checkmark$$

---

### Q3. Comparing Two Models via Perplexity
**Question:**
A test set of $100$ tokens gives a total $\log_2$-probability $= -650$ under Model A and a total $\log_2$-probability $= -700$ under Model B. Which model is better? Compute the perplexity for both.

**Given:**
* $N = 100 \text{ tokens}$
* $\sum \log_2 P_A = -650$
* $\sum \log_2 P_B = -700$

**Formula:**
* $H = -\frac{1}{N} \sum \log_2 q(w_t)$
* $PPL = 2^H$

**Step-by-Step Solution:**
1.  **Step 1 (Model A Entropy):** $H_A = -\frac{-650}{100} = 6.50 \text{ bits/token}$
2.  **Step 2 (Model B Entropy):** $H_B = -\frac{-700}{100} = 7.00 \text{ bits/token}$
3.  **Step 3 (Model A PPL):** $PPL_A = 2^{6.50} = 2^6 \times 2^{0.5} = 64 \times 1.4142 = 90.51$
4.  **Step 4 (Model B PPL):** $PPL_B = 2^{7.00} = 128$

**Final Answer:**
* **Model A:** $H_A = 6.50 \text{ bits}$, $PPL_A = 90.5$
* **Model B:** $H_B = 7.00 \text{ bits}$, $PPL_B = 128.0$
* **Conclusion:** **Model A is better** because it achieves a lower cross-entropy and a lower perplexity.

---

### Q4. Good-Turing Smoothing - Adjusted Counts
**Question:**
A corpus has $N = 100$ tokens. The counts-of-counts are: $N_1 = 10$ (words seen once), $N_2 = 4$ (words seen twice), and $N_3 = 2$ (words seen three times). Find: (a) the Good-Turing adjusted count $c^*$ for a word seen once ($c=1$), and (b) the total probability mass reserved for unseen events.

**Given:**
* $N = 100$
* $N_1 = 10, \; N_2 = 4, \; N_3 = 2$

**Formula:**
* **Adjusted Count:** $c^* = \frac{(c+1) \cdot N_{c+1}}{N_c}$
* **Unseen Mass:** $P(\text{unseen}) = \frac{N_1}{N}$

**Step-by-Step Solution:**
1.  **Step 1:** Calculate $c^*$ for $c=1$:
    $$c^* = \frac{(1+1) \cdot N_2}{N_1} = \frac{2 \times 4}{10} = \frac{8}{10} = 0.80$$
2.  **Step 2:** Calculate mass reserved for unseen words:
    $$P(\text{unseen}) = \frac{10}{100} = 0.10 \text{ (or } 10\%)$$

**Final Answer:**
* **Adjusted Count ($c^*$):** $0.80$
* **Unseen Mass Percentage:** $10\%$

> 💡 **Note:** A once-seen word is discounted to behave as if it was only encountered 0.8 times. While Maximum Likelihood Estimation (MLE) assigns a raw $0$ probability to unseen words, Good-Turing re-allocates this $10\%$ mass to make sure unseen sequences don't crash evaluations.

---

### Q5. Katz Backoff - Alpha Weight
**Question:**
Explain the algebraic expression for the Katz backoff weight $\alpha(w_{\text{prev}})$ that ensures the conditional distribution sums to 1. Then state why the numerator and denominator take those specific forms.

**Formula:**
$$\alpha(w_{\text{prev}}) = \frac{1 - \sum_{w_i \in \text{seen}} P^*(w_i \mid w_{\text{prev}})}{\sum_{w_i \in \text{unseen}} P_{\text{unigram}}(w_i)}$$
Where $P^*$ represents the Good-Turing discounted probability for observed bigrams.

**Explanation & Steps:**
1.  **Total Probability Constraint:** Total probability over the entire vocabulary must sum exactly to 1.
2.  **Leftover Mass Allocation:** The observed bigrams already consume a combined probability mass equal to $\sum_{\text{seen}} P^*(w_i \mid w_{\text{prev}})$.
3.  **The Numerator Role:** The numerator represents the total **leftover probability mass** available for distribution: $1 - (\text{mass used by observed bigrams})$.
4.  **The Denominator Role:** The denominator calculates the total unigram mass of all unseen words. We divide by this factor to **rescale** the leftover mass proportionally over the target vocabulary words.

**Final Answer:**
The $\alpha$ weight dynamically acts as a rescaling normalization constant that collects the discounted mass leftover from Good-Turing estimation and splits it back down to unseen bigrams proportional to their underlying baseline unigram frequencies.

---
---

## SECTION B: Neural Networks (Backprop, Activations, CNNs)

### Q6. Forward & Backward Pass Chain Rule (Core Backprop)
**Question:**
For a single hidden-layer neural network with no bias parameters, perform a forward and backward pass to calculate the loss and the gradients for all weights and input variables.

**Given Network Constants:**
* Input $x = 2.0$
* Weights: $w_1 = 3.0, \; w_2 = -1.0$
* Bias $b = 1.0$
* Target Output $= 5.0$
* **Operations:** $$z_1 = w_1 \cdot x + b \implies a_1 = \text{ReLU}(z_1) \implies z_2 = w_2 \cdot a_1 \implies \text{Loss} = (z_2 - \text{target})^2$$

**Analytical Derivatives (Chain Rule):**
* $\frac{\partial \text{Loss}}{\partial z_2} = 2 \cdot (z_2 - \text{target})$
* $\frac{\partial \text{Loss}}{\partial w_2} = \frac{\partial \text{Loss}}{\partial z_2} \cdot a_1$
* $\frac{\partial \text{Loss}}{\partial a_1} = \frac{\partial \text{Loss}}{\partial z_2} \cdot w_2$
* $\frac{\partial \text{Loss}}{\partial z_1} = \frac{\partial \text{Loss}}{\partial a_1} \cdot \text{ReLU}'(z_1) \quad \text{where } \text{ReLU}'(z_1) = 1 \text{ if } z_1 > 0 \text{ else } 0$
* $\frac{\partial \text{Loss}}{\partial w_1} = \frac{\partial \text{Loss}}{\partial z_1} \cdot x$
* $\frac{\partial \text{Loss}}{\partial b} = \frac{\partial \text{Loss}}{\partial z_1}$
* $\frac{\partial \text{Loss}}{\partial x} = \frac{\partial \text{Loss}}{\partial z_1} \cdot w_1$

**Step-by-Step Execution:**
1.  **Step 1:** $z_1 = (3.0 \times 2.0) + 1.0 = 7.0$
2.  **Step 2:** $a_1 = \text{ReLU}(7.0) = 7.0$
3.  **Step 3:** $z_2 = -1.0 \times 7.0 = -7.0$
4.  **Step 4:** $\text{Loss} = (-7.0 - 5.0)^2 = (-12.0)^2 = 144.0$
5.  **Step 5:** $\frac{\partial \text{Loss}}{\partial z_2} = 2 \times (-12.0) = -24.0$
6.  **Step 6:** $\frac{\partial \text{Loss}}{\partial w_2} = -24.0 \times 7.0 = -168.0$
7.  **Step 7:** $\frac{\partial \text{Loss}}{\partial a_1} = -24.0 \times (-1.0) = 24.0$
8.  **Step 8:** Since $z_1 = 7.0 > 0$, $\text{ReLU}'(7.0) = 1 \implies \frac{\partial \text{Loss}}{\partial z_1} = 24.0 \times 1 = 24.0$
9.  **Step 9:** $\frac{\partial \text{Loss}}{\partial w_1} = 24.0 \times 2.0 = 48.0$
10. **Step 10:** $\frac{\partial \text{Loss}}{\partial b} = 24.0 \quad \vert \quad \frac{\partial \text{Loss}}{\partial x} = 24.0 \times 3.0 = 72.0$

**Final Answer:**
* $\text{Loss} = 144.0$
* $\text{Gradients: } w_1\text{.grad} = 48.0, \; w_2\text{.grad} = -168.0, \; b\text{.grad} = 24.0, \; x\text{.grad} = 72.0$

> ⚠️ **Dying ReLU Warning:** If our input had been $x = -5.0$, then $z_1 = -14.0$. At this negative point, $\text{ReLU}(-14) = 0$ and its derivative drops to $0$. This causes **all** upstream parameter gradients to instantly zero out, demonstrating the classic *dying ReLU problem*.

---

### Q7. Sigmoid Derivative and Softplus
**Question:**
The Softplus activation is defined as $\text{SP}(x) = \ln(1 + e^x)$ and the Sigmoid function is $\sigma(x) = \frac{1}{1 + e^{-x}}$.  
(a) Prove that $\frac{d}{dx}[\text{SP}(x)] = \sigma(x)$.  
(b) Compute the second derivative $\frac{d^2}{dx^2}[\text{SP}(x)]$ in terms of $\sigma(x)$.  
(c) Evaluate its mathematical behavior as $x \to +\infty$ and $x \to -\infty$.

**Formula References:**
* $\frac{d}{dx}[\ln(u)] = \frac{1}{u} \frac{du}{dx}$
* $\frac{d}{dx}[\sigma(x)] = \sigma(x)(1 - \sigma(x))$

**Step-by-Step Proof & Solution:**
1.  **Step 1 (First Derivative):** Let $u = 1 + e^x$. Taking the derivative yields:
    $$\frac{d}{dx}[\ln(1 + e^x)] = \frac{e^x}{1 + e^x}$$
2.  **Step 2 (Simplification):** Divide both the numerator and denominator by $e^x$:
    $$\frac{\frac{e^x}{e^x}}{\frac{1 + e^x}{e^x}} = \frac{1}{e^{-x} + 1} = \sigma(x) \quad \checkmark$$
3.  **Step 3 (Second Derivative):** $$\frac{d^2}{dx^2}[\text{SP}(x)] = \frac{d}{dx}[\sigma(x)] = \sigma(x)(1 - \sigma(x)) = \frac{e^x}{(1 + e^x)^2}$$
4.  **Step 4 (Asymptotic Behavior):** * As $x \to +\infty$: $\sigma(x) \to 1 \implies \sigma(x)(1 - \sigma(x)) \to 1(0) = 0$
    * As $x \to -\infty$: $\sigma(x) \to 0 \implies \sigma(x)(1 - \sigma(x)) \to 0(1) = 0$
5.  **Step 5 (Maximum Curvature):** The maximum value of this second derivative peaks precisely at $x = 0$, where $\sigma(0) = 0.5 \implies 0.5 \times (1 - 0.5) = 0.25$.

**Final Answer:**
* (a) Proved: $\frac{d}{dx}\text{SP}(x) = \sigma(x)$
* (b) $\frac{d^2}{dx^2}\text{SP}(x) = \sigma(x)(1 - \sigma(x))$ with a maximum value of $0.25$ at $x=0$.
* (c) The curvature vanishes smoothly at both extreme tails, ensuring stable structural dynamics without gradient explosions.

---

### Q8. CNN Output Shape Calculation
**Question:**
An encoder network applies three sequence layers of Conv2D operations. Given the structural specifications below, compute the final output feature dimensions after each layer.

**Given Configuration:**
* Input Image Shape: `(Batch=1, Channels=3, Height=64, Width=64)`
* Layer Sequence Output Channels: $3 \to 32 \to 64 \to 128$
* Hyperparameters (All Layers): $\text{Kernel } (k) = 3, \; \text{Stride } (s) = 2, \; \text{Padding } (p) = 1$

**Formula:**
$$\text{Output Spatial Size} = \lfloor \frac{n + 2p - k}{s} \rfloor + 1$$

**Step-by-Step Shape Tracking:**
1.  **Layer 1:** $$\text{Dim} = \lfloor \frac{64 + 2(1) - 3}{2} \rfloor + 1 = \lfloor \frac{63}{2} \rfloor + 1 = 31 + 1 = 32 \implies \mathbf{(1, 32, 32, 32)}$$
2.  **Layer 2:** $$\text{Dim} = \lfloor \frac{32 + 2(1) - 3}{2} \rfloor + 1 = \lfloor \frac{31}{2} \rfloor + 1 = 15 + 1 = 16 \implies \mathbf{(1, 64, 16, 16)}$$
3.  **Layer 3:** $$\text{Dim} = \lfloor \frac{16 + 2(1) - 3}{2} \rfloor + 1 = \lfloor \frac{15}{2} \rfloor + 1 = 7 + 1 = 8 \implies \mathbf{(1, 128, 8, 8)}$$

**Final Answer:**
The final feature representation shape outputting from the encoder is **`(1, 128, 8, 8)`**.

---

### Q9. CNN Parameter Count and Linear Layer Bug
**Question:**
A student constructs a vision pipeline for $28 \times 28$ input images:  
`Conv2d(1, 32, k=5, s=1, no_pad)` $\to$ `MaxPool(2, 2)` $\to$ `Conv2d(32, 64, k=5, s=1, no_pad)` $\to$ `MaxPool(2, 2)` $\to$ `Linear(64*7*7, 10)`.  
(a) Trace tensor dimensions across layers.  
(b) Will the code execute safely or crash at the linear layer interface?  
(c) Calculate total trainable parameters inside the linear head if corrected.

**Formula References:**
* $\text{Conv Size (Unpadded)} = n - k + 1$
* $\text{MaxPool Size (Stride 2)} = \lfloor \frac{n}{2} \rfloor$
* $\text{Linear Module Parameters} = (\text{input\_features} \times \text{output\_features}) + \text{biases}$

**Step-by-Step Debugging:**
1.  **Step 1 (Conv1):** Input $28 \times 28 \implies 28 - 5 + 1 = 24 \to \text{Shape: } (B, 32, 24, 24)$
2.  **Step 2 (MaxPool1):** Dimensions halve $\implies \frac{24}{2} = 12 \to \text{Shape: } (B, 32, 12, 12)$
3.  **Step 3 (Conv2):** Input $12 \times 12 \implies 12 - 5 + 1 = 8 \to \text{Shape: } (B, 64, 8, 8)$
4.  **Step 4 (MaxPool2):** Dimensions halve $\implies \frac{8}{2} = 4 \to \text{Shape: } \mathbf{(B, 64, 4, 4)}$
5.  **Step 5 (Crash Validation):** The true flattened structural output size is $64 \times 4 \times 4 = 1024$. The student's code initializes the layer expecting $64 \times 7 \times 7 = 3136$ channels. **The program will crash** due to size mismatch errors.
6.  **Step 6 (Parameter Computations):** Correcting the network head to `Linear(1024, 10)` requires:
    $$\text{Params} = (1024 \times 10) + 10 = 10,240 + 10 = 10,250$$

**Final Answer:**
* **Actual Final Tensor Shape:** $(B, 64, 4, 4)$
* **Execution Status:** **Will Crash** at runtime.
* **Corrected Linear Layer Parameters:** $10,250$ weights and biases.

---

### Q10. PyTorch Code - Forward Pass Shape and Element Count
**Question:**
A network architecture contains: `Conv2d(1, 8, k=3, s=1, p=1)` $\to$ `ReLU` $\to$ `MaxPool(2,2)` $\to$ `Conv2d(8, 16, k=3, s=1, p=1)` $\to$ `ReLU` $\to$ `MaxPool(2,2)`. Given a random tensor initialized as `torch.randn(1, 1, 28, 28)`, determine the exact final shape and total scalar elements returned by calling `out.numel()`.

**Structural Properties:**
* A convolutional layer with stride$=1$ and padding$= \lfloor \frac{\text{kernel}}{2} \rfloor$ preserves original spatial boundaries.
* `MaxPool(2,2)` scales down spatial dimensions by a factor of 2.

**Step-by-Step Dimension Walkthrough:**
1.  **Step 1:** `Conv1` preserves boundaries $\to (1, 8, 28, 28)$
2.  **Step 2:** First `MaxPool` halves spatial dimensions $\to (1, 8, 14, 14)$
3.  **Step 3:** `Conv2` preserves boundaries $\to (1, 16, 14, 14)$
4.  **Step 4:** Second `MaxPool` halves spatial dimensions $\to \mathbf{(1, 16, 7, 7)}$
5.  **Step 5:** Total elements calculation via `numel()` product rule:
    $$\text{Elements} = 1 \times 16 \times 7 \times 7 = 784$$

**Final Answer:**
* **Output Tensor Shape:** `torch.Size([1, 16, 7, 7])`
* **Total Output Elements (`numel`):** $784$

---

### Q11. MSE Loss Forward Pass (Two-Layer Unactivated Network)
**Question:**
Calculate the raw Mean Squared Error (MSE) forward loss across a linear two-layer architecture without biases or nonlinear activation blocks.

**Given Variables:**
* Input Vector: $x = [1, 1, 1]^T$
* Target Scalar $y = 50$
* Weight Matrix 1 ($2 \times 3$): 
    $$W_1 = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$
* Weight Matrix 2 ($1 \times 2$): 
    $$W_2 = \begin{pmatrix} 1 & 1 \end{pmatrix}$$

**Formula Matrix Framework:**
* $h = W_1 \cdot x$
* $\hat{y} = W_2 \cdot h$
* $\text{MSE Loss} = (\hat{y} - y)^2$

**Step-by-Step Solution:**
1.  **Step 1:** Matrix-vector product for hidden layer $h$:
    $$h = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \cdot \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1(1) + 2(1) + 3(1) \\ 4(1) + 5(1) + 6(1) \end{pmatrix} = \begin{pmatrix} 6 \\ 15 \end{pmatrix}$$
2.  **Step 2:** Matrix product for prediction scalar $\hat{y}$:
    $$\hat{y} = \begin{pmatrix} 1 & 1 \end{pmatrix} \cdot \begin{pmatrix} 6 \\ 15 \end{pmatrix} = (1 \times 6) + (1 \times 15) = 21$$
3.  **Step 3:** Compute squared distance loss metric:
    $$\text{Loss} = (21 - 50)^2 = (-29)^2 = 841$$

**Final Answer:**
The calculated **MSE loss is $841$**.

---
---

## SECTION C: RNNs, LSTMs, and BPTT

### Q12. RNN / GRU / LSTM Parameter Counts
**Question:**
A recurrent network layer uses an input embedding dimension of $m = 100$ alongside a hidden state size of $n = 128$. Find the total number of trainable parameter weights for: (a) Vanilla RNN, (b) GRU, and (c) LSTM blocks.

**Structural Equations:**
* A single network gate represents an affine mapping transforming stacked vectors $[h_{t-1}; x_t]$ from dimension $(n+m)$ to hidden space layer size $n$.
* Total parameter calculation per gating function $= n \times (n + m) + n \text{ (biases)} = n \times (m + n + 1)$
* **Gate Counts:** Vanilla RNN $= 1$ gate $\mid$ GRU $= 3$ gates $\mid$ LSTM $= 4$ gates

**Step-by-Step Scaling Calculations:**
1.  **Step 1:** Compute parameter footprint for a single core base gating layer block:
    $$\text{Base Block} = 128 \times (100 + 128 + 1) = 128 \times 229 = 29,312 \text{ parameters}$$
2.  **Step 2 (Vanilla RNN):** $1 \times 29,312 = \mathbf{29,312}$
3.  **Step 3 (GRU):** $3 \times 29,312 = \mathbf{87,936}$
4.  **Step 4 (LSTM):** $4 \times 29,312 = \mathbf{117,248}$

**Final Answer:**
* **(a) Vanilla RNN Parameters:** $29,312$
* **(b) GRU Layer Parameters:** $87,936$
* **(c) LSTM Layer Parameters:** $117,248$

> 💡 **Structural Context:** GRUs require 3 internal gates (Reset $r$, Update $z$, and Candidate Hidden $\tilde{h}$). LSTMs track 4 standard architectural gates (Forget $f$, Input $i$, Candidate Cell $\tilde{C}$, and Output $o$).

---

### Q13. BPTT - Vanishing Gradient in Vanilla RNN
**Question:**
In a vanilla RNN modeled by $h_t = \tanh(W \cdot h_{t-1} + U \cdot x_t)$, write the mathematical expression for the temporal backpropagation gradient $\frac{\partial h_T}{\partial h_1}$ as a product of Jacobians and explain why gradients vanish for extended time contexts.

**Mathematical Proof Steps:**
1.  **Step 1 (Chain Rule):** Apply the chain rule across sequential time intervals:
    $$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$
2.  **Step 2 (Jacobian Expansion):** Each individual transition step evaluates to:
    $$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}\left(1 - \tanh^2(z_t)\right) \cdot W \quad \text{where } z_t = W \cdot h_{t-1} + U \cdot x_t$$
3.  **Step 3 (Norm Upper Bound):** Since the derivative of hyperbolic tangent is bounded within the range $(0, 1]$, the scale of each factor is upper-bounded by the matrix norm $\|W\|$:
    $$\left\| \frac{\partial h_T}{\partial h_1} \right\| \le \|W\|^{T-1}$$
4.  **Step 4 (Asymptotic Convergence):** If the spectral radius or norm constraint $\|W\| < 1$, then as sequence depth increases, $\|W\|^{T-1} \to 0$ exponentially. This prevents optimization signals from reaching early sequence updates.

**Final Answer:**
Gradients diminish exponentially according to $\|W\|^{T-1}$. When $\|W\| < 1$ for large $T$, early hidden step states become untrainable because their updates drop to zero.

> 💡 **Note:** Conversely, if $\|W\| > 1$, gradients can explode out of bounds. This specific problem is commonly handled by applying hard **gradient clipping threshold cuts**.

---

### Q14. LSTM - Forget Gate and Long-Term Memory
**Question:**
An LSTM cell updates its structural memory via $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. To carry information from sequence step $t=1$ all the way to a classification head down at step $t=100$:  
(a) What value must the forget gate vectors $f_t$ maintain?  
(b) Explain mathematically how this design bypasses vanishing gradients.

**Derivation & Analysis:**
1.  **Step 1 (Partial Cell Derivative):** The derivative of the state channel across a single step matches the forget gate matrix:
    $$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$
2.  **Step 2 (Long Range Product Tracking):** Expanding this down to the start of the sequence yields:
    $$\frac{\partial C_{100}}{\partial C_1} = \prod_{t=2}^{100} f_t$$
3.  **Step 3 (Stable Highway Condition):** If the model sets the gate biases to keep $f_t \approx 1.0$, the gradient product simplifies to:
    $$\frac{\partial C_{100}}{\partial C_1} \approx (1.0)^{99} = 1.0$$

**Final Answer:**
* (a) The gate activations must be **$f_t \approx 1$**.
* (b) Setting the forget gate to 1 preserves the error signal across 99 steps without exponential decay. This **additive state update** path acts as an error-propagation highway, giving LSTMs a clear advantage over vanilla RNNs.

---

### Q15. RNN Sentiment Classification: Why Aggregate Hidden States
**Question:**
When building a text sentiment classifier over an RNN backbone, contrast the strategy of using only the terminal hidden vector $h_T$ against averaging all hidden states $\frac{1}{T}\sum_{t=1}^T h_t$. Why does using only the final state $h_T$ often fail on longer documents?

**Step-by-Step Explanatory Analysis:**
1.  **Information Loss:** Although RNNs theoretically forward sequence history inside $h_T$, vanishing gradients mean early tokens ($h_1, h_2$) are weakly represented in the final vector.
2.  **Vulnerability to Long Sequences:** If a crucial sentiment indicator appears at the very beginning of a 50-word review (e.g., *"Terrible script, though the acting was passable..."*), the vanishing gradient effect can wash out its influence by the time the model processes step 50.
3.  **The Averaging Advantage:** Averaging all states ($\frac{1}{T}\sum h_t$) balances the contribution of every token position, ensuring early sentiment indicators are preserved.

**Final Answer:**
Relying only on $h_T$ leaves the model vulnerable to vanishing gradients and long-term forgetting. Aggregating all hidden states preserves information from the entire sequence.

---
---

## SECTION D: Transformers (Attention, Masking, Parameter Counts)

### Q16. Scaled Dot-Product Attention (Manual Calculation)
**Question:**
Given the simplified sequence token matrices below with a query size of $d_k = 2$, calculate the exact attention output vector for the first row query ($q_1$).

**Given Matrices:**
$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$
$$\text{Query 1: } q_1 = [1, 0] \quad \vert \quad \text{Keys: } k_1 = [1, 0], \, k_2 = [1, 0] \quad \vert \quad \text{Values: } v_1 = [1, 0], \, v_2 = [0, 1]$$

**Attention Mechanism Formula:**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$

**Step-by-Step Solution:**
1.  **Step 1 (Raw Dot Scores):** Compute dot products of $q_1$ against all key vectors:
    * $\text{score}_1 = q_1 \cdot k_1 = [1, 0] \cdot [1, 0] = 1$
    * $\text{score}_2 = q_1 \cdot k_2 = [1, 0] \cdot [1, 0] = 1 \implies \text{Scores} = [1, 1]$
2.  **Step 2 (Scaling):** Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.4142$:
    $$\text{Scaled Scores} = \left[\frac{1}{\sqrt{2}}, \, \frac{1}{\sqrt{2}}\right] \approx [0.707, \, 0.707]$$
3.  **Step 3 (Softmax Normalization):** Since both scaled inputs are identical, the normalized weights are equal:
    $$\alpha = \text{softmax}([0.707, \, 0.707]) = [0.5, \, 0.5]$$
4.  **Step 4 (Value Aggregation):** Linear combination of value vectors:
    $$\text{Output} = 0.5 \cdot v_1 + 0.5 \cdot v_2 = 0.5 \cdot [1, 0] + 0.5 \cdot [0, 1] = [0.5, \, 0.5]$$

**Final Answer:**
The scaled dot-product attention output vector for query row 1 is **`[0.5, 0.5]`**.

> 💡 **Core Insight:** When key representations are identical, attention splits evenly across them. Scaling by $\frac{1}{\sqrt{d_k}}$ keeps dot products from growing too large and saturating the softmax function.

---

### Q17. Self-Attention - Two Output Rows
**Question:**
Using the defined model parameters below where $d_k = 2$ and $\sqrt{2} \approx 1.4142$, compute both output rows produced by the self-attention layer.

**Given Matrices:**
$$Q = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$
$$\text{Tokens: } q_1 = [1, 1], \, q_2 = [0, 2] \quad \vert \quad k_1 = [1, 0], \, k_2 = [1, 1] \quad \vert \quad v_1 = [2, 0], \, v_2 = [0, 2]$$

**Step-by-Step Processing Pipeline:**
1.  **Step 1 (Row 1 Raw Scores):** * $\text{score}_{1,1} = q_1 \cdot k_1 = (1 \times 1) + (1 \times 0) = 1$
    * $\text{score}_{1,2} = q_1 \cdot k_2 = (1 \times 1) + (1 \times 1) = 2$
2.  **Step 2 (Row 1 Scaling):** $$\text{Scaled}_1 = \left[\frac{1}{1.414}, \, \frac{2}{1.414}\right] = [0.707, \, 1.414]$$
3.  **Step 3 (Row 1 Softmax Activation):**
    * $e^{0.707} = 2.028, \quad e^{1.414} = 4.113 \implies \text{Sum} = 2.028 + 4.113 = 6.141$
    * $\alpha_1 = \left[\frac{2.028}{6.141}, \, \frac{4.113}{6.141}\right] = [0.330, \, 0.670]$
    * $\text{Output}_1 = 0.330 \cdot [2, 0] + 0.670 \cdot [0, 2] = [0.660, \, 1.340]$
4.  **Step 4 (Row 2 Raw Scores):**
    * $\text{score}_{2,1} = q_2 \cdot k_1 = (0 \times 1) + (2 \times 0) = 0$
    * $\text{score}_{2,2} = q_2 \cdot k_2 = (0 \times 1) + (2 \times 1) = 2$
5.  **Step 5 (Row 2 Scaling & Softmax):**
    * $\text{Scaled}_2 = \left[\frac{0}{1.414}, \, \frac{2}{1.414}\right] = [0.0, \, 1.414]$
    * $e^{0} = 1.0, \quad e^{1.414} = 4.113 \implies \text{Sum} = 1.0 + 4.113 = 5.113$
    * $\alpha_2 = \left[\frac{1.0}{5.113}, \, \frac{4.113}{5.113}\right] = [0.196, \, 0.804]$
    * $\text{Output}_2 = 0.196 \cdot [2, 0] + 0.804 \cdot [0, 2] = [0.392, \, 1.608]$

*(Note: The provided PDF context contains a calculation shortcut stating row 2 matches score vectors $2$ and $2$ leading to uniform $[1,1]$ output weights, but exact evaluation reveals the numbers above based on the inputs).*

**Final Answer:**
* **Attention Output Matrix Row 1:** `[0.66, 1.34]`
* **Attention Output Matrix Row 2:** `[0.39, 1.61]` (or `[1.0, 1.0]` under reference approximation)

---

### Q18. Three Masking Types in Transformers
**Question:**
Describe the three distinct masking methods used during Transformer training and inference pipelines.

**Detailed Explanatory Breakdowns:**
1.  **Causal Masking (Decoder Self-Attention):** Ensures that at any given sequence token position $t$, the model cannot attend to future contexts where $t' > t$. Future attention scores are overridden with $-\infty$ prior to running softmax, driving those connection weights to $0$. This enforces **autoregressive** sequence generation.
2.  **Padding Masking (Encoder & Decoder Layers):** Prevents variable-length sequence padding tokens from influencing core attention updates. Their raw values are set to $-\infty$ so they contribute zero weight to the context representations.
3.  **Masked Language Modeling (MLM Masking - BERT Pre-training):** Used during bidirectional pre-training objectives. $15\%$ of all input sequence tokens are selected for processing. Out of those selected tokens: $80\%$ are replaced with a static `[MASK]` token, $10\%$ are substituted with a random vocabulary token, and the remaining $10\%$ are left completely unchanged. The network is then optimized to predict the original tokens at these modified positions.

**Final Answer Summary:**
* **Causal Mask:** Enforces step-by-step autoregressive generation.
* **Padding Mask:** Neutralizes structural padding within batched training sequences.
* **MLM Mask:** Drives bidirectional token prediction objectives during pre-training.

---

### Q19. Transformer Encoder Parameter Count
**Question:**
Calculate total trainable parameter weights (ignoring biases) for a standard Transformer encoder network matching the architectural configuration below.

**Given Configuration:**
* Hidden Embedding Layer Size ($d_{\text{model}}$) $= 512$
* Total Transformer Layers ($L$) $= 6$
* Feed-Forward Hidden Network Size ($d_{\text{ff}}$) $= 2048$
* Vocabulary Dimensions ($V$) $= 30,000$
* Maximum Sequence Context Depth ($\text{max\_len}$) $= 512$ (Learned Positional Approach)
* **Head Configuration:** Tied Output Weights (Output head parameters share space with Input Token Embeddings)

**Formula Framework:**
* **Attention Weights Per Layer:** $4 \times d_{\text{model}}^2 \quad (\text{comprising matrices } W_Q, W_K, W_V, W_O \text{ each sized } d \times d)$
* **FFN Weights Per Layer:** $2 \times d_{\text{model}} \times d_{\text{ff}} \quad (\text{comprising } W_1: d \times d_{\text{ff}} \text{ and } W_2: d_{\text{ff}} \times d)$
* **Token Embedding Allocation:** $V \times d_{\text{model}}$
* **Positional Embedding Allocation:** $\text{max\_len} \times d_{\text{model}}$
* **Grand Total Weight Formula:** $$\text{Params} = L \times \left(4d_{\text{model}}^2 + 2d_{\text{model}}d_{\text{ff}}\right) + (V \times d_{\text{model}}) + (\text{max\_len} \times d_{\text{model}})$$

**Step-by-Step Parameter Computation:**
1.  **Step 1 (Layer Attention Weights):** $4 \times 512^2 = 4 \times 262,144 = 1,048,576 \text{ params}$
2.  **Step 2 (Layer FFN Weights):** $2 \times 512 \times 2048 = 2,097,152 \text{ params}$
3.  **Step 3 (Single Layer Sum):** $1,048,576 + 2,097,152 = 3,145,728 \text{ params}$
4.  **Step 4 (All 6 Layers Combined):** $6 \times 3,145,728 = 18,874,368 \text{ params}$
5.  **Step 5 (Token Embeddings):** $30,000 \times 512 = 15,360,000 \text{ params}$
6.  **Step 6 (Learned Position Parameters):** $512 \times 512 = 262,144 \text{ params}$
7.  **Step 7 (Grand Total Matrix Sum):** $$\text{Total} = 18,874,368 + 15,360,000 + 262,144 = 34,496,512$$

**Final Answer:**
The network contains approximately **$34.5 \text{ Million}$ parameters**. 

> 💡 **Design Variation:** If the model used **Sinusoidal Positional Embeddings** instead of learned ones, this cost would drop to $0$ parameters, bringing the total down to $34.23 \text{ Million}$. Notice that the Feed-Forward blocks ($\approx 12.6\text{M}$) and Token Embeddings ($\approx 15.4\text{M}$) dominate the parameter footprint.

---

### Q20. BERT-Base Parameter Count Verification
**Question:**
Verify the standard $110\text{M}$ parameter size figure for the standard BERT-Base model by checking its structural layout configuration.

**Given Layout Constants:**
* $d_{\text{model}} = 768$
* $L = 12 \text{ Encoder Layers}$
* $d_{\text{ff}} = 3072$
* Vocabulary $V = 30,522$
* $\text{max\_len} = 512$ (Learned Positional Vectors)
* Tied Output Prediction Head

**Step-by-Step Parameter Evaluation:**
1.  **Step 1 (Layer Attention):** $4 \times 768^2 = 4 \times 589,824 = 2,359,296 \text{ params/layer}$
2.  **Step 2 (Layer FFN):** $2 \times 768 \times 3072 = 4,718,592 \text{ params/layer}$
3.  **Step 3 (Total Core Layer Cost):** $2,359,296 + 4,718,592 = 7,077,888 \text{ params}$
4.  **Step 4 (All 12 Layers Cumulative):** $12 \times 7,077,888 = 84,934,656 \text{ params}$
5.  **Step 5 (Token Embeddings Matrix):** $30,522 \times 768 = 23,440,896 \text{ params}$
6.  **Step 6 (Position Embeddings Matrix):** $512 \times 768 = 393,216 \text{ params}$
7.  **Step 7 (Summing All Vectors):** $$\text{Total} = 84,934,656 + 23,440,896 + 393,216 = 108,768,768 \approx 110\text{ Million}$$

**Final Answer:**
The analytical parameters calculate exactly to **$108.77\text{M}$**, verifying the industry-standard reference of **$110\text{M}$** parameters for BERT-Base.

---

### Q21. Why Scaled Dot-Product? Softmax Saturation
**Question:**
Explain why raw token attention dot products are scaled down by dividing by $\sqrt{d_k}$ prior to entering the softmax function. What happens if this scaling factor is omitted as the network dimension size $d_k$ grows large?

**Mathematical Derivation:**
1.  **Step 1 (Component Assumptions):** Assume the elements of query vector $Q$ and key vector $K$ are independent random variables with a mean of zero and unit variance ($\mu=0, \sigma^2=1$).
2.  **Step 2 (Variance Expansion):** The dot product computation represents a summation of $d_k$ individual products:
    $$\text{Variance}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \implies \text{Standard Deviation} = \sqrt{d_k}$$
3.  **Step 3 (The Problem with High Dimensions):** For large hidden dimensions (e.g., $d_k = 512$), the standard deviation reaches $\approx 22.6$. This wide distribution leads to extremely large raw dot product values.
4.  **Step 4 (Softmax Saturation):** These large values push the softmax function into its flat, outer regions. The function outputs a nearly flat **one-hot vector**, which causes its mathematical derivatives to drop to near-zero. This stalls gradient flow and stops training.
5.  **Step 5 (The Fix):** Dividing all scores by $\sqrt{d_k}$ scales the variance back down to $1.0$, stabilizing the softmax distribution and keeping gradients flowing during training.

**Final Answer:**
Without this scaling factor, large values of $d_k$ cause the softmax function to saturate, which kills gradient flow and stalls training. Dividing by $\sqrt{d_k}$ controls variance and keeps gradients active.

---
---

## SECTION E: Pre-training, Fine-tuning, and PEFT / LoRA

### Q22. Last-Layer vs Full Fine-Tuning Parameter Ratio
**Question:**
A BERT-Base network ($\approx 110\text{M}$ parameters, $d=768$) is fine-tuned for a 3-class classification task. Compare (a) the parameters updated via last-layer tuning versus (b) a full fine-tuning run. Compute the parameter ratio and evaluate the memory implications.

**Given Constants:**
* Total baseline parameters $= 110,000,000$
* Hidden State Size $d = 768$
* Target Classification Classes $C = 3$

**Formula:**
$$\text{Linear Classification Head parameters} = (d \times C) + C = (d + 1) \times C$$
$$\text{Storage Ratio} = \frac{\text{Head Params}}{\text{Total Base Params}}$$

**Step-by-Step Processing:**
1.  **Step 1 (Head Allocation):** $\text{Params} = (768 + 1) \times 3 = 769 \times 3 = 2,307 \text{ parameters}$
2.  **Step 2 (Full Run Footprint):** Full parameter updates require optimization across all $110,000,000$ parameters.
3.  **Step 3 (Proportional Scaling):** $$\text{Ratio} = \frac{2,307}{110,000,000} \approx 0.00002097 \implies \mathbf{0.0021\%}$$
4.  **Step 4 (Optimizer Memory Analysis):** Standard Adam optimization maintains 3 state values per active parameter (the tracking gradient plus two rolling momentum metrics). 
    * **Full Fine-Tuning Optimization Memory:** $110\text{M} \times 3 \times 4 \text{ bytes (FP32)} \approx \mathbf{1.32 \text{ GB}}$
    * **Last-Layer Tuning Optimization Memory:** $2307 \times 3 \times 4 \text{ bytes} \approx \mathbf{27.6 \text{ KB}}$

**Final Answer:**
* **(a) Last-Layer Head Count:** $2,307$ parameters.
* **(b) Full Fine-Tuning Count:** $110\text{M}$ parameters.
* **(c) Parameter Ratio:** $\approx 0.002\%$
* **(d) Memory Footprint Profile:** Full fine-tuning demands **$1.3 \text{ GB}$** of active GPU memory just for the optimizer states, whereas tuning only the last layer requires a negligible **$28 \text{ KB}$**.

---

### Q23. LoRA Parameter Savings Calculation
**Question:**
Apply Low-Rank Adaptation (LoRA) using a rank of $r = 8$ to a single base weight matrix sized $768 \times 768$. Then calculate the total parameter savings when applying this configuration to the projection tracking matrices $W_Q$ and $W_V$ across all 12 BERT-Base layers.

**Given Specification Constraints:**
* Hidden Dimensions $d = 768$
* LoRA Hyperparameter Rank $r = 8$
* Target Target Density: 2 tracking matrices ($W_Q, W_V$) per layer across 12 layers total.

**Formula Framework:**
LoRA freezes the primary weights $W_0$ and updates them via low-rank decomposition matrices: $\Delta W = B \cdot A$, where $B$ is a matrix of shape $d \times r$ and $A$ is a matrix of shape $r \times d$.
$$\text{LoRA Trainable Params Per Matrix} = (d \times r) + (r \times d) = 2 \times d \times r$$
$$\text{Standard Full Parameter Matrix Count} = d \times d = d^2$$

**Step-by-Step Parameter Comparison:**
1.  **Step 1 (LoRA Per Matrix Size):** $2 \times 768 \times 8 = 12,288$ parameters.
2.  **Step 2 (Original Full Matrix Size):** $768^2 = 589,824$ parameters.
3.  **Step 3 (Proportional Evaluation):** $$\% \text{ Size Layer Cost} = \frac{12,288}{589,824} \approx 0.02083 \implies \mathbf{2.08\%}$$
4.  **Step 4 (Total Model System Overhead):** $$\text{Global LoRA Params} = 12,288 \times 2 \text{ matrices} \times 12 \text{ layers} = 294,912 \text{ parameters } (\approx 0.29\text{M})$$
5.  **Step 5 (Global Model Ratio Comparison):** $\frac{0.29\text{M}}{110\text{M}} \approx \mathbf{0.27\%}$

**Final Answer:**
* **(a) Matrix Parameter Comparison:** LoRA updates $12,288$ parameters compared to the original matrix size of $589,824$.
* **(b) Single Matrix Ratio:** **$2.08\%$** of full weight size.
* **(c) Total Model System Overhead:** **$294,912$ total active parameters** (which represents only **$0.27\%$** of the entire BERT-Base architecture).

---

### Q24. Prompt Tuning - Trainable Parameter Count
**Question:**
In soft prompt tuning, $m$ learnable embedding prompt vectors of dimension $d$ are prepended to incoming token sequences while the rest of the network is frozen. Calculate the total trainable parameter footprint when applying this approach to a $175\text{B}$ parameter GPT-3 model.

**Given Configuration:**
* Prepended Soft Tokens ($m$) $= 50$
* Model Embedding Width ($d$) $= 768$
* GPT-3 Base Footprint Size $= 175,000,000,000$ parameters

**Formula:**
$$\text{Trainable Parameters} = m \times d$$

**Step-by-Step Parameter Computation:**
1.  **Step 1:** Multiply soft token length by feature dimension size:
    $$\text{Params} = 50 \times 768 = 38,400 \text{ parameters}$$
2.  **Step 2 (Proportional Ratio Calculation):**
    $$\text{Ratio} = \frac{38,400}{175,000,000,000} \approx \mathbf{2.2 \times 10^{-7}}$$

**Final Answer:**
Prompt tuning requires updating only **$38,400$ parameters**, which is a tiny **$0.000022\%$** of the total GPT-3 model footprint (making it roughly 4.5 million times smaller than a full fine-tuning run).

---

### Q25. MLM Masking Budget - BERT Pre-training
**Question:**
A input sequence containing $200$ text tokens is processed using standard BERT Masked Language Modeling (MLM). Calculate the exact token distribution split across the masking budget.

**Given Constraints:**
* Total Input Context Size $= 200 \text{ tokens}$
* **BERT Masking Rules:** Select $15\%$ of total tokens, then split those selected tokens into an $80 / 10 / 10$ ratio.

**Step-by-Step Distribution Computations:**
1.  **Step 1 (Determine Selected Token Count):** $200 \times 0.15 = 30 \text{ target tokens}$
2.  **Step 2 (Apply 80% Hard Mask Rule):** $30 \times 0.80 = \mathbf{24 \text{ tokens replaced with `[MASK]`}}$
3.  **Step 3 (Apply 10% Random Mutation Rule):** $30 \times 0.10 = \mathbf{3 \text{ tokens replaced with a random word}}$
4.  **Step 4 (Apply 10% Constant Retention Rule):** $30 \times 0.10 = \mathbf{3 \text{ tokens kept completely unchanged}}$

> 🔍 **Sanity Check:** $24 + 3 + 3 = 30 \text{ selected tokens} \quad \checkmark$

**Final Answer:**
* **`[MASK]` Substitutions:** $24$ tokens
* **Random Token Swaps:** $3$ tokens
* **Unchanged Retentions:** $3$ tokens

---
---

## SECTION F: RLHF (Bradley-Terry, PPO-CLIP, and DPO)

### Q26. Bradley-Terry Loss Derivation and Numerical Computation
**Question:**
The Bradley-Terry preference framework defines winning probabilities via $P(y_w \succ y_l) = \sigma(r_w - r_l)$. A reward model is optimized by minimizing the negative log-likelihood of these pairwise comparisons. Compute the individual losses and the final average loss across the three preference evaluation pairs listed below.

**Given Dataset Pairs:**
* Pair 1: $\text{Winner Reward } (r_w) = 2.0 \quad \vert \quad \text{Loser Reward } (r_l) = 1.0$
* Pair 2: $\text{Winner Reward } (r_w) = 0.5 \quad \vert \quad \text{Loser Reward } (r_l) = 1.5$
* Pair 3: $\text{Winner Reward } (r_w) = 3.0 \quad \vert \quad \text{Loser Reward } (r_l) = 0.0$
* **Sigmoid Function:** $\sigma(z) = \frac{1}{1 + e^{-z}}$

**Formula:**
$$\text{Loss}_{\text{Pair}} = -\ln\left(\sigma(r_w - r_l)\right)$$
$$\text{Average Loss} = \frac{1}{3} \sum_{i=1}^{3} \text{Loss}_i$$

**Step-by-Step Numerical Processing:**
1.  **Step 1 (Pair 1 Evaluation):**
    * $z_1 = 2.0 - 1.0 = 1.0 \implies \sigma(1.0) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.3678} = 0.731$
    * $\text{Loss}_1 = -\ln(0.731) = \mathbf{0.313}$
2.  **Step 2 (Pair 2 Evaluation):**
    * $z_2 = 0.5 - 1.5 = -1.0 \implies \sigma(-1.0) = \frac{1}{1 + e^{1}} = \frac{1}{1 + 2.718} = 0.269$
    * $\text{Loss}_2 = -\ln(0.269) = \mathbf{1.313}$
3.  **Step 3 (Pair 3 Evaluation):**
    * $z_3 = 3.0 - 0.0 = 3.0 \implies \sigma(3.0) = \frac{1}{1 + e^{-3}} = \frac{1}{1 + 0.0498} = 0.953$
    * $\text{Loss}_3 = -\ln(0.953) = \mathbf{0.049}$
4.  **Step 4 (Average Aggregation):**
    $$\text{Loss}_{\text{Avg}} = \frac{0.313 + 1.313 + 0.049}{3} = \frac{1.675}{3} = \mathbf{0.5583}$$

**Final Answer:**
* **Individual Losses:** Pair 1 $= 0.313 \mid$ Pair 2 $= 1.313 \mid$ Pair 3 $= 0.049$
* **Dataset Average Loss:** **$0.558$**
* **Conclusion:** Pair 2 contributes the most to the loss ($1.313$). This happens because the reward model incorrectly ranked the rejected response higher than the preferred one, resulting in a large optimization penalty.

---

### Q27. Regularized RLHF Objective KL Penalty
**Question:**
The RLHF objective standard optimization equation penalizes model policy drifting away from the initial SFT reference model baseline:  
$$J(\theta) = \mathbb{E}\left[r(x,y) - \beta \ln \left(\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right)\right]$$  
Given a specific system output tracking state, (a) calculate the regularized reward scalar $r_s$, (b) interpret what this value implies about the model's current behavior, and (c) analyze the system's response if the log-probability ratio increases to $15$ and $25$.

**Given Evaluation State Constants:**
* Raw Model Reward ($r$) $= 1.5$
* Log Probability Policy Scaling Ratio $\ln\left(\frac{\pi_\theta}{\pi_{\text{ref}}}\right) = 0.3$
* Penalty Scaling Parameter ($\beta$) $= 0.1$

**Step-by-Step Processing:**
1.  **Step 1:** Calculate the regularized penalty reward $r_s$:
    $$r_s = 1.5 - \left(0.1 \times 0.3\right) = 1.5 - 0.03 = \mathbf{1.47}$$
2.  **Step 2 (Interpretation):** The small KL penalty ($0.03$) shows the updated policy has stayed close to the original SFT reference model. Since the regularized reward stays high, this response is beneficial and should be kept.
3.  **Step 3 (Escalation Case A):** If the log-probability ratio jumps to $15$:
    $$r_s = 1.5 - (0.1 \times 15) = 1.5 - 1.5 = \mathbf{0.0}$$
    This indicates **reward hacking** is starting to happen. The policy is shifting significantly just to game the reward metric, wiping out the positive evaluation score.
4.  **Step 4 (Escalation Case B):** If the log-probability ratio reaches $25$:
    $$r_s = 1.5 - (0.1 \times 25) = 1.5 - 2.5 = \mathbf{-1.0}$$
    The regularized reward becomes negative, which will actively suppress this generation path during optimization.

**Final Answer:**
* (a) **$r_s = 1.47$**
* (b) Indicates a stable, high-quality update path with minimal drift from the baseline model.
* (c) High log-ratios show the model is reward hacking, which drops the net reward to zero or negative values to stop unstable optimization paths.

---

### Q28. PPO-CLIP Objective - Three Cases
**Question:**
The proximal policy optimization clipped objective is defined using a tracking boundary of $\epsilon = 0.2$:  
$$L_{\text{CLIP}} = \min\left(\rho \cdot \hat{A}, \; \text{clip}(\rho, \, 1-\epsilon, \, 1+\epsilon) \cdot \hat{A}\right)$$  
Calculate the exact objective value $L_{\text{CLIP}}$ returned across each of the three operational edge cases listed below.

**Given Configuration Profiles:**
* Clipping Evaluation Range Bounds: $[1 - \epsilon, \, 1 + \epsilon] \implies [0.8, \, 1.2]$
* **Case (a):** Advantage Metric ($\hat{A}$) $= +2, \quad$ Probability Ratio ($\rho$) $= 1.5$
* **Case (b):** Advantage Metric ($\hat{A}$) $= +2, \quad$ Probability Ratio ($\rho$) $= 0.7$
* **Case (c):** Advantage Metric ($\hat{A}$) $= -2, \quad$ Probability Ratio ($\rho$) $= 1.5$

**Step-by-Step Evaluation:**
1.  **Step 1 (Case a):** Positive Advantage with a high probability ratio:
    * $\text{Term}_1 = \rho \cdot \hat{A} = 1.5 \times 2 = 3.0$
    * $\rho_{\text{clipped}} = \text{clip}(1.5, 0.8, 1.2) = 1.2 \implies \text{Term}_2 = 1.2 \times 2 = 2.4$
    * $L_{\text{CLIP}} = \min(3.0, 2.4) = \mathbf{2.4}$
2.  **Step 2 (Case b):** Positive Advantage with a low probability ratio:
    * $\text{Term}_1 = \rho \cdot \hat{A} = 0.7 \times 2 = 1.4$
    * $\rho_{\text{clipped}} = \text{clip}(0.7, 0.8, 1.2) = 0.8 \implies \text{Term}_2 = 0.8 \times 2 = 1.6$
    * $L_{\text{CLIP}} = \min(1.4, 1.6) = \mathbf{1.4}$
3.  **Step 3 (Case c):** Negative Advantage with a high probability ratio:
    * $\text{Term}_1 = \rho \cdot \hat{A} = 1.5 \times (-2) = -3.0$
    * $\rho_{\text{clipped}} = \text{clip}(1.5, 0.8, 1.2) = 1.2 \implies \text{Term}_2 = 1.2 \times (-2) = -2.4$
    * $L_{\text{CLIP}} = \min(-3.0, -2.4) = \mathbf{-3.0}$

**Final Answer:**
* **Case (a) Objective Value:** $2.4$
* **Case (b) Objective Value:** $1.4$
* **Case (c) Objective Value:** $-3.0$

> 💡 **Core Insight:** Clipping limits the incentive to over-optimize on highly positive actions (Case a), but it retains the full penalty if the model takes a bad action more frequently (Case c). This keeps policy updates conservative and stable.

---

### Q29. DPO - Direct Preference Optimisation
**Question:**
Direct Preference Optimization (DPO) eliminates the need for an independent reward model. (a) Write out the mathematical equation for the final DPO loss function, and (b) explain conceptually why this formulation removes the need for separate reinforcement learning loops.

**Formula Framework:**
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\ln \sigma \left(\beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**Step-by-Step Derivation Logic:**
1.  **Step 1 (The Optimal Policy Identity):** The analytical solution for the regularized RLHF objective shows the optimal policy $\pi^*(y \mid x)$ is mathematically related to the reward function:
    $$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x,y)}{\beta}\right)}{Z(x)}$$
2.  **Step 2 (Algebraic Inversion):** Rearranging this equation lets us express the implicit reward directly using log-probability ratios:
    $$r(x,y) = \beta \ln \left(\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right) + \beta \ln Z(x)$$
3.  **Step 3 (Canceling the Partition Function):** Substituting this reward expression back into the Bradley-Terry preference loss causes the complex partition function $Z(x)$ to cancel out completely.
4.  **Step 4 (Direct Optimization Formulation):** The final loss function depends entirely on log-probabilities that can be directly evaluated from the active policy $\pi_\theta$ and the baseline model $\pi_{\text{ref}}$.

**Final Answer:**
DPO reparameterizes the reward function using the policy itself. This collapses the traditional multi-stage RLHF pipeline into a single, stable supervised learning objective over preference pairs, completely eliminating the need for reward models, actor-critic architectures, or complex RL rollouts.

---
---

## SECTION G: GenAI (Seq2Seq, Decoding Strategies, and RAG)

### Q30. Greedy vs Beam Search Two-Step Decoding
**Question:**
Given a two-character generation vocabulary $\mathcal{V} = \{A, B\}$, trace a two-step decoding pass to find (a) the output sequence and probability generated via Greedy Decoding, (b) the output sequence and probability generated via a Width-2 Beam Search, and (c) explain why greedy decoding fails to find the global optimum here.

**Given Step Distribution Probabilities:**
* Step 1 Base Priors: $P(A) = 0.6, \quad P(B) = 0.4$
* Continuations branching out from token choice A: $P(A \mid A) = 0.5, \quad P(B \mid A) = 0.5$
* Continuations branching out from token choice B: $P(A \mid B) = 0.95, \quad P(B \mid B) = 0.05$

**Formula:**
$$\text{Total Sequence Probability } P(W) = P(w_1) \times P(w_2 \mid w_1)$$

**Step-by-Step Search Path Tracking:**
1.  **Step 1 (Greedy Execution Pass):**
    * **Step 1:** Select the argmax of the initial distribution: $\max(0.6, 0.4) \to \text{Pick Token } \mathbf{A}$
    * **Step 2:** Select the next argmax branching from A: $\max(0.5, 0.5) \to \text{Pick Token } \mathbf{A}$ (breaking the tie)
    * **Greedy Path Result:** Path `AA` $\implies P(\text{AA}) = 0.6 \times 0.5 = \mathbf{0.30}$
2.  **Step 2 (Beam Search Width-2 Full Path Evaluation):**
    Evaluate all possible length-2 generation paths across the tree:
    * **Path AA:** $P = 0.6 \times 0.5 = \mathbf{0.30}$
    * **Path AB:** $P = 0.6 \times 0.5 = \mathbf{0.30}$
    * **Path BA:** $P = 0.4 \times 0.95 = \mathbf{0.38} \quad \leftarrow \text{HIGHEST PROBABILITY}$
    * **Path BB:** $P = 0.4 \times 0.05 = \mathbf{0.02}$
3.  **Step 3 (Comparison):** The width-2 beam search tracks and selects path `BA`, which achieves a total probability of $0.38$.

**Final Answer:**
* **(a) Greedy Search Output:** Sequence `AA` with a probability of **$0.30$**.
* **(b) Beam Search (Width 2) Output:** Sequence `BA` with a probability of **$0.38$**.
* **(c) Suboptimality Explanation:** Greedy search committed early to token `A` because it looked best at step 1 ($0.6 > 0.4$). However, choosing the lower-probability token `B` opens up a highly likely continuation path ($0.95$). Greedy decoding's locally optimal choice led to a globally suboptimal sequence, whereas beam search successfully recovered the higher-probability path.

---
---

### QUICK REFERENCE - FORMULA CHEAT SHEET

| Domain Topic | Targeted Operational Concept | Mathematical Formula Syntax |
| :--- | :--- | :--- |
| **Language Modelling** | Unsmoothed Bigram Frequency | $P(w_2 \mid w_1) = \frac{\text{count}(w_1 w_2)}{\text{count}(w_1)}$ |
| **Language Modelling** | Add-1 Laplace Smoothing | $P(w_2 \mid w_1) = \frac{\text{count}(w_1 w_2) + 1}{\text{count}(w_1) + \vert\mathcal{V}\vert}$ |
| **Language Modelling** | Per-Token Cross-Entropy | $H = -\frac{1}{N} \sum \log_2 q(w_t)$ |
| **Language Modelling** | Perplexity Metric | $PPL = 2^H$ |
| **Language Modelling** | Good-Turing Adjusted Counts | $c^* = \frac{(c+1) \cdot N_{c+1}}{N_c}$ |
| **Vision Layers** | Conv2D Spatial Output Dimensions | $\text{Out} = \lfloor \frac{n + 2p - k}{s} \rfloor + 1$ |
| **Recurrent Systems** | Vanilla RNN Parameters Per Layer | $\text{Params} = n \times (m + n + 1)$ |
| **Recurrent Systems** | GRU Parameter Architecture Size | $\text{Params} = 3 \times n \times (m + n + 1)$ |
| **Recurrent Systems** | LSTM Parameter Architecture Size | $\text{Params} = 4 \times n \times (m + n + 1)$ |
| **Attention Layers** | Scaled Dot-Product Evaluation | $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$ |
| **Attention Layers** | Transformer Encoder Weight Layer | $\text{Layer Params} = 4d_{\text{model}}^2 + 2d_{\text{model}}d_{\text{ff}}$ |
| **Parameter Efficiency**| LoRA Low-Rank Parameter Layout | $\text{LoRA Params} = 2 \times d \times r \quad (\text{per target matrix})$ |
| **Parameter Efficiency**| Soft Prompt Tuning Footprint | $\text{Params} = m \times d$ |
| **Preference Tuning** | Bradley-Terry Loss (Single Input) | $\mathcal{L} = -\ln \sigma(r_w - r_l)$ |
| **Preference Tuning** | Regularized Reward Function | $r_s = r(x,y) - \beta \ln\left(\frac{\pi_\theta}{\pi_{\text{ref}}}\right)$ |
| **Preference Tuning** | PPO-CLIP Objective Engine | $L_{\text{CLIP}} = \min\left(\rho \hat{A}, \, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A}\right)$ |
| **Preference Tuning** | Direct Preference Optimization Loss | $\mathcal{L}_{\text{DPO}} = -\ln \sigma \left(\beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$ |
"""

# Write content out to target destination path
with open("DL_GenAI_ExamPrep_StudyGuide.md", "w") as f:
    f.write(markdown_content.strip())