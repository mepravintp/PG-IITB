# Deep Learning Midterm - Sample Answers

---

## Section 1: Multiple Choice Questions

### Question 1: Optimizer Comparison (SGD vs. SGD with Momentum)

**Question:** After one parameter update step on the same gradient, which statement is true regarding standard SGD (opt_A) and SGD with momentum=0.9 (opt_B)?

**Answer:** **(b)** opt_B produces the same update as opt_A on the very first step, but will produce different (typically larger) updates on subsequent steps as the momentum buffer accumulates.

**Explanation:**
- In PyTorch, the momentum buffer is initialized to zero
- During the first update: both optimizers use only current gradient × learning rate
- Starting from the second step: opt_B adds the accumulated buffer to the gradient, resulting in different updates

---

### Question 2: Autoencoder Loss Function

**Question:** Which loss function is most commonly used for reconstructing grayscale images with pixel values normalized to [0, 1]?

**Answer:** **(b) Binary Cross-Entropy (BCE) Loss**

**Explanation:**
- When pixel values are normalized between 0 and 1, they are treated as probabilities
- BCE loss is mathematically suitable for this range
- Effectively penalizes the difference between original and reconstructed pixel intensities

---

### Question 3: CNN Architecture Properties (Multi-Select)

#### Part (a)
**Question:** What is the effective receptive field (RF) of a single neuron in the output of Layer 1 (kernel size 3, dilation 2, padding 2)?

**Answer:** **(ii) 5 × 5**

**Explanation:**
- Formula: $K_{eff} = K + (K-1)(D-1)$
- For Layer 1: K=3, D=2
- Calculation: $3 + (3-1)(2-1) = 3 + 2 = 5$

#### Part (b)
**Question:** Which statements are true about this network?

**Answer:** **(i) and (iii)** are true

**Explanation for (i):**
- Output spatial dimensions remain 128×128
- Padding and dilation choices maintain the input size

**Explanation for (iii):**
- Receptive field calculated iteratively:
  - Layer 1 RF: 5
  - Layer 2 (K=3, D=1): adds (3-1) = 2 → RF becomes 7
  - Layer 3 (K=3, D=1): adds 2 → RF becomes 9×9

**Note:** Dilation does not increase learnable parameters; it only changes weight spacing
---

## Section 2: Short Answer Type Questions

### Question 1: PyTorch Autograd

**Question:** What is the value of y.grad for $z = x^2 + 2xy + y^3$ where $x=3.0$ and $y=4.0$?

**Answer:** **54.0**

**Explanation:**
- Take the partial derivative: $\frac{\partial z}{\partial y} = 2x + 3y^2$
- Substitute values: $2(3.0) + 3(4.0^2) = 6 + 3(16) = 6 + 48 = 54.0$

---

### Question 2: Image Segmentation Techniques

**Answer:** Common techniques include:
- **Semantic Segmentation** - classifies each pixel into a category
- **Instance Segmentation** - differentiates between individual objects of the same category
- **Panoptic Segmentation** - combines semantic and instance segmentation
- Classical methods: Thresholding, Edge Detection

**Note:** Based on general deep learning knowledge

---

### Question 3: BPTT and RNNs

**Answer:** 
- **BPTT (Backpropagation Through Time)** - unrolls the RNN over time steps to calculate gradients
- **Problem:** RNNs suffer from Vanishing/Exploding Gradients
- **Solutions:** 
  - LSTMs and GRUs use gating mechanisms to allow information flow through long sequences
  - Gradient Clipping limits gradient magnitude

**Note:** Based on general deep learning knowledge

---

## Section 3: Long Answer Type Questions

### Question 1: Linear Model Code Trace

**Question:** Predict the exact printed output for the provided linear model code with parameters:
- w = 1.0, b = 0.5, lr = 0.5, x = 2.0, target = 3.0

**Step-by-Step Calculations:**

| Step | Formula | Calculation |
|------|---------|-------------|
| **Prediction** | $\text{pred} = w \cdot x + b$ | $(1.0 \times 2.0) + 0.5 = 2.5$ |
| **Loss** | $\text{loss} = (pred - target)^2$ | $(2.5 - 3.0)^2 = 0.25$ |
| **w.grad** | $\frac{\partial \text{Loss}}{\partial w} = 2(pred - target) \cdot x$ | $2(-0.5) \times 2.0 = -2.0$ |
| **b.grad** | $\frac{\partial \text{Loss}}{\partial b} = 2(pred - target)$ | $2(-0.5) = -1.0$ |
| **new_w** | $w - (lr \cdot w.grad)$ | $1.0 - (0.5 \times -2.0) = 2.0$ |
| **new_b** | $b - (lr \cdot b.grad)$ | $0.5 - (0.5 \times -1.0) = 1.0$ |

**Expected Output:**
```
pred: 2.5
loss: 0.25
w.grad: -2.0
b.grad: -1.0
new_w: 2.0
new_b: 1.0
```

---

### Question 2: CustomCNN Trace and Analysis

#### Part (a): Dimension Calculation (32×32 input)

Using the convolution output formula: $\text{out} = \lfloor \frac{H + 2P - K}{S} \rfloor + 1$

| Layer | Formula | Calculation | Output Size |
|-------|---------|-------------|-------------|
| **Conv1** | k=5, s=2, p=1 | $\lfloor \frac{32 + 2 - 5}{2} \rfloor + 1$ | **15×15** |
| **Conv2** | k=3, s=1, p=0 | $\lfloor \frac{15 + 0 - 3}{1} \rfloor + 1$ | **13×13** |
| **MaxPool** | k=2, s=2 | $\lfloor \frac{13 - 2}{2} \rfloor + 1$ | **6×6** |

**Feature shape:** `torch.Size([batch, 32, 6, 6])`  
**Output shape:** `torch.Size([batch, 10])`

#### Part (b): 64×64 Input Analysis

Following the same logic with 64×64 input:
- Conv1: 31×31
- Conv2: 29×29
- MaxPool: 14×14
- **Feature map:** 32×14×14

**Will it crash?** **YES**

**Why?**
- Linear layer is fixed to accept: $32 \times 6 \times 6 = 1152$ inputs
- 64×64 input produces: $32 \times 14 \times 14 = 6272$ inputs
- **Error:** Dimension mismatch at `x = self.classifier(x)`

#### Part (c): Total Learnable Parameters

| Layer | Calculation | Parameters |
|-------|-------------|-----------|
| **Conv1** | $(16 \times 3 \times 5 \times 5) + 16$ | **1,216** |
| **Conv2** | $(32 \times 16 \times 3 \times 3) + 32$ | **4,640** |
| **Linear** | $(1152 \times 10) + 10$ | **11,530** |
| **TOTAL** | $1216 + 4640 + 11530$ | **17,386** |

---
