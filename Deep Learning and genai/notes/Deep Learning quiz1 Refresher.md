# Computer Vision Basics - Deep Learning Fundamentals

---

## 1. The Perceptron Model

A **perceptron** is a linear classifier inspired by biological neurons.

**What it does:**
- Computes a weighted sum of inputs
- Outputs a binary value based on a threshold
- Can model simple Boolean gates: AND, OR
- **Cannot** solve non-linearly separable problems (e.g., XOR)

**Formula:**
```
y = 1 if ∑(w_i × x_i) ≥ T, else 0
```

**Real-world example:**
- Recognizing a printed letter "A" using a 20×20 grid of photocells

### ⚠️ The Perceptron's Limitation
**Problem:** A single-layer perceptron is fundamentally a linear classifier that can only learn **linearly separable patterns**. It cannot solve complex logic problems like the **XOR function** (which requires non-linear decision boundaries).

---

## 2. Activation Functions

**Purpose:** Introduce non-linearity into neural networks

**Why they matter:**
- WITHOUT activation functions: Multi-layer networks just behave like a simple linear model
- WITH activation functions: Networks can model complex, non-linear data patterns

**Common activation functions:**

| Function | Formula | Key Property |
|----------|---------|--------------|
| **ReLU** | f(x) = max(0, x) | Prevents gradient vanishing; computationally fast |
| **Sigmoid** | f(x) = 1/(1+e^-x) | Smooth output; can cause vanishing gradients |
| **Tanh** | f(x) = (e^x - e^-x)/(e^x + e^-x) | Similar to sigmoid, range [-1, 1] |

**Example:**
- If a neuron receives -5, ReLU outputs **0**
- If a neuron receives +3, ReLU outputs **3**

### 🔄 From Sigmoid/Tanh to ReLU: The Evolution

**Problem with Sigmoid/Tanh:**
- Both have **maximum derivatives around 0.25 or less**
- In deep networks, gradients multiply during backpropagation: 0.25 × 0.25 × 0.25 = 0.015625 (and it gets worse!)
- This leads to the **vanishing gradient problem** where early layers stop learning entirely

**ReLU Solution:**
- Has a **constant derivative of 1** for all positive values
- Preserves the gradient signal as it flows backward through layers
- Enables training of **much deeper networks** without gradient collapse
- Result: **Faster training** and networks that can actually learn meaningful patterns in deep layers

---

## 3. Multi-Layer Perceptron (MLP)

**Structure:**
- Input layer → Hidden layers (one or more) → Output layer
- Each neuron connects to **every** neuron in the next layer (**fully connected**)

**Key insight:**
- Hidden layers allow the network to learn a **hierarchical decomposition** of features
- Enables solving complex, **non-linear problems**

**Formula (Layer Output):**
```
a_1 = σ(W_1 × x + b_1)
```
where σ is the activation function

**Real-world example:**
- Predicting image category ("T-shirt," "Trouser," etc.) from Fashion-MNIST dataset

### ✅ From Perceptron to MLP: The Solution

**Problem (Revisited):** Single-layer perceptrons cannot solve non-linear problems

**MLP Solution:**
- **Introduces hidden layers** with non-linear activation functions
- These hidden units act as **flexible building blocks**
- The network can now **approximate any continuous function** (Universal Approximation Theorem)
- **Result:** Networks can solve XOR, image classification, and other complex, non-linearly separable problems

**Why it works:**
- Each hidden layer learns to detect different patterns
- Stacking layers creates increasingly abstract representations
- Non-linear activation functions prevent the whole network from reducing to a linear model

---

## 4. Universal Approximation Theorem

**Statement:**
> A feedforward network with **at least one hidden layer** and a **finite number of neurons** can approximate **any continuous function** to **arbitrary accuracy**.

**Key insight:**
- Neurons act as **flexible building blocks**
- With enough neurons, you can mimic any complex curve

**Formula:**
```
g(x) = ∑(a_j × σ(w_j^T × x + b_j))
```

**Practical example:**
- To replicate a highly jagged mathematical function, simply increase the number of hidden neurons

---

## 5. Deep vs. Wide Networks

### Wide Networks (Many neurons, few layers)
- ❌ Tend to **memorize** training examples (overfitting)
- ❌ Inefficient in parameters

### Deep Networks (Few neurons, many layers)
- ✅ Learn **hierarchical features** (edges → parts → objects)
- ✅ More parameter-efficient
- ✅ Represent complex functions with fewer weights

**Complexity Formula:**
```
Number of "pieces" modeled ≈ m^L
(where L = depth, m = width)
```

**Example - Deep network hierarchy:**
- Lower layers: Learn color blobs
- Higher layers: Learn semantic concepts (e.g., "dog's ear")

### 🔄 From Wide to Deep Networks: The Efficiency Gain

**Problem with Wide Networks:**
- A **single massive layer** with millions of neurons to memorize data
- Massive parameter count → slow training and overfitting
- Cannot capture the **hierarchical structure** present in real-world data
- Example: A 256×256 image needs 65,536 input nodes; adding one wide hidden layer creates billions of parameters!

**Deep Networks Solution:**
- **Multiple layers, each learning different levels of abstraction**
- Early layers learn simple patterns (edges, textures)
- Middle layers combine those patterns (shapes, parts)
- Deep layers learn complex concepts (objects, faces)
- **Result:** Represent the same complexity with **exponentially fewer parameters** (m^L instead of linear growth)
- **Bonus:** Better generalization because the network learns meaningful hierarchical features

---

## 6. Softmax Layer (Classification)

**Purpose:** Convert raw output scores into probability distribution

**What it does:**
- Outputs values that sum to 1.0
- Allows model to express confidence levels for each class

**Formula:**
```
f_c = e^(g_c) / ∑(e^(g_i))
```

**Example:**
- A 3-class model outputs: **[0.8, 0.1, 0.1]**
- Interpretation: 80% confidence in class 1, 10% in class 2, 10% in class 3

---

## 7. Loss Functions

Loss functions measure **error between prediction and reality**.

### Type 1: Mean Squared Error (MSE)
**Use case:** Regression problems

**Formula:**
```
J = (1/N) × ∑(y_i - y_i*)²
```

### Type 2: Cross-Entropy
**Use case:** Classification problems

**Formula:**
```
L = -∑(y_i × log(f_i))
```

**Key property:** Penalizes wrong confident predictions heavily

**Example:**
- Model predicts "Cat" with 10% confidence when true label is "Cat"
- Result: High Cross-Entropy loss (model was confident but wrong!)

---

## 8. Stochastic Gradient Descent (SGD)

**Purpose:** Train the network by minimizing loss

**How it works:**
1. Calculate gradient of loss function
2. Take a small step in the **opposite direction** of the gradient
3. Update weights to reduce error
4. Repeat until convergence

**Formula:**
```
θ_(t+1) = θ_(t) - η × ∇ℓ(f_θ(x_i), y_i)
```
where η = learning rate

**Analogy:**
- Like hiking down a hill in the direction of steepest descent

**Example:**
- Weight changes from 0.5 → 0.48 after an error
- "Sliding down" the slope of the loss curve

---

## 9. Computation Graphs

**Definition:** Unrolled representation of network operations as nodes and edges

### Types:

**Static Graphs**
- ✅ Built once before execution
- ✅ Strong optimization
- ❌ Less flexible

**Dynamic Graphs**
- ✅ Built during execution
- ✅ Easier debugging
- ❌ Slower optimization

**Graph components:**
- **Nodes:** Values (x, W, b) and operations (matmul, add, ReLU)
- **Directed Edges:** Show dependencies (which values flow into an operation)

---

## 10. Automatic Differentiation (Backpropagation)

**Reverse-mode AD** = Backpropagation

### Why Reverse-mode for Neural Networks?
- Computes gradients for **millions of parameters** in a **single backward pass**
- Much more efficient than forward-mode for single-output functions (Loss)

**Key question it answers:**
> "How does the loss depend on each intermediate variable?"

**Process:**
1. Start from final output (loss)
2. Propagate backward
3. Calculate how each weight affected the final error

**Efficiency example:**
- Forward-mode: Compute gradients for each parameter separately (slow)
- Reverse-mode: One backward pass for all parameters (fast)

---

## 11. Backpropagation and the Chain Rule

**Purpose:** Determine how much each weight **contributed to the total loss**

**Connection weights:** Adjusted proportionally to their contribution

**Chain Rule Formula:**
```
∂y_i/∂x_k = ∑(∂y_i/∂u_j × ∂u_j/∂x_k)
```

**Concrete example:**
```
If L = v² and v = u + c
Then ∂L/∂u = (∂L/∂v) × (∂v/∂u) = 2v × 1
```

---

## 12. Vanishing and Exploding Gradients

### Problem 1: Vanishing Gradients

**What happens:**
- Small derivatives (e.g., Sigmoid max = 0.25) multiply repeatedly
- Gradients in early layers become **near-zero**
- Those layers **stop learning**

**Why it occurs:**
- Multiplying many small numbers
- Example: 0.1 × 0.1 × 0.1 = **0.001** (after only 3 layers!)

### Problem 2: Exploding Gradients

**What happens:**
- Large weights/derivatives cause gradients to grow exponentially
- Results in **huge, unstable weight updates**
- Training becomes chaotic

**Why it occurs:**
- Multiplying many large numbers
- Example: 3.0 × 3.0 × 3.0 = **27.0** (after only 3 layers!)

### Solution:
- **ReLU** activation helps prevent vanishing gradients
- **Gradient clipping** prevents exploding gradients

---

## 13. Invariance vs. Equivariance

### Invariance
**Definition:** Output stays the same when input transforms

**Formula:**
```
f[x] = f[t(x)]
```

**Example:**
- Image classifier says "Cat" regardless of where the cat appears in the image

### Equivariance
**Definition:** Output transforms the same way as the input

**Formula:**
```
f[t(x)] = t(f[x])
```

**Example:**
- If you shift the input image right, the feature map also shifts right

---

## 14. Convolutional Neural Networks (CNNs)

### 🔄 From MLP to CNN: The Spatial Revolution

**Problems with MLPs on Images:**
1. **Loss of spatial information:** Images are flattened into 1D vectors, destroying the 2D structure
2. **Extreme parameter explosion:** A 256×256 image = 65,536 inputs; one hidden layer = billions of weights!
3. **No translation invariance:** An MLP trained to recognize an object in the top-left corner fails when the object appears in the bottom-right
4. **Local feature blindness:** MLPs process all inputs globally and struggle to learn local patterns like edges or textures

**CNN Solution:**
- **Local connectivity:** Small kernels only process local regions, preserving spatial relationships
- **Parameter sharing:** Same kernel weights reused across the image → massive parameter reduction
- **Translation invariance:** Because the same kernel slides everywhere, the network recognizes features regardless of position
- **Result:** Networks can efficiently learn spatial patterns in images

---

## 14a. Convolution Operation

**Core idea:** Slide a learnable **kernel (filter)** across an image

**What happens:**
1. Element-wise multiply kernel with image patch
2. Sum all products
3. Move to next position

**Formula:**
```
(I * K)(i,j) = ∑∑ I(i+m, j+n) × K(m,n)
```

**Real example:**
- 3×3 kernel moving across 10×10 image to detect vertical lines

**Benefits:**
- Preserves spatial relationships
- Weight sharing reduces parameters
- Translation invariance

---

## 15. Convolution Parameters

### Three key hyperparameters:

| Parameter | What it does | Effect |
|-----------|-------------|--------|
| **Padding** | Add zeros to border | Maintains spatial size |
| **Stride** | Kernel shift size | Stride=2 downsamples by half |
| **Dilation** | Spread kernel weights | Increase receptive field without extra parameters |

**Output Size Formula:**
```
O = ⌊(I - K + 2P) / S⌋ + 1
```
where:
- I = input size
- K = kernel size
- P = padding
- S = stride

**Practical example:**
- Input: 227×227
- Kernel: 11×11
- Stride: 4
- Padding: 0
- **Output:** 55×55

**Practical example:**
- Input: 227×227
- Kernel: 11×11
- Stride: 4
- Padding: 0
- **Output:** 55×55

### 🔄 From Large Kernels to Stacked 3×3 Filters (VGG Era)

**Problem with Large Kernels:**
- Early CNN architectures (like AlexNet) used **11×11 or 7×7 kernels**
- Large kernels = **expensive computation** and **many parameters** for just one non-linear step
- Result: Limited network depth due to computational constraints

**VGG Insight:**
- **Multiple stacked 3×3 convolutions** achieve the same receptive field as larger kernels **with fewer parameters**
- Example: Three 3×3 layers emulate one 7×7 layer but with **more non-linear activation steps**
- **Benefits:**
  - More non-linearity → more expressive network
  - Fewer parameters → faster computation and less overfitting
  - Easier to train very deep networks
- **Result:** This principle became fundamental to modern CNN design

---

## 16. Receptive Field

**Definition:** The region of input image that a neuron "sees"

**Key insight:**
- **Shallow layers:** Small receptive field (local features: edges)
- **Deep layers:** Large receptive field (global features: objects)

**Efficiency trick:**
- Multiple stacked small kernels (two 3×3) emulate one large kernel (5×5) with fewer parameters

---

## 17. Separable Convolutions

### Spatial Separability
Split 2D kernel into two 1D kernels

**Example:**
- 3×3 kernel → 3×1 and 1×3 kernels
- Reduces multiplications significantly

### Depthwise Separability
Split into two steps:
1. **Depthwise:** Process each channel separately
2. **Pointwise:** 1×1 convolution to combine channels

**Efficiency comparison:**
- Standard 3×3 on 10×10 image: **576 multiplications**
- Spatially separable: **432 multiplications**
- **25% faster!**

---

## 18. Pooling (Downsampling)

**Purpose:**
- Reduce resolution of feature maps
- Decrease computation and memory
- Provide local invariance to small shifts

### Max Pooling (Most common)
Takes **highest value** in window

**Formula:**
```
P_out = max(F_in(m,n)) for m,n in window
```

**Example:**
```
Input window:  [3, 2]
               [0, 7]
Output: 7 (the maximum value)
```

### Mean Pooling
Takes **average value** in window

---

## 19. Batch Normalization (BN)

**Purpose:** Stabilize and accelerate deep network training

**What it does:**
- Standardizes the **activations within a mini-batch** to have zero mean and unit variance
- Reduces **internal covariate shift** (the constant change in input distributions to layers as weights update)

**How it helps:**
1. ✅ **Stabilizes training:** Prevents activations from becoming too large or too small
2. ✅ **Allows higher learning rates:** Networks can train faster without diverging
3. ✅ **Regularization effect:** Acts like a regularizer, reducing overfitting
4. ✅ **Less sensitive to weight initialization:** Networks don't depend as heavily on careful initialization

**Trade-off:**
- ⚠️ Depends on batch size (fails when batches are very small)
- ⚠️ Requires different treatment during training vs. inference

### 🔄 From Standard CNNs to Batch Normalization

**Problem:**
- Training deep networks is **slow and unstable**
- As weights in early layers change, the **distribution of inputs to later layers keeps shifting** (internal covariate shift)
- This forces subsequent layers to constantly adapt to new input distributions
- Result: Slow convergence, need for careful hyperparameter tuning

**BN Solution:**
- **Normalizes activations within each mini-batch** → consistent input distributions across training
- Early and deep layers can learn independently without worrying about input distribution shifts
- **Impact:** Networks train **much faster**, can use **higher learning rates**, and **generalize better**
- **Historical significance:** Enabled training of much deeper networks more reliably

---

## 20. Layer Normalization (LayerNorm)

**Purpose:** Normalize activations in a **batch-independent** way

**What it does:**
- Calculates statistics (mean and variance) **across features of a single sample** rather than across a batch
- Normalizes each sample independently

**Key advantages:**
1. ✅ **Batch-size independent:** Works with any batch size, even size 1
2. ✅ **Works with sequential data:** Effective for RNNs, Transformers, and variable-length sequences
3. ✅ **No train/inference difference:** Behavior is identical during training and inference
4. ✅ **Stable for online learning:** Can process samples one at a time

### 🔄 From Batch Normalization to Layer Normalization

**Problems with Batch Normalization:**
- **Highly dependent on batch size:** Very small batches (or batch size 1) give noisy statistics
- **Fails for sequence models:** RNNs and Transformers process sequences where lengths vary
- **Different behavior at train/test time:** Must maintain running statistics, making deployment trickier

**LayerNorm Solution:**
- **Normalizes each sample independently** across its features
- Works perfectly with any batch size, even single samples
- **Ideal for Transformers and sequence models** where batch normalization struggles
- **No train/inference difference:** Same computation during training and deployment
- **Result:** Became the standard normalization method for modern Transformers (BERT, GPT, etc.)

**When to use:**
- **Batch Norm:** CNN image models where large batches are standard
- **Layer Norm:** Transformers, RNNs, variable-batch scenarios

---

## 21. Residual Networks (ResNet)

**Purpose:** Enable training of **extremely deep networks** (100+ layers)

**Core innovation:** Skip connections (residual mappings)

**How it works:**
- Instead of learning the full mapping **F(x)**, learn the **residual** (difference): **F(x) = input + residual**
- Gradients can flow directly through skip connections, bypassing intermediate layers

**Formula:**
```
x_(l+1) = x_l + F(x_l)
```

### 🔄 From VGG to ResNet: Solving the Degradation Problem

**Problem - The Degradation Paradox:**
- Intuitively, deeper networks should perform better (more capacity, more parameters)
- In practice, **very deep networks (20+ layers) showed lower accuracy than shallower networks**
- Not due to overfitting, but because **deep networks failed to learn at all**
- **Root cause:** Vanishing gradients in extremely deep networks make learning the identity mapping difficult

**ResNet Solution:**
- **Skip connections** allow gradients to flow directly backward without passing through many layers
- Network learns the **residual (difference)** instead of the full mapping
- If layers are not useful, the residual can just be zero (identity mapping preserved)
- **Benefits:**
  1. ✅ **Immediate gradient flow:** Backpropagation reaches early layers directly
  2. ✅ **Identity shortcuts:** If a layer doesn't help, skip it
  3. ✅ **Enables extreme depth:** Successfully trained 152-layer, 1001-layer networks
  4. ✅ **Better accuracy:** ResNet-152 > ResNet-50 (unlike VGG where depth hurt accuracy)

**Impact:**
- **Solved the degradation problem** and enabled the "depth revolution"
- Opened the door to modern very-deep architectures (DenseNet, EfficientNet, etc.)
- **Became the foundation of modern deep learning** in computer vision

---

## 22. Transposed Convolutions & Upsampling

**Purpose:** Increase image resolution (opposite of standard convolution)

**How it works:**
- Maps single pixel to larger area
- Example: 1×1 → 3×3 region

**Output Height Formula:**
```
H_out = (H_in - 1) × stride - 2×padding + kernel_size
```

**Other upsampling methods:**
- Simple duplication (nearest neighbor)
- Bilinear interpolation

---

## 23. CNN Architecture Patterns

### Typical Basic CNN Workflow (Before ResNet):
```
Input Image
    ↓
Convolution (extract features) → Activation (non-linearity)
    ↓
Pooling (downsample) → reduce spatial dimensions
    ↓
[Repeat above 2-3 times]
    ↓
Flatten → Convert to 1D vector
    ↓
Fully Connected Layers (MLPs)
    ↓
Output (Classification)
```

### Key insight:
- **Early layers:** Learn simple features (edges, textures)
- **Middle layers:** Learn parts (wheels, eyes)
- **Deep layers:** Learn objects (cars, faces)

### Modern CNN Workflow (With ResNet Skip Connections):
```
Input
  ↓
┌─Convolution → ReLU → Batch Norm
│     ↓
└─→ [Skip Connection]
     ↓
   Add + ReLU
     ↓
┌─Convolution → ReLU → Batch Norm
│     ↓
└─→ [Skip Connection]
     ↓
   Add + ReLU
     ↓
[Repeat residual blocks]
     ↓
Global Average Pooling
     ↓
Fully Connected Layer
     ↓
Output (Classification)
```

---

## Summary Table: Key Concepts & Their Evolution

| Evolution Journey | Problem | Solution | Era |
|---------|---------|----------|------|
| **Perceptron → MLP** | Can't solve non-linear problems (XOR) | Hidden layers + activations | 1980s |
| **Sigmoid → ReLU** | Vanishing gradients in deep nets | Constant derivative preserves signal | 2011 |
| **Wide Nets → Deep Nets** | Parameter explosion, no hierarchies | Learn features hierarchically | 2012+ |
| **MLP → CNN** | Spatial info lost, param explosion | Local connectivity, weight sharing | 2012 |
| **Large Kernels → 3×3 Stacked** | Expensive computation in AlexNet | Receptive field equiv. with more nonlinearity | 2014 (VGG) |
| **Vanilla CNN → Batch Norm** | Training slow, internal covariate shift | Normalize activations per batch | 2015 |
| **Batch Norm → Layer Norm** | Fails with small batches, bad for sequences | Normalize per sample, per feature | 2016 |
| **VGG → ResNet** | Degradation: deeper = worse accuracy | Skip connections preserve gradient flow | 2015 |

---

## Quick Reference: Architectural Components

| Component | Purpose | Key Property |
|-----------|---------|---------------|
| **Convolution** | Extract local features | Weight sharing reduces parameters |
| **Activation (ReLU)** | Add non-linearity | Constant gradient prevents vanishing |
| **Batch Norm** | Stabilize training | Normalize activations within batch |
| **Pooling** | Reduce spatial size | Provide local invariance |
| **Skip Connection** | Enable deep networks | Gradients flow directly to early layers |
| **Layer Norm** | Sequence/Transformer norm | Batch-size independent |

---

## Evolution of Challenges Solved

| Generation | Architecture | Major Challenge Solved |
|-----------|-------------|----------------------|
| **1950-1970** | Perceptron | Linear classification |
| **1980s-1990** | MLP | Non-linear problems |
| **2012** | AlexNet (CNN) | Image classification at scale + GPU training |
| **2014** | VGG | Receptive fields with efficiency |
| **2015** | ResNet | Ultra-deep networks (100+ layers) |
| **2015** | Batch Norm | Faster, more stable training |
| **2017** | Transformer | Sequence modeling without RNNs |

---

**Note:** All formulas and examples aim to build intuition for how neural networks learn to solve real-world problems in computer vision.
