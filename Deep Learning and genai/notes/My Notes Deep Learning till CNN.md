# 🎓 Deep Learning & Computer Vision: Complete Guide

## 📚 Quick Navigation

1. [Stage 1: Vision Foundation](#stage-1)  2. [Stage 2: Neural Networks](#stage-2)  3. [Stage 3: Training](#stage-3)
4. [Stage 4: Backpropagation](#stage-4)  5. [Stage 5: CNNs](#stage-5)  6. [Advanced Architectures](#advanced)
7. [Practice Problems & Solutions](#practice)

---

## Stage 1: Vision Fundamentals {#stage-1}

**Vision Tasks:** Classification (is it a cat?), Detection (where is it?), Segmentation (which pixels?)

**Why Vision is Hard:** Viewpoint variation, illumination changes, occlusion, intra-class variation

**Formula:** $f(x) = y$ where $x$ = image, $f$ = model, $y$ = prediction

---

## Stage 2: Neural Network Basics {#stage-2}

### Perceptron
$y = 1$ if $\sum w_i x_i \geq T$, else $0$

**Example:** AND gate with weights [1, 1], threshold 1.5
- Input [1,1] → sum=2 → y=1 ✓
- Input [1,0] → sum=1 → y=0 ✓

### Activation Functions

| Function | Formula | Derivative | Use | Problem |
|---|---|---|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ ≈ 0.25 | Binary prob | Vanishes over layers |
| Tanh | $\tanh(z)$ | Max ≈ 1.0 | Normalized output | Still vanishes |
| ReLU | $f(x) = \max(0,x)$ | 1 if x>0 | Default | Dying ReLU ✗ |
| Leaky ReLU | $f(x) = x$ if x>0, else 0.01x | 1 or 0.01 | Better ReLU | Slower |
| Swish | $x \cdot \sigma(x)$ | Smooth | Modern | Slightly slower |

**Key Insight:** ReLU gradient = 1, preventing vanishing gradients (ReLU > Sigmoid/Tanh)

### Softmax for Multi-Class
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Example:** Scores [2.0, 1.0, 0.1] → Probabilities [66%, 24%, 10%] (sums to 100%)

---

## Stage 3: Training & Optimization {#stage-3}

### Cross-Entropy Loss
$$L = -\sum y_i \log f_i$$

**Example:** Predict cat=0.66, actual cat=1 → Loss = -log(0.66) ≈ 0.415

**Interpretation:** Lower loss = better predictions. Loss=0 when prediction is certain

### SGD (Gradient Descent)
$$w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}$$

**Example:** $w=0.5$, gradient=2.0, η=0.1 → $w_{\text{new}} = 0.5 - 0.2 = 0.3$

| Learning Rate | Effect |
|---|---|
| 0.01 | Slow, stable |
| 0.1 | Ideal, balanced ✓ |
| 0.5 | Fast, risky |
| 1.0 | Diverges ✗ |

---

## Stage 4: Backpropagation {#stage-4}

**Chain Rule:** $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$

**Concept:** Trace error backward through network layers to update weights

**Forward:** Input → Layer1 → Layer2 → Loss
**Backward:** Loss ← derivative flows backward, updating each layer's weights

**Key benefit:** Calculate exact gradient for EACH parameter (vs random trial-and-error)

---

## Stage 5: Convolutional Neural Networks {#stage-5}

### Convolution Operation
$$\text{Output Size} = \left\lfloor \frac{\text{Input} - \text{Kernel} + 2(\text{Padding})}{\text{Stride}} \right\rfloor + 1$$

**Example:** 10×10 input, 3×3 kernel, stride=1, padding=0 → 8×8 output

### Parameter Sharing
- Normal: 224×224 image × 100 neurons = 5M+ parameters ✗
- CNN: 1 kernel × 9 weights, reuse everywhere = 9 parameters ✓
- **Reduction: 99.98% fewer parameters!**

### Pooling Operations

**4×4 Grid Example:**
```
[ 1  3 | 2  15]    Max Pooling 2×2:   [9  15]
[ 5  9 | 0   4] →                     [4   8]
[--+--|--+---]
[ 2  1 | 8   3]    Window 1: max(1,3,5,9)=9
[ 0  4 | 5   2]    Window 4: max(8,3,5,2)=8
```

| Pooling Type | What it does | When to use |
|---|---|---|
| Max | Takes strongest signal | Most CNNs (default) |
| Average | Takes mean | Smoothing |
| Min | Takes weakest | Rarely |

**Benefits:** 75% spatial reduction, translation invariance, robustness

### Edge Detection Kernels

**Vertical edges (Sobel):**
```
[-1  0  +1]
[-2  0  +2]  (Dark on left, white on right)
[-1  0  +1]
```

**Horizontal edges:**
```
[-1  -2  -1]
[ 0   0   0]  (Dark on top, white on bottom)
[+1  +2  +1]
```

---

## Advanced Architectures {#advanced}

### Inception (GoogleNet)
- **Core idea:** Multi-scale features using 1×1, 3×3, 5×5, pooling in parallel
- **1×1 bottlenecks:** Reduce dimensions before expensive ops (80% parameter saving)
- **Key stats:** 9 modules, 22 layers, 5-7M params vs VGG's 138M ✓
- **DOG kernel sizes:** 5×5 for edges, 7×7 for corners, multiple scales for features

### ResNet
- **Skip connections:** $H(x) = F(x) + x$ (learn residuals, not raw outputs)
- **Why it works:** Preserves gradient flow, enables very deep networks (50-152 layers)
- **Building blocks:**
  - Basic: Conv3×3 → Conv3×3 + skip
  - Bottleneck: Conv1×1 (reduce) → Conv3×3 → Conv1×1 (expand) + skip
- **Architecture:** Input → Layers of increasing depth → Global Avg Pool → FC → Output

| Model | Depth | Parameters | ImageNet Acc |
|---|---|---|---|
| ResNet-50 | 50 | 25M | 76.1% |
| ResNet-101 | 101 | 44M | 77.4% |
| ResNet-152 | 152 | 60M | 77.6% |

### DenseNet
- **Dense connections:** Every layer connects to ALL previous layers (concatenate)
- **Why efficient:** Feature reuse, fewer parameters needed
- **Breakthrough:** DenseNet-121 (121 layers, 7M params) beats ResNet-101 (78.2% vs 77.4%) ✓✓

| Architecture | Depth | Parameters | ImageNet Acc |
|---|---|---|---|
| ResNet-101 | 101 | 44M | 77.4% |
| DenseNet-121 | 121 | 7M | 78.2% |

**DenseNet wins:** 6× fewer params, higher accuracy!

---

## Practice Problems & Solutions {#practice}

### Q1: Feedforward Networks
**Q:** What ensures info flows Input→Output without loops?
**A:** Directed Acyclic Graph (DAG) structure. Each layer connects only to next layer ✓

### Q2: Bias Role
**Q:** Main role of bias?
**A:** Shifts activation threshold. Allows learning patterns not centered at zero ✓

### Q3: Vanishing Gradient
**Q:** Most severe with which activation?
**A:** Sigmoid (max derivative ≈ 0.25). After 100 layers: gradient ≈ 10⁻⁶⁰

| Activation | Issue |
|---|---|
| Sigmoid | Severe vanishing |
| Tanh | Moderate vanishing |
| ReLU | None (gradient=1) ✓ |

### Q4: Parameter Sharing in CNNs
**Q:** Primary benefit?
**A:** Reduced learnable parameters (99.98% fewer vs fully connected) ✓

### Q5: Translation Invariance
**Q:** Which layer helps?
**A:** Pooling (Max Pool especially) - shifts within window don't change max value ✓

### Q6: Stride Effect
**Q:** Effect of stride > 1?
**A:** Spatial downsampling. Stride=2 reduces dimensions by 2× (4×4→2×2)

### Q7: Dilated Convolutions
**Q:** Primary use?
**A:** Increase receptive field without adding parameters. Dilation=2 makes 3×3 kernel see 5×5 area ✓

### Q8: Dying ReLU
**Q:** Most prone activation?
**A:** ReLU (neuron outputs 0, gradient=0, never recovers). Solution: Leaky ReLU, ELU

### Q9: Softmax Usage
**Q:** Why use in output layer?
**A:** Outputs valid probability distribution (sums to 1), interpretable as multi-class probabilities ✓

### Q10: Swish Activation
**Q:** Formula?
**A:** $\text{Swish}(x) = x \cdot \sigma(x)$ - smooth self-gating, better than ReLU ✓

### Q11: Pooling Exercise
**Given:** 4×4 grid
```
[ 1  3 | 2 15]
[ 5  9 | 0  4]
------+------
[ 2  1 | 8  3]
[ 0  4 | 5  2]
```
**2×2 Max Pooling:** 
- Window1: max(1,3,5,9) = **9**
- Window2: max(2,15,0,4) = **15**
- Window3: max(2,1,0,4) = **4**
- Window4: max(8,3,5,2) = **8**
- **Output:** [9 15; 4 8]

---

## Key Takeaways

✅ **Vision:** Classification, Detection, Segmentation
✅ **Neurons:** Perceptrons + Activation Functions = Non-linearity
✅ **Training:** Loss function + Gradient Descent = Learning
✅ **CNNs:** Kernels + Pooling + Parameter Sharing = Spatial efficiency
✅ **Deep Networks:** Skip connections (ResNet) or dense connections (DenseNet) enable very deep nets
✅ **Modern Tricks:** Batch Norm, Dropout, Learning rate scheduling, Data augmentation

**Golden Rule:** Use ReLU (not Sigmoid), add skip connections (deeper), batch normalize (stable), use appropriate architecture (Inception/ResNet/DenseNet)

---

## Formula Quick Reference

| Concept | Formula |
|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| Softmax | $\frac{e^{z_i}}{\sum e^{z_j}}$ |
| Cross-Entropy | $L = -\sum y_i \log f_i$ |
| Gradient Descent | $w_{\text{new}} = w_{\text{old}} - \eta \nabla L$ |
| Conv Output Size | $\lfloor\frac{I-K+2P}{S}\rfloor + 1$ |
| ResNet | $H(x) = F(x) + x$ |
| ReLU | $\max(0, x)$ |
| Swish | $x \cdot \sigma(x)$ |

---

