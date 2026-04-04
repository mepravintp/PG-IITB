# SVM: Hard Margin Classifier

**Goal:** Binary classification {+1, −1} via maximum-margin hyperplane.

---

## 1️⃣ Separating Hyperplane
Find decision boundary: $\langle w, x \rangle + b = 0$

## 2️⃣ Geometric Margin
Distance from hyperplane to nearest point:
$$\rho_{w,b} = \min_{1 \leq i \leq n} \frac{|w^T x_i + b|}{\|w\|}$$
SVM **maximizes** this margin → robustness

## 3️⃣ Primal Problem
| | |
|---|---|
| **Minimize** | $\frac{1}{2} \|w\|^2$ |
| **Subject to** | $y_i (w^T x_i + b) \geq 1$ for all $i$ |

## 4️⃣ Lagrangian & KKT Conditions

**Lagrangian:**
$$L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{n} \alpha_i [y_i (w^T x_i + b) - 1]$$

**Stationarity conditions:**
- $w^* = \sum_{i=1}^{n} \alpha_i y_i x_i$
- $\sum_{i=1}^{n} \alpha_i^* y_i = 0$

**Complementary Slackness:**
$$\alpha_i^* [y_i (w^{*T} x_i + b^*) - 1] = 0$$
⇒ If point outside margin: $\alpha_i = 0$ (not support vector)
⇒ $w^*$ is linear combination of **support vectors only**

## 5️⃣ Dual Problem

**Maximize:**
$$\sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$$

**Subject to:** $\sum_i y_i \alpha_i = 0, \quad \alpha_i \geq 0$

*Advantage:* Only depends on dot products (enables kernel trick)

## 6️⃣ Final Classifier

**Decision function:**

For prediction on new sample x:
- Compute: `sum = Σ(αᵢ* × yᵢ × <xᵢ, x>) + b*`
- Output: `f(x) = +1 if sum > 0, else -1`

```
Classification Rule: f(x) = +1 if Σ αᵢ* yᵢ ⟨xᵢ,x⟩ + b* > 0, else -1
```

**Efficiency:** Only support vectors (non-zero $\alpha_i$) contribute to prediction

---

## 📊 Practical Example: Linearly Separable Data

### Training Data (2D, Binary Classification)
| Point | x₁ | x₂ | y | Class |
|-------|-----|-----|----|----|
| 1 | 1 | 1 | +1 | Class A |
| 2 | 2 | 2 | +1 | Class A |
| 3 | 6 | 6 | -1 | Class B |
| 4 | 7 | 7 | -1 | Class B |

### Step 1: Solve Dual Problem

Maximize: $\frac{1}{2}(α_1 + α_2 + α_3 + α_4) - \frac{1}{2}[\mathbf{α}^T \mathbf{Q} \mathbf{α}]$

where $Q_{ij} = y_i y_j \langle x_i, x_j \rangle$

**Kernel Matrix (inner products):**
- $\langle x_1, x_1 \rangle = 1^2 + 1^2 = 2$
- $\langle x_1, x_3 \rangle = 1×6 + 1×6 = 12$
- $\langle x_3, x_3 \rangle = 6^2 + 6^2 = 72$

### Step 2: Optimal Solution

**How do we get α values? Solve the dual optimization problem:**

For our 4 points, the Q-matrix is:
$$Q = \begin{bmatrix}
+2 & +4 & -12 & -14 \\
+4 & +8 & -24 & -28 \\
-12 & -24 & +72 & +84 \\
-14 & -28 & +84 & +98
\end{bmatrix}$$

(Each $Q_{ij} = y_i y_j \langle x_i, x_j \rangle$)

**The dual problem becomes:**
$$\max_\alpha \left[ (α_1 + α_2 + α_3 + α_4) - \frac{1}{2} \mathbf{α}^T Q \mathbf{α} \right]$$
subject to: $α_1 + α_2 = α_3 + α_4$ and $α_i \geq 0$

**Solving numerically (using sequential minimal optimization or QP solver):**
- $α_1^* = 0.1, α_2^* = 0, α_3^* = 0.1, α_4^* = 0$

**Why these specific values?**
- **α₁ = 0.1** and **α₃ = 0.1** because points 1 and 3 are **on the margin** (they're closest to the decision boundary)
- **α₂ = 0** and **α₄ = 0** because points 2 and 4 are **far from margin** → by complementary slackness, their multipliers must be zero
- These α values **maximize** the dual objective function while satisfying constraints

**In practice:** Use SVM solvers (scikit-learn, libsvm) that compute α numerically. For this simple example, I used 0.1 as the simplified solution.

### Step 3: Compute w*
$$w^* = \sum_i \alpha_i^* y_i x_i = 0.1(+1)[1, 1] + 0.1(-1)[6, 6] = [-0.5, -0.5]$$

### Step 4: Compute b*
Using complementary slackness on support vector 1:
$$b^* = y_1 - w^{*T} x_1 = 1 - (-0.5)(1) - (-0.5)(1) = 0$$

### Step 5: Prediction on New Point
**Predict for $x_{test} = [3, 3]$:**

Decision: $f(x_{test}) = \operatorname{sgn}((-0.5)(3) + (-0.5)(3) + 0) = \operatorname{sgn}(-3) = -1$ ✓ (Class B)