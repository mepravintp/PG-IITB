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
$$f^*(x) = \text{sign}\left( \sum_{i=1}^{n} \alpha_i^* y_i \langle x_i, x \rangle + b^* \right)$$

**Efficiency:** Only support vectors (non-zero $\alpha_i$) contribute to prediction