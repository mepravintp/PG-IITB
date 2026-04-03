# Convex Optimization

## 1️⃣ Convex Sets
**Definition:** Line segment between any two points stays in X.
$$\alpha x + (1-\alpha)y \in X \quad \forall x,y \in X, \alpha \in [0,1]$$

**Convex sets preserved by:** Intersection, Cartesian product, Set summation

**Example - Polyhedral sets:** $\{x \in \mathbb{R}^n : Ax \leq b\}$

---

## 2️⃣ Convex Functions
**Definition:**
$$f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y)$$

**Tests:**
- **1st order:** $f(y) \geq f(x) + \nabla f(x)^T (y-x)$
- **2nd order:** $\nabla^2 f(x) \succeq 0$ (Hessian positive semidefinite)

**Key facts:** Jensen's inequality, norms are convex, -f concave

---

## 3️⃣ Constrained Optimization

**Primal:** $\min_{x \in X} f(x) \quad \text{s.t.} \quad Ax \leq b$

**Lagrangian:** $L(x, \alpha) = f(x) + \alpha^T(Ax - b)$

**Dual Function:** $F(\alpha) = \inf_{x \in X} L(x, \alpha)$

**Dual:** $\max_{\alpha \geq 0} F(\alpha)$

**Duality:** Weak: $d^* \leq p^*$ | Strong: $p^* = d^*$ (Slater)

---

## 4️⃣ KKT Conditions
Optimal point (x*, α*) satisfies:
1. **Stationarity:** $\nabla_x L(x^*, \alpha^*) = 0$
2. **Primal Feasibility:** $Ax^* \leq b$
3. **Complementary Slackness:** $\alpha_i (a_i x^* - b_i) = 0$

---

## 5️⃣ Example

**Problem:** Min $x_1^2 + 4x_2^2$ s.t. $x_1 + x_2 \leq -1$

**Lagrangian:** $L = x_1^2 + 4x_2^2 + \alpha(x_1 + x_2 + 1)$

**From ∂L/∂x = 0:**
$$x_1^* = -\frac{\alpha}{2}, \quad x_2^* = -\frac{\alpha}{8}$$

**Dual objective:** $F(\alpha) = \alpha(1 - \frac{5\alpha}{16})$

**Optimize:** $\alpha^* = \frac{8}{5}$

**Solution:** $x_1^* = -\frac{4}{5}, \quad x_2^* = -\frac{1}{5}$
