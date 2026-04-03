# Convex Optimization (1-Page Cheat Sheet)

**1. Convex Sets**
- Line segment between any two points stays in X: $\alpha x + (1-\alpha)y \in X$
- Preserved by: Intersection, Cartesian product, Set summation
- Example: $\{x : Ax \leq b\}$ (polyhedra)

**2. Convex Functions**
- Definition: $f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y)$
- 1st order test: $f(y) \geq f(x) + \nabla f(x)^T(y-x)$
- 2nd order test: $\nabla^2 f(x) \succeq 0$

**3. Optimization Problem**
- Primal: $\min_x f(x)$ s.t. $Ax \leq b$
- Lagrangian: $L(x,\alpha) = f(x) + \alpha^T(Ax-b)$
- Dual: $\max_{\alpha \geq 0} \inf_x L(x,\alpha)$
- **Weak Duality:** $d^* \leq p^*$ | **Strong:** $p^*=d^*$ (Slater)

**4. KKT Conditions (Optimality)**
- Stationarity: $\nabla_x L = 0$
- Primal Feasibility: $Ax^* \leq b$
- Complementary Slackness: $\alpha_i(a_i x^* - b_i) = 0$

**5. Example:** Min $x_1^2 + 4x_2^2$ s.t. $x_1+x_2 \leq -1$
- From $\nabla_x L=0$: $x_1^*=-\frac{\alpha}{2}$, $x_2^*=-\frac{\alpha}{8}$
- Dual: $F(\alpha)=\alpha(1-\frac{5\alpha}{16})$
- Optimal: $\alpha^*=\frac{8}{5}$ → $x_1^*=-\frac{4}{5}$, $x_2^*=-\frac{1}{5}$
