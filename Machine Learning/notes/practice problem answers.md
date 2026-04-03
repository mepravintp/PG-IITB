# Practice Problem Answers

---

## 📌 Topic 1: Bias-Variance Trade-off & Model Selection

### ❓ Q1: What happens to optimal least-squares loss if you scale a feature by 2?

**Answer:** The optimal loss does **NOT** change.

**Explanation:** 
- When feature x_j is scaled (e.g., 2x_j), weights adjust inversely: w'_j = w_j/2
- Predictions stay the same → residuals unchanged → loss unchanged
- Only weights are rescaled

---

### ❓ Q2: When is flexible model better/worse than inflexible?

**Answer:**
| Scenario | Better |
|----------|--------|
| Large n, small p | Flexible ✓ |
| Large p, small n | Inflexible ✓ |
| Highly non-linear | Flexible ✓ |
| Very high noise | Inflexible ✓ |

**Explanation:** Flexible models have lower bias but higher variance. They excel with large datasets but overfit with limited data or noise.

---

### ❓ Q3: What are trade-offs between flexible and inflexible models?

**Answer:** 
- **Flexible:** Low bias, high variance, needs more data, less interpretable
- **Inflexible:** High bias, low variance, simpler, more interpretable

**Explanation:** Flexible models (e.g., trees) capture complex patterns but risk overfitting. Inflexible models (e.g., linear regression) are stable but may miss relationships.

---

### ❓ Q4: Differences between Parametric and Non-Parametric models?

**Answer:**
- **Parametric:** Fixed functional form (e.g., linear). Simple, fast, needs less data
- **Non-Parametric:** No fixed form, grows with data (e.g., KNN, trees). Flexible but expensive

**Explanation:** Parametric models fail if assumption is wrong. Non-parametric models are flexible but computationally costly and prone to overfitting.

---

### ❓ Q5: How do training/test errors change with model complexity?

**Answer:**
- **Training error:** Decreases continuously
- **Test error:** U-shaped curve (decreases then increases)

**Explanation:** More complex models fit training data better but overfit. Test error increases after the optimal point.

---

## 📌 Topic 2: Linear and Polynomial Regression

### ❓ Q1: Best-fit line for points (1,2), (2,5), (3,4)?

**Answer:** y = x + 5/3  (w = 1, b = 5/3)

**Explanation:**
- Mean: x̄ = 2, ȳ = 11/3
- Slope: Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)² = 2/2 = 1
- Intercept: b = ȳ - w·x̄ = 11/3 - 1(2) = 5/3

---

### ❓ Q2: Effect of high-leverage outlier on regression line?

**Answer:** 
- Slope changes **significantly**
- R² **decreases**

**Explanation:** A high-leverage point far in x-direction pulls the line toward itself, altering slope and worsening fit quality.

---

### ❓ Q3: How to choose optimal polynomial degree?

**Answer:** Choose degree where **validation/test error is minimum**.

**Explanation:**
- Training error always decreases (more flexibility)
- Validation error: decreases then increases (U-shape)
- Optimal = bottom of U-curve (sweet spot before overfitting)