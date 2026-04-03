# Machine Learning Exam Notes (Lectures 1-8)

A comprehensive study guide covering key concepts, formulas, and examples from your ML course.

---

## Lecture 1: Introduction to Machine Learning

### What is Machine Learning?

**Definition:** Learn patterns from data without explicit programming.

**Three Components:** Data, Model, Learning Algorithm

### ML Taxonomy (This Course = Supervised Learning)

- **Supervised Learning:** Labeled data $(x, y)$ pairs
  - Regression: Continuous output
  - Classification: Categorical output
- **Unsupervised Learning:** Unlabeled data $\{x\}$
  - Clustering, Dimensionality Reduction
- **Reinforcement Learning:** Learning through environment interaction

### Supervised Learning

**Definition:** Learn from labeled training data $\{(x_i, y_i)\}_{i=1}^{n}$

**Two Main Tasks:**

| Task | Goal | Example |
|------|------|---------|
| **Prediction** | Predict $y$ for new $x$ | House price forecasting |
| **Inference** | Understand relationships | Which features matter most? |

**Subtasks:**
- **Regression:** $y$ is continuous (price, temperature)
- **Classification:** $y$ is categorical (spam/not-spam, cat/dog)

### Unsupervised Learning

Works with unlabeled data $\{x_i\}$ only.

**Main Tasks:** Clustering (group similar), Dimensionality Reduction (reduce features)

### Training Data Essentials

**Structure:** $\{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$

- $x_i$ = features/inputs
- $y_i$ = label/target output

**Split Data:**
- Training: 70-80% (learn model)
- Validation: 10-15% (tune parameters)
- Test: 10-15% (evaluate final performance)

**Why?** Prevent overfitting; test error shows true generalization

**Data Quality Matters:**
- Sufficient quantity ($n > p$)
- Accurate labels
- Representative of population
- No systematic bias

---

## Lecture 2: Linear Algebra Fundamentals

### Inner (Dot) Product

For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:
$$\langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^T \mathbf{b}$$

**Example:** If $\mathbf{a} = [1, 2]$ and $\mathbf{b} = [3, 4]$, then:
$$\langle \mathbf{a}, \mathbf{b} \rangle = (1 \times 3) + (2 \times 4) = 11$$

### Vector Norm (Magnitude)

$$\|\mathbf{a}\| = \sqrt{\langle \mathbf{a}, \mathbf{a} \rangle} = \sqrt{\sum_i a_i^2}$$

**Example:** For vector $\mathbf{a} = [3, 4]$:
$$\|\mathbf{a}\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$

### Distance Between Vectors

$$\text{dist}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|$$

**Example:** The distance between $\mathbf{a} = [0, 0]$ and $\mathbf{b} = [3, 4]$ is:
$$\text{dist}(\mathbf{a}, \mathbf{b}) = \sqrt{(-3)^2 + (-4)^2} = 5$$

### Projection of Vector b onto Vector a

$$\text{proj}_{\mathbf{a}} \mathbf{b} = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\|\mathbf{a}\|^2} \mathbf{a}$$

**Example:** Projecting vector $\mathbf{b}$ onto $\mathbf{a}$ finds the component of $\mathbf{b}$ that lies in the direction of $\mathbf{a}$.

### Linear Independence & Dependence

A collection of vectors is **linearly dependent** if:
$$\sum \beta_i \mathbf{a}_i = \mathbf{0} \text{ for } \beta \text{ not all zero}$$

**Example:** If $\mathbf{c} = \beta_1 \mathbf{a} + \beta_2 \mathbf{b}$, then $\{\mathbf{a}, \mathbf{b}, \mathbf{c}\}$ are linearly dependent.

### Matrix Rank

**Definition:** The rank of a matrix is the **maximum number of linearly independent rows (or columns)**.

$$\text{rank}(A) = r, \quad \text{where } 0 \leq r \leq \min(m, n)$$

for an $m \times n$ matrix.

#### Methods to Calculate Rank

**Method 1: Row Reduction (Gaussian Elimination)**

Reduce the matrix to **Row Echelon Form (REF)** or **Reduced Row Echelon Form (RREF)**. The rank equals the number of non-zero rows.

**Example:**
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{bmatrix}$$

After row reduction:
$$\text{REF} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

Number of non-zero rows = 1, so $\text{rank}(A) = 1$.

**Method 2: Singular Value Decomposition (SVD)**

$$A = U \Sigma V^T$$

The rank equals the number of non-zero singular values in $\Sigma$.

**Method 3: Determinant (for Square Matrices)**

- If $\det(A) \neq 0$, then $\text{rank}(A) = n$ (full rank)
- If $\det(A) = 0$, then $\text{rank}(A) < n$ (rank deficient)

**Example:** 
$$\det\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix} = 1 \cdot 6 - 2 \cdot 3 = 0 \Rightarrow \text{rank} < 2$$

#### Python Implementation

```python
import numpy as np

A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]])

rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")  # Output: Rank: 1
```

#### Importance in Regression

**For OLS to have a unique solution:**
$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

The matrix $(X^T X)$ must be invertible, which requires:
$$\text{rank}(X) = p + 1 \quad \text{(full column rank)}$$

**Key Points:**
- If $\text{rank}(X) < p + 1$: Features are linearly dependent → $(X^T X)$ is singular → **no unique OLS solution**
- If $\text{rank}(X) = p + 1$: Features are linearly independent → $(X^T X)$ is invertible → **unique OLS solution exists**

---

## Lecture 4: Probability & Statistics Fundamentals

### Probability Without Replacement

**Concept:** When drawing multiple items from a set without replacement, each draw affects subsequent probabilities.

**Formula:**
$$P(\text{Event 1 and Event 2}) = P(\text{Event 1}) \times P(\text{Event 2 | Event 1})$$

**Example - Drawing Balls:**

A bag contains 5 red and 3 blue balls. What's the probability of drawing 2 red balls in a row?

$$P(\text{2 red}) = \frac{5}{8} \times \frac{4}{7} = \frac{20}{56} = \frac{5}{14} \approx 0.357$$

**Explanation:**
- First draw: 5 red balls out of 8 total
- Second draw: 4 red balls remain out of 7 total (since one red was removed)

### Binomial Distribution

**Use Case:** Modeling number of successes in a fixed number of independent trials.

**Parameters:** $X \sim \text{Binomial}(n, p)$
- $n$ = number of trials
- $p$ = probability of success on each trial

**Probability Mass Function:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Expected Value:**
$$E[X] = np$$

**Variance:**
$$\operatorname{Var}(X) = np(1-p)$$

**Example:** 

$X \sim \text{Binomial}(20, 0.3)$ (20 trials, probability 0.3)

$$E[X] = 20 \times 0.3 = 6$$

$$\operatorname{Var}(X) = 20 \times 0.3 \times 0.7 = 4.2$$

**Interpretation:** On average, expect 6 successes with a variance of 4.2.

### Exponential Distribution

**Use Case:** Modeling time until an event occurs (e.g., waiting times, lifetimes).

**Parameter:** $\lambda$ = rate parameter (average number of events per unit time)

**Cumulative Distribution Function (CDF):**
$$P(T \leq t) = 1 - e^{-\lambda t}$$

**Probability Density Function (PDF):**
$$f(t) = \lambda e^{-\lambda t}, \quad t \geq 0$$

**Expected Value:**
$$E[T] = \frac{1}{\lambda}$$

**Variance:**
$$\operatorname{Var}(T) = \frac{1}{\lambda^2}$$

**Example:** 

Customers arrive at a rate of $\lambda = 3$ per hour. What's the probability that the next customer arrives within 15 minutes (0.25 hours)?

$$P(T \leq 0.25) = 1 - e^{-3 \times 0.25} = 1 - e^{-0.75} \approx 1 - 0.472 = 0.528$$

**Interpretation:** There's a 52.8% chance the next customer arrives within 15 minutes.

### Expected Value and Variance of Portfolio

**Concept:** Combining multiple risky investments with assigned weights.

**Portfolio Return:**
$$E[P] = \sum_{i=1}^{n} w_i E[X_i]$$

where $w_i$ = weight of asset $i$, $E[X_i]$ = expected return of asset $i$.

**Portfolio Variance (Independent Assets):**
$$\operatorname{Var}(P) = \sum_{i=1}^{n} w_i^2 \operatorname{Var}(X_i)$$

**Example:**

Two independent investments with equal weights ($w_1 = w_2 = 0.5$):
- Investment 1: Expected return 8%, Variance 9
- Investment 2: Expected return 6%, Variance 4

**Portfolio Expected Return:**
$$E[P] = 0.5 \times 0.08 + 0.5 \times 0.06 = 0.07 \quad (7\%)$$

**Portfolio Variance:**
$$\operatorname{Var}(P) = (0.5)^2 \times 9 + (0.5)^2 \times 4 = 0.25 \times 9 + 0.25 \times 4 = 2.25 + 1 = 3.25$$

**Key Insight:** Diversification reduces risk through weighted variance combination.

### Bayes' Theorem

**Formula:**
$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**Components:**
- $P(A)$ = **Prior** probability of A
- $P(B|A)$ = **Likelihood** of B given A
- $P(B)$ = **Marginal** probability of B
- $P(A|B)$ = **Posterior** probability of A given B

**Law of Total Probability (for denominator):**
$$P(B) = P(B|A) P(A) + P(B|\neg A) P(\neg A)$$

**Example - Gene Test with Disease:**

Prior: 2% of population has the gene  
Test accuracy: Sensitivity = 98%, Specificity = 98%

Question: If test is positive, what's probability the person has the gene?

**Given:**
- $P(\text{Gene}) = 0.02$, $P(\text{No Gene}) = 0.98$
- Sensitivity = $P(+|\text{Gene}) = 0.98$
- Specificity = $P(-|\text{No Gene}) = 0.98$ → $P(+|\text{No Gene}) = 0.02$

**Calculate Marginal Probability of Positive Test:**
$$P(+) = P(+|\text{Gene})P(\text{Gene}) + P(+|\text{No Gene})P(\text{No Gene})$$
$$= 0.98 \times 0.02 + 0.02 \times 0.98 = 0.0196 + 0.0196 = 0.0392$$

**Apply Bayes' Theorem:**
$$P(\text{Gene}|+) = \frac{P(+|\text{Gene}) P(\text{Gene})}{P(+)} = \frac{0.98 \times 0.02}{0.0392} = \frac{0.0196}{0.0392} = 0.5$$

**Interpretation:** Despite high test accuracy (98%), a positive result only gives 50% confidence of having the gene. This is because the disease is rare (only 2% base rate).

---

## Lectures 3 & 5: OLS Regression & Interpretation

### Linear Regression Model

$$f(x) = \hat{\beta}_0 + \sum_{i=1}^{p} \hat{\beta}_i x^{(i)}$$

**Example:** Predicting Sales based on three features ($p=3$): TV, Radio, and Newspaper advertising budgets.

### Residual Sum of Squares (RSS)

$$\text{RSS}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|Y - X\beta\|^2$$

**Example:** In the Advertising dataset, RSS measures the total squared difference between actual sales and predicted sales.

### Closed-Form OLS Solution

$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

**Example:** This formula calculates the optimal coefficients ($\beta$) that minimize the error, provided the feature matrix $X$ has full column rank.

**Key Assumption:** The matrix $X^T X$ must be invertible (columns of $X$ are linearly independent).

### The Hat Matrix

$$H = X(X^T X)^{-1} X^T$$

$$\hat{Y} = HY$$

**Example:** The Hat Matrix is a geometric tool that projects actual observed outputs ($Y$) onto the space spanned by features to find best-fitting predictions ($\hat{Y}$).

---

## Lecture 6: Model Evaluation

### Total Sum of Squares (TSS)

$$\text{TSS} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**Meaning:** Captures total variability in the output variable before considering any features.

**Example:** How much sales figures vary naturally in the dataset.

### Residual Sum of Squares (RSS)

$$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Meaning:** Captures variability that the model fails to explain (mismatch between actual and predicted values).

### Model Sum of Squares (MSS)

$$\text{MSS} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$$

**Key Identity:**
$$\text{TSS} = \text{MSS} + \text{RSS}$$

### R-Squared (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{MSS}}{\text{TSS}} \quad \in [0,1]$$

**Interpretation:** Proportion of variance explained by the model.

**Example:** If $R^2 = 0.968$, then 96.8% of the variability in sales is explained by the model.

### Unbiased Estimate of Noise Variance

$$\hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1}$$

**Meaning:** Average squared error per observation, adjusted for degrees of freedom.

**Example:** Provides an estimate of the variance of random error ($\epsilon$) in the data.

### Standard Error of a Coefficient

$$\text{SE}(\hat{\beta}_i) = \sqrt{\hat{\sigma}^2 (X^T X)^{-1}_{ii}}$$

**Meaning:** Quantifies uncertainty in each regression coefficient estimate.

**Example:** In advertising data, if $\text{SE}(\hat{\beta}_{\text{TV}}) = 0.0014$, then the TV coefficient estimate is reliable.

### Confidence Interval for Coefficient

$$\hat{\beta}_i \pm t_{\alpha/2, n-p-1} \cdot \text{SE}(\hat{\beta}_i)$$

**Example:** 95% confidence interval for TV coefficient: $[\hat{\beta}_{\text{TV}} - 1.96 \cdot \text{SE}, \hat{\beta}_{\text{TV}} + 1.96 \cdot \text{SE}]$

### Hypothesis Test Statistic (t-statistic)

$$t = \frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)} \sim t_{n-p-1}$$

**Testing:** $H_0: \beta_i = 0$ vs $H_1: \beta_i \neq 0$

**Example:** To test if Newspaper ads are useful, test $H_0: \beta_{\text{newspaper}} = 0$.

- If p-value is small (e.g., $< 0.05$), reject $H_0$ → feature is significant
- If p-value is large (e.g., $0.8599$), fail to reject $H_0$ → feature may not be useful

---

## Lecture 7: Feature Engineering & Selection

### Dummy Variables for Categorical Data

For a categorical variable with $k$ levels, create $k-1$ dummy variables:

$$y_i = \beta_0 + \beta_1 x_i^{(1)} + \beta_2 x_i^{(2)} + \cdots + \beta_{k-1} x_i^{(k-1)} + \epsilon_i$$

where $x_i^{(j)} \in \{0, 1\}$.

**Example:** Predicting credit card balance by gender:
$$y_i = \beta_0 + \beta_1 \text{Female}_i + \epsilon_i$$

- $\beta_0$ = average balance for males (baseline)
- $\beta_0 + \beta_1$ = average balance for females

### Interaction (Synergy) Model

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \times x_2) + \epsilon$$

**Example:** Advertising data with interaction:
$$\text{sales} = \beta_0 + \beta_1 \text{TV} + \beta_2 \text{Radio} + \beta_3 (\text{TV} \times \text{Radio}) + \epsilon$$

**Interpretation:** The effect of TV advertising depends on the level of Radio advertising (and vice versa).

### Polynomial Regression

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \epsilon$$

**Example:** Auto fuel efficiency:
$$\text{mpg} = \beta_0 + \beta_1 \text{horsepower} + \beta_2 \text{horsepower}^2 + \epsilon$$

**Use Case:** Non-linear relationships between predictor and response.

### Model Selection Criteria

#### Mallow's Cp

$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} + 2p - n$$

or equivalently:

$$C_p = \frac{1}{n}\left(\text{RSS} + 2p\hat{\sigma}^2\right)$$

**Interpretation:** Trade-off between fit quality and model complexity.

#### Bayesian Information Criterion (BIC)

$$\text{BIC} = \frac{1}{n}\left(\text{RSS} + \log(n) \cdot p \cdot \hat{\sigma}^2\right)$$

**Comparison:** Since $\log(n) > 2$ for $n > 7$, BIC penalizes extra features more heavily than $C_p$.
- BIC tends to select simpler models
- $C_p$ tends to allow more complexity

### Adjusted R²

$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n - p - 1)}{\text{TSS}/(n - 1)}$$

**Advantage Over $R^2$:**
- Penalizes for adding unnecessary variables
- Does not always increase with model size
- Better for comparing models with different numbers of predictors

**Formula Component Breakdown:**
- Numerator: MSE of residuals
- Denominator: MSE of mean

---

## Lecture 8: Gradient Descent & Variants

### Supervised Learning Framework

**Problem:** We want to find optimal model parameters that minimize prediction error.

**Components:**
- **Model:** Parametric function with parameters $\theta = (\theta_1, \theta_2, \ldots, \theta_d)^T \in \mathbb{R}^d$
- **Loss Function:** $L(\theta)$ - measures how wrong the model predictions are
- **Goal:** Find $\theta^* = \arg\min_{\theta} L(\theta)$

### Loss Function

**Definition:** A function that quantifies the difference between observed and predicted values.

**Common Loss Functions:**
- **Regression (OLS):** $L(\theta) = \text{RSS}(\theta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
- **General:** Any differentiable function can be used

**Key Point:** Not all loss functions have closed-form solutions. We need iterative algorithms.

### Optimization Problem

**Find optimal parameters:**
$$\theta^* = \arg\min_{\theta} L(\theta)$$

**Meaning:** $\theta^*$ is the value of $\theta$ at which the loss function achieves its minimum.

---

### Gradient Descent Algorithm

#### One-Dimensional Case

**Simple Example:** $L(\theta) = (\theta - 3)^2$

**Solution by Calculus:**
$$L'(\theta) = 2(\theta - 3)$$
$$L'(\theta) = 0 \Rightarrow \theta = 3$$
$$L''(\theta) = 2 > 0 \text{ (convex, minimum confirmed)}$$

So $\theta^* = 3$ minimizes the loss.

**Gradient Descent Intuition:**

- At $\theta = 1$: $L'(1) = 2(-2) = -4 < 0$ → negative derivative → move right
- At $\theta = 6$: $L'(6) = 2(3) = 6 > 0$ → positive derivative → move left
- At $\theta = 3$: $L'(3) = 0$ → optimum found!

**Algorithm Principle:**
1. Start with an initial guess $\theta_0$
2. Take a step in the **negative gradient direction** to reduce loss
3. Repeat until convergence: $\theta_{t+1} = \theta_t - \eta \cdot L'(\theta_t)$

where $\eta$ is the **learning rate** (step size).

#### Multi-Dimensional Case

**When we have multiple parameters:**
$$\theta = (\theta_1, \theta_2, \ldots, \theta_d)^T \in \mathbb{R}^d$$

**Loss function becomes:**
$$L: \mathbb{R}^d \rightarrow \mathbb{R}$$

**Gradient (vector of partial derivatives):**
$$\nabla_\theta L(\theta) = \begin{bmatrix} \frac{\partial L}{\partial \theta_1} \\ \frac{\partial L}{\partial \theta_2} \\ \vdots \\ \frac{\partial L}{\partial \theta_d} \end{bmatrix} \in \mathbb{R}^d$$

### Gradient Descent Update Rule

**General form for multi-dimensional optimization:**
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)$$

**Components:**
- $\theta_t$ = current parameter values
- $\eta$ = **learning rate** (controls step size)
- $\nabla_\theta L(\theta_t)$ = gradient at current point
- Negative gradient points toward steepest descent

**Interpretation:**
- The gradient $\nabla L(\theta)$ points in the direction of steepest **increase**
- The negative gradient $-\nabla L(\theta)$ points toward steepest **decrease**
- We move in this direction to reduce the loss

### Key Insights from Gradient Descent

✓ **Closed-form solutions often don't exist** for complex models or loss functions

✓ **Iterative algorithm required** - we take small steps toward the optimum

✓ **Any differentiable loss function can be used** - not restricted to OLS/RSS

✓ **Learning rate is crucial** - too small = slow convergence, too large = overshoot

✓ **Convergence condition** - stop when $\|\nabla_\theta L(\theta)\| \approx 0$ (gradient near zero)

### Example: Gradient Descent on Quadratic Loss

**Loss Function:**
$$L(\theta) = (\theta - 3)^2$$

**Gradient:**
$$L'(\theta) = 2(\theta - 3)$$

**Starting from $\theta_0 = 1$ with learning rate $\eta = 0.1$:**

| Iteration | $\theta_t$ | $L'(\theta_t)$ | $\theta_{t+1}$ |
|-----------|-----------|---------------|---------------|
| 0 | 1.000 | -4.0 | 1.4 |
| 1 | 1.400 | -3.2 | 1.72 |
| 2 | 1.720 | -2.56 | 1.976 |
| ... | ... | ... | ... |
| ∞ | 3.000 | 0 | 3.000 |

**Pattern:** Each update gets closer to $\theta^* = 3$

### Gradient Descent Variants

**Core Algorithm Adaptations:**

1. **Batch Gradient Descent**
   - Use entire dataset to compute gradient
   - Stable convergence, but computationally expensive
   - Best for small to medium datasets

2. **Stochastic Gradient Descent (SGD)**
   - Use one sample at a time to compute gradient
   - Fast updates, lower memory usage
   - Noisier convergence path but can escape local minima

3. **Mini-batch Gradient Descent**
   - Use small batch of samples (e.g., 32, 64)
   - Balance between computational efficiency and stability
   - **Most commonly used in practice**

### Important Properties

**Convex Loss Functions:**
- Single global optimum
- Gradient descent is guaranteed to converge to global minimum
- Example: Linear regression with squared loss

**Non-Convex Loss Functions:**
- Multiple local optima
- Gradient descent may converge to local minimum instead of global
- Outcome depends on initialization and learning rate
- Example: Deep neural networks

### Learning Rate ($\eta$) Effects

| Learning Rate | Effect |
|--------------|--------|
| Too small ($\eta \ll 1$) | Very slow convergence, may not reach optimum in reasonable time |
| Optimal | Smooth convergence to optimum |
| Too large ($\eta \gg 1$) | Oscillation around optimum, may diverge |

---

**Key Takeaway:** Gradient descent is a general-purpose optimization algorithm that iteratively improves model parameters by moving in the direction of steepest descent. It fundamentally enables learning for any differentiable loss function and is the foundation of modern machine learning.

---



## Quick Reference Table

| Concept | Formula | Use Case |
|---------|---------|----------|
| Binomial Expectation | $E[X] = np$ | Expected successes |
| Binomial Variance | $\operatorname{Var}(X) = np(1-p)$ | Variability in successes |
| Exponential CDF | $P(T \leq t) = 1 - e^{-\lambda t}$ | Probability of event within time |
| Bayes' Theorem | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | Posterior probability |
| Dot Product | $\mathbf{a}^T \mathbf{b}$ | Vector similarity |
| Matrix Rank | # of non-zero rows in REF | Linear independence |
| Euclidean Distance | $\|\mathbf{a} - \mathbf{b}\|$ | Distance between points |
| OLS Solution | $\hat{\beta} = (X^T X)^{-1} X^T Y$ | Find regression coefficients |
| R-squared | $1 - \frac{\text{RSS}}{\text{TSS}}$ | Model fit percentage |
| MSE | $\frac{\text{RSS}}{n}$ | Average prediction error |
| Std Error | $\sqrt{\hat{\sigma}^2 (X^T X)^{-1}_{ii}}$ | Coefficient uncertainty |
| t-statistic | $\frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)}$ | Hypothesis testing |
| Adj-R² | $1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$ | Compare different models |

---


