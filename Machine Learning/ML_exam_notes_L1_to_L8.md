# Machine Learning Exam Notes (Lectures 1-8)

A comprehensive study guide covering key concepts, formulas, and examples from your ML course.

---

## Lecture 2: Linear Algebra Fundamentals

### Inner (Dot) Product

For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:
$$\langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^T \mathbf{b}$$

**Example:** If $\mathbf{a} = [1, 2]$ and $\mathbf{b} = [3, 4]$, then:
$$\langle \mathbf{a}, \mathbf{b} \rangle = (1 \times 3) + (2 \times 4) = 11$$

### Vector Norm (Magnitude)

$$\|\mathbf{a}\| = \sqrt{\langle \mathbf{a}, \mathbf{a} \rangle}$$

**Example:** For vector $\mathbf{a} = [3, 4]$:
$$\|\mathbf{a}\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$

### Distance Between Vectors

$$\text{dist}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|$$

**Example:** The distance between $\mathbf{a} = [0, 0]$ and $\mathbf{b} = [3, 4]$ is:
$$\|\mathbf{a} - \mathbf{b}\| = \|[-3, -4]\| = \sqrt{(-3)^2 + (-4)^2} = 5$$

### Projection of b onto a

$$\text{proj}_{\mathbf{a}} \mathbf{b} = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\|\mathbf{a}\|^2} \mathbf{a}$$

**Example:** Projecting vector $\mathbf{b}$ onto $\mathbf{a}$ finds the component of $\mathbf{b}$ that lies in the direction of $\mathbf{a}$.

### Linear Independence/Dependence

A collection of vectors is **linearly dependent** if:
$$\sum \beta_i \mathbf{a}_i = \mathbf{0} \text{ for } \beta \text{ not all zero}$$

**Example:** If $\mathbf{c} = \beta_1 \mathbf{a} + \beta_2 \mathbf{b}$, then $\{\mathbf{a}, \mathbf{b}, \mathbf{c}\}$ are linearly dependent because $\beta_1 \mathbf{a} + \beta_2 \mathbf{b} + (-1)\mathbf{c} = \mathbf{0}$.

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
.
Example: This formula calculates the optimal coefficients (β) that minimize the error, provided the feature matrix X has full column rank (columns are linearly independent)
.
The Hat Matrix (H): H=X(X 
T
 X) 
−1
 X 
T
 , where  
Y
^
 =HY
.
Example: The Hat Matrix is a geometric tool that "projects" the actual observed outputs (Y) onto the space spanned by your features to find the best-fitting predictions ( 
Y
^
 )
.
--------------------------------------------------------------------------------
Lecture 6: Model Evaluation
Total Sum of Squares (TSS): TSS=∑(y 
i
​
 − 
y
ˉ
​
 ) 
2
 
.
Example: TSS captures the inherent variability of the training outputs (e.g., how much sales figures vary naturally before considering advertising)
.
R-squared (R 
2
 ): R 
2
 =1− 
TSS
RSS
​
 
.
Example: If R 
2
 =0.968 (as seen in the advertising interaction model), it means 96.8% of the variability in sales is explained by the model
.
Unbiased Estimate of Noise Variance ( 
σ
^
  
2
 ):  
σ
^
  
2
 = 
n−p−1
RSS
​
 
.
Example: This provides an estimate of the variance of the random error (ϵ) associated with the measurements
.
Standard Error of a Coefficient: SE( 
β
^
​
  
i
​
 )= 
σ
^
  
2
 C 
i
​
 

​
 , where C 
i
​
  is a diagonal entry of (X 
T
 X) 
−1
 
.
Example: In the advertising data, the SE for the TV coefficient was 0.0014, allowing for the calculation of a confidence interval
.
Hypothesis Test Statistic: Z= 
SE( 
β
^
​
  
i
​
 )
β
^
​
  
i
​
 
​
 
.
Example: To see if Newspaper ads are useful, we test H 
0
​
 :β 
newspaper
​
 =0. If the resulting p-value is large (e.g., 0.8599), we fail to reject the null and conclude the feature might not be useful
.
--------------------------------------------------------------------------------
Lecture 7: Feature Engineering & Selection
Dummy Variables for Categorical Data: y 
i
​
 =β 
0
​
 +β 
1
​
 x 
i
​
 +ϵ 
i
​
 , where x 
i
​
 =1 if Female and 0 if Male
.
Example: In the credit card dataset, β 
0
​
  represents the average balance for males (the baseline), while β 
0
​
 +β 
1
​
  represents the average balance for females
.
Interaction (Synergy) Model: sales=β 
0
​
 +β 
1
​
 TV+β 
2
​
 radio+β 
3
​
 (radio×TV)+ϵ
.
Example: This model allows the effect of TV advertising to increase if the Radio advertising budget also increases, capturing "synergy"
.
Polynomial Regression: mpg=β 
0
​
 +β 
1
​
 horsepower+β 
2
​
 horsepower 
2
 +ϵ
.
Example: This accounts for non-linear relationships, such as when the efficiency of a car (mpg) drops more sharply as horsepower increases
.
Mallow’s C 
p
​
  & BIC: C 
p
​
 = 
n
1
​
 (RSS+2d 
σ
^
  
2
 ) and BIC= 
n
1
​
 (RSS+log(n)d 
σ
^
  
2
 )
.
Example: These criteria are used to select the best model from a group. Because log(n)>2 for n>7, BIC places a heavier penalty on having too many features (d) and usually selects smaller models than C 
p
​
 
.
Adjusted R 
2
 : 1− 
TSS/(n−1)
RSS/(n−d−1)
​
 
.
Example: Unlike standard R 
2
 , Adjusted R 
2
  "pays a price" for including unnecessary variables, making it a better metric for comparing models of different sizes
.
