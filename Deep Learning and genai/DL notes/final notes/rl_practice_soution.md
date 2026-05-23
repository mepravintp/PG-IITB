# SOLUTION MANUAL: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
**Course Module:** Advanced Machine Learning  
**Topic:** Reward Modeling & PPO Policy Optimization

---

## Section 4.2: Reward Model Training and Regularized Objective

### Question (a)
#### **Problem Statement**
* **Given:** A reward model outputs the following scores for three preference pairs:
  
  | Pair | Chosen Reward $r(x,y^{+})$ | Rejected Reward $r(x,y^{-})$ |
  | :--- | :--- | :--- |
  | 1 | 2.0 | 1.0 |
  | 2 | 0.5 | 1.5 |
  | 3 | 3.0 | 0.0 |

  The Bradley-Terry loss formula for an individual pair is defined as:
  $$l_{i} = \log \sigma(r(x,y_{i}^{+}) - r(x,y_{i}^{-}))$$
  where the sigmoid function is:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  The average loss optimization function across all samples is:
  $$\overline{l} = -\frac{1}{3}\sum_{i}l_{i}$$

* **Asked:** Compute the Bradley-Terry loss for each individual pair ($l_{1}, l_{2}, l_{3}$) and find the final negative average loss ($\overline{l}$).

#### **Step-by-Step Calculation**

**1. Loss for Pair 1:**
* *Step 1: Calculate the core score difference ($z_{1}$)*
  $$z_{1} = 2.0 - 1.0 = 1.0$$
* *Step 2: Pass the result through the sigmoid function*
  $$\sigma(1.0) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.3679} \approx 0.7311$$
* *Step 3: Compute the log value*
  $$l_{1} = \log(0.7311) \approx -0.3133$$

**2. Loss for Pair 2:**
* *Step 1: Calculate the core score difference ($z_{2}$)*
  $$z_{2} = 0.5 - 1.5 = -1.0$$
* *Step 2: Pass the result through the sigmoid function*
  $$\sigma(-1.0) = \frac{1}{1 + e^{-(-1)}} = \frac{1}{1 + 2.7183} \approx 0.2689$$
* *Step 3: Compute the log value*
  $$l_{2} = \log(0.2689) \approx -1.3133$$

**3. Loss for Pair 3:**
* *Step 1: Calculate the core score difference ($z_{3}$)*
  $$z_{3} = 3.0 - 0.0 = 3.0$$
* *Step 2: Pass the result through the sigmoid function*
  $$\sigma(3.0) = \frac{1}{1 + e^{-3}} = \frac{1}{1 + 0.0498} \approx 0.9526$$
* *Step 3: Compute the log value*
  $$l_{3} = \log(0.9526) \approx -0.0486$$

**4. Average Loss Evaluation ($\overline{l}$):**
* *Step 1: Sum the calculated log losses*
  $$\sum_{i=1}^{3}l_{i} = (-0.3133) + (-1.3133) + (-0.0486) = -1.6752$$
* *Step 2: Compute the negative average scaled across the 3 pairs*
  $$\overline{l} = -\frac{1}{3} \times (-1.6752) = 0.5584$$

---

### Question (b)
#### **Problem Statement**
* **Given:**
  * Base scalar reward for a single response string: $r(x,y) = 1.5$
  * Evaluated log policy ratio: $\log\left(\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}\right) = 0.3$
  * Regularization scaling coefficient: $\beta = 0.1$
* **Asked:**
  * Write out the mathematical regularized RLHF reward maximization objective function $\mathcal{J}(\theta)$ for a policy $\pi$.
  * Compute the localized regularized reward $r_{s}(x,y)$ for this response sample.
  * Interpret whether this response is structurally beneficial to keep or suppress.

#### **Step-by-Step Solution**

**1. Optimization Objective Function Formula:**
The global KL-regularized reward maximization objective is defined as:
$$\mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}(y|x)}\left[r(x,y) - \beta \log\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

**2. Local Computation of Regularized Reward $r_{s}(x,y)$:**
* *Step 1: Isolate the single-sample evaluation equation*
  $$r_{s}(x,y) = r(x,y) - \beta \log\left(\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}\right)$$
* *Step 2: Substitute the provided problem variables into the equation*
  $$r_{s}(x,y) = 1.5 - (0.1 \times 0.3)$$
* *Step 3: Evaluate the final arithmetic*
  $$r_{s}(x,y) = 1.5 - 0.03 = 1.47$$

**3. Behavioral Interpretation:**
Because the regularized reward value ($1.47$) remains **highly positive**, this specific generated response is mathematically optimal and **beneficial to keep**. The regularizing penalty subtracted from the base reward ($0.03$) is minimal and does not override the fundamental alignment utility provided by the raw reward score of $1.5$.

---

## Question 4.2: Mathematical Core Concepts

### Part (a)
#### **Problem Statement**
* **Given:** The Kullback-Leibler (KL) penalty expression: $\beta \log(\pi_{\theta}/\pi_{\text{ref}})$.
* **Asked:** Explain the primary algorithmic role of this term and deduce what happens when $\beta \rightarrow 0$ and $\beta \rightarrow \infty$.

#### **Answer**
* **Role:** The KL penalty acts as a structural anchor during policy optimization. It restricts the parameterized policy $\pi_{\theta}$ from straying too far from the safety boundaries of the initial reference model $\pi_{\text{ref}}$. This calculation suppresses "reward hacking"—a failure mode where a language model generates nonsensical, unreadable formatting strings that mathematically exploit blind spots in the reward model to claim high scores.
* **When $\beta \rightarrow 0$:** The constraints are dissolved. The optimization loop optimizes purely for raw reward model points, resulting in severe distribution shift, training instability, and chaotic policy outputs due to exploitation of the reward mechanism.
* **When $\beta \rightarrow \infty$:** The penalty scaling grows infinite. The optimization framework prioritizes exact baseline alignment over reward gains, forcing the updated policy to stay completely identical to its origin ($\pi_{\theta} = \pi_{\text{ref}}$), ending all behavioral improvement.

---
### Part (b)
#### **Problem Statement**
* **Given:** The objective function for Proximal Policy Optimization with Clipping (PPO-Clip):
  $$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}\left[\min(\rho_{t}A_{t}, \text{clip}(\rho_{t}, 1-\epsilon, 1+\epsilon)A_{t})\right]$$
  where $\rho_{t} = \frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{k}(a_{t}|s_{t})}$ tracks the active importance sampling ratio and $A_{t}$ is the state advantage.
* **Asked:** * Clarify exactly what the programmatic clipping boundary achieves.
  * Resolve what the effective gradient evaluates to when $A_{t} > 0$ and $\rho_{t} > 1 + \epsilon$.

#### **Answer**
* **Clipping Objective:** The clipping operation enforces a strict mathematical safety corridor around updates. By capping the importance ratio $\rho_{t}$ inside a tight localized interval $[1-\epsilon, 1+\epsilon]$, it prevents sudden, destructively massive steps in policy configuration space that would invalidate the variance properties of the data distribution collected by the older policy generation ($\pi_{k}$).
* **Effective Gradient:** When $A_{t} > 0$, the chosen action performs better than baseline expectations, and the optimizer tries to scale up its production probability (driving $\rho_{t}$ higher). However, once $\rho_{t}$ breaches the upper bound ($\rho_{t} > 1 + \epsilon$), the min-clip function switches to outputting a completely static value:
  $$\text{Value} = (1 + \epsilon)A_{t}$$
  Because this clipped value is a flat ceiling that no longer changes with variations in the target policy parameter tensor $\theta$, its derivative vanishes entirely:
  $$\nabla_{\theta} \mathcal{L}_{\text{PPO}}(\theta) = 0$$
  Consequently, the **effective gradient becomes zero**, freezing any further policy shifts for this action within the current optimization cycle to avoid over-correcting.

---

### Part (c)
#### **Problem Statement**
* **Given:** The state-action quality value function $Q(s_{t}, a_{t})$ and the localized state value baseline function $V(s_{t})$.
* **Asked:** * Formulate the mathematical definition of the advantage function $A(s_{t}, a_{t})$ using $Q$ and $V$.
  * Explain intuitively why switching from $Q$ values to advantage $A$ values drops variance in policy gradient update paths.

#### **Answer**
* **Formulation:**
  $$A(s_{t}, a_{t}) = Q(s_{t}, a_{t}) - V(s_{t})$$

* **Intuitive Variance Reduction:** In raw policy gradient approaches using $Q(s_{t}, a_{t})$, step trajectories carry absolute environment reward values. If an environment features some states that inherently yield massive base payouts and other states that yield minuscule rewards, gradient directions will shift wildly due to the pure luck of state assignment rather than the objective quality of action selections.
  
  By subtracting $V(s_{t})$, we remove the state's baseline background score. Th