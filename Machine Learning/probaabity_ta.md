1. Bag with 5 red and 3 blue balls
Two draws without replacement, both red:
$$
P(\text{2 red}) = \frac{5}{8} \times \frac{4}{7} = \frac{20}{56} = \frac{5}{14} \approx 0.357
$$

2. Binomial Distribution X\sim Binomial(20,0.3)
$$
E[X] = np = 20 \times 0.3 = 6
$$
$$
\operatorname{Var}(X) = np(1-p) = 20 \times 0.3 \times 0.7 = 4.2
$$

3. Exponential Distribution \lambda =3 per hour
Convert 15 minutes = 0.25 hours.
For exponential distribution,
$$
P(T \leq t) = 1 - e^{-\lambda t}
$$
$$
P(T \leq 0.25) = 1 - e^{-3 \times 0.25} = 1 - e^{-0.75} \approx 1 - 0.472 = 0.528
$$
So probability ≈ 52.8%.

4. Portfolio of two independent investments
Equal weights: w_1=w_2=0.5.
$$
E[P] = 0.5 \times E[X_1] + 0.5 \times E[X_2] = 0.5 \times 0.08 + 0.5 \times 0.06 = 0.07 \quad (7\%)
$$
$$
\operatorname{Var}(P) = (0.5^2) \operatorname{Var}(X_1) + (0.5^2) \operatorname{Var}(X_2) = 0.25 \times 9 + 0.25 \times 4 = 2.25 + 1 = 3.25
$$

5. Gene test with Bayes’ theorem
 Prior: $P(\text{Gene}) = 0.02$, $P(\text{No Gene}) = 0.98$.
 Test accuracy: Sensitivity $= 0.98$, Specificity $= 0.98$.
=0.98\cdot 0.02+0.02\cdot 0.98=0.0196+0.0196=0.0392
$$
$$
P(+) = P(+|\text{Gene}) P(\text{Gene}) + P(+|\text{No Gene}) P(\text{No Gene}) 
$$
$$
= 0.98 \times 0.02 + 0.02 \times 0.98 = 0.0196 + 0.0196 = 0.0392
$$
$$
P(\text{Gene}|+) = \frac{0.98 \times 0.02}{0.0392} = \frac{0.0196}{0.0392} = 0.5
$$
So probability = 50%.
So probability $= 50\%$.
So probability = 50%.
