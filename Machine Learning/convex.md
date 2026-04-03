Convex Optimization: Sets, Functions, and Duality
1. Convex Sets
Definition
A set X⊆R 
n
  is convex if, for every pair of points x,y∈X, the line segment connecting them lies entirely within X
,
. Mathematically, for any α∈
: 
αx+(1−α)y∈X∀x,y∈X
,
Operations Preserving Convexity
If C 
1
​
  and C 
2
​
  are convex sets, the following are also convex
:
Intersection: C 
1
​
 ∩C 
2
​
 
Cartesian Product: C 
1
​
 ×C 
2
​
 
Set Summation: C 
1
​
 +C 
2
​
 :={x 
1
​
 +x 
2
​
 :x 
1
​
 ∈C 
1
​
 ,x 
2
​
 ∈C 
2
​
 }
Example: Polyhedral Sets
A polyhedral set is the intersection of half-spaces, defined by linear inequalities
,
: 
{x∈R 
n
 :Ax≤b}
 Where A is an m×n matrix and b is an m×1 vector
,
.

--------------------------------------------------------------------------------
2. Convex Functions
Definition
A function f:X→R defined on a convex set X is convex if
,
: 
f(αx+(1−α)y)≤αf(x)+(1−α)f(y)∀x,y∈X,α∈
 Geometrically, the graph of the function lies below the chord joining any two points on the graph
,
.
Characterizations
First-Order Condition: If f is differentiable, it is convex iff its graph sits above its tangent hyperplanes
,
: 
f(y)≥f(x)+∇f(x) 
T
 (y−x)∀x,y∈X
Second-Order Condition: If f is twice-differentiable, it is convex iff its Hessian matrix (∇ 
2
 f) is positive semidefinite for all x
,
: 
∇ 
2
 f(x)⪰0∀x∈X
Concavity: A function f is concave if −f is convex
,
. The graph of a concave function lies above the chord joining two points
.
Key Properties
Jensen’s Inequality: For a random variable X and convex function f, f(E[X])≤E[f(X)]
.
Pointwise Supremum: The supremum of a family of convex functions {f 
i
​
 } 
i∈I
​
  is also convex
.
Norms: Any norm over a convex set is a convex function due to the triangle inequality
,
.

--------------------------------------------------------------------------------
3. Constrained Convex Optimization
The Primal Problem (P)
The goal is to minimize a convex objective function f(x) over a convex set defined by linear or convex constraints
,
: 
x∈X
min
​
 f(x)subject to Ax≤b
The Lagrangian and Dual Function
To solve (P), we define the Lagrangian L, which moves constraints into the objective as "violation penalties" using Lagrange multipliers α≥0
,
: 
L(x,α)=f(x)+α 
T
 (Ax−b)
 The Lagrange Dual Function F(α) is the infimum of the Lagrangian over x
,
: 
F(α)= 
x∈X
inf
​
 L(x,α)
The Dual Optimization Problem (D)
The dual problem aims to maximize F(α)
,
: 
α≥0
max
​
 F(α)
Weak Duality: The optimal value of the dual problem d 
∗
  is always ≤ the optimal primal value p 
∗
 
.
Strong Duality: Under certain conditions (like Slater’s condition), p 
∗
 =d 
∗
 
,
.

--------------------------------------------------------------------------------
4. Karush-Kuhn-Tucker (KKT) Conditions
A point (x 
∗
 ,α 
∗
 ) is optimal if it satisfies
,
:
Stationarity: ∇ 
x
​
 L(x 
∗
 ,α 
∗
 )=0
Primal Feasibility: Ax 
∗
 ≤b
Complementary Slackness: α 
i
​
 (a 
i
​
 x 
∗
 −b 
i
​
 )=0∀i

--------------------------------------------------------------------------------
5. Practical Numerical Example
Problem: Minimize f(x)=x 
1
2
​
 +4x 
2
2
​
  subject to x 
1
​
 +x 
2
​
 ≤−1
.
Lagrangian: L=x 
1
2
​
 +4x 
2
2
​
 +α(x 
1
​
 +x 
2
​
 +1)
.
Minimize L w.r.t x:
∂x 
1
​
 
∂L
​
 =2x 
1
​
 +α=0⟹x 
1
∗
​
 =−α/2
∂x 
2
​
 
∂L
​
 =8x 
2
​
 +α=0⟹x 
2
∗
​
 =−α/8
Dual Objective: Substitute x 
∗
  into L: F(α)=α(1− 
16
5α
​
 )
.
Maximize F(α):  
dα
dF
​
 =1− 
16
10α
​
 =0⟹α 
∗
 =8/5
.
Final Solution: x 
1
∗
​
 =−4/5,x 
2
∗
​
 =−1/5
.