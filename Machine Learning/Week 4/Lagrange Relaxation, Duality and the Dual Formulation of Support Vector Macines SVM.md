## Maximum Classifiers with Basis Expansion
$$
\arg\min_{w, b} \left\{ \frac{1}{2} \|w\|^2 \right\}

$$
Subject to:
$$
y^{(n)}(\mathbf{w}^T\phi(\mathbf{x})+b)  \geq 1, \, \forall (\mathbf{x}^{(n)}, y^{(n)}) \in \mathcal{T}
$$
The problem is that depending on $\phi(\mathbf{x})$, its computation can be very expensive, as it may be taking us to a very high dimensional problem.

## Rewriting Our Optimization Problem

We will get rid of w and b, and also of $\mathbf{w}^T\phi(x^{n})$ and $||\mathbf{w}||^2$ on Dual Formulation.

### Lagrange Relaxation

- Primal Formulation
$$
\min_xF(\mathbf{x}) \quad \text{Subject to:   } f(\mathbf{x}) \leq 0 
$$
- Lagrange Relaxation
$$
\min_x \{  F(\mathbf{x}) +af(x)\}
$$
Where a $\geq$ 0 is called a Lagrange multiplier and $L(\mathbf{x},a) = F(x)+af(x)$  is called the Lagranian.

- The objective will be penalized when the constraint is violated, unless a = 0, so we will be searching for solution that does not violate the constraint. 
	- We need to choose appropriate values for $a_i$ to ensure that the penalty is large enough.
	- We will be rewarding the objective when a constraint is not violated, unless its a= 0
	- When a feasible solution is found $L(\mathbf{x,a}) \leq F(\mathbf{x})$

Formulas:

- Lagrange Relaxation:
$$
\min_{x} \left( F(x) + \sum_{i=1}^{N} a_i f_i(x) \right)
$$
- Lagrange multipliers
$$
a_i \geq 0, \quad i \in \{1, \dots, N\}
$$
- Lagranian:
$$
L(x, \mathbf{a}) = F(x) + \sum_{i=1}^{N} a_i f_i(x)
$$

When a constraint is violated, the penalty will become infinitely large, so an optimization algorithm will try to avoid violations as much as possible.

![[Pasted image 20241113120034.png]]

- When constraint is not violated, $f(\mathbf{x})$ is either - or 0 so $L(\mathbf{x,a)} = F(\mathbf{x})$ 
### Minimax Primal Formulation
$$
\min_{x} F(x) \quad \text{Subject to: } f_i(x) \leq 0, \, i \in \{1, \dots, N\}
$$
$$
\min_x \left\{ F(x) + \max_{\mathbf{a}} \sum_{i=1}^{N} a_i f_i(x) \right\}
$$
$$
\min_{x} \max_{\mathbf{a}} \left( F(x) + \sum_{i=1}^{N} a_i f_i(x) \right)
$$
$$
L(x, \mathbf{a}) = F(x) + \sum_{i=1}^{N} a_i f_i(x)

$$
### Dual Formulation 

- Minimax Primal Formulatoin
$$
\min_{x} \max_{\mathbf{a}} \left\{ F(x) +  \sum_{i=1}^{N} a_i f_i(x) \right\}
$$
- Dual Formulation

$$
\max_{\mathbf{a}}\min_{x} \left\{ F(x) +  \sum_{i=1}^{N} a_i f_i(x) \right\}
$$
Dual may be easier to solve however it is not always equivalent to the primal, this is known as weak duality.

### Strong Duality
Minimax = Maxmin

![[Pasted image 20241113121239.png]]

#### Minimax = Maxmin Illustrative Example

![[Pasted image 20241113121342.png]]

## Karush-Kuhn-Tucker (KKT) Conditions



- **Unconstrained Convex Optimization**: The necessary and sufficient condition for optimality is:
$$
\nabla F(x) = 0

$$
- **KKT Conditions**: For a convex optimization problem:
$$
\min_{x} F(x) \quad \text{with constraints } f_i(x) \leq 0

$$
- **Stationarity**: The stationarity condition for the primal-dual problem:
$$
\nabla_{x} L(x, \mathbf{a}) = \nabla_{x} F(x) + \sum_{i=1}^{N} a_i \nabla_{x} f_i(x) = 0

$$
- **Complementary Slackness**: The complementary slackness condition:
$$
a_i f_i(x) = 0, \quad \forall i \in \{1, \dots, N\}

$$
- **Feasibility**: Primal and Dual Constraints
 $$
 f_i(x) \leq 0, \quad \forall i \in \{1, \dots, N\}

$$
$$
a_i \geq 0, \quad \forall i \in \{1, \dots, N\}
$$
## SVM: From Primal to Dual

- Primal Foundation:
$$
\min_{w, b} \left\{ \frac{1}{2} \|w\|^2 \right\}
\quad
y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \geq 1, \quad \forall (x^{(n)}, y^{(n)}) \in \mathcal{T}
$$
- Rewritten Constraints:
$$
y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) - 1 \geq 0
$$
$$
1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \leq 0
$$
### New Lagrange Relaxation
$$
\min_{w, b} \left\{ \frac{1}{2} \|w\|^2 + \sum_{n=1}^{N} a^{(n)} \left( 1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \right) \right\}
$$
- Constraint:
$$
a^{(n)} \geq 0, \quad \forall (x^{(n)}, y^{(n)}) \in \mathcal{T}
$$
### New Minimax


- Minimax Primal Formulation:
$$
\min_{w, b} \max_{\mathbf{a}} \left\{ \frac{1}{2} \|w\|^2 + \sum_{n=1}^{N} a^{(n)} \left( 1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \right) \right\}
$$
- Dual Formulation:
$$
\max_{\mathbf{a}} \min_{w, b} \left\{ \frac{1}{2} \|w\|^2 + \sum_{n=1}^{N} a^{(n)} \left( 1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \right) \right\}
$$
- Constraints:
$$
a^{(n)} \geq 0, \quad \forall (x^{(n)}, y^{(n)}) \in \mathcal{T}
$$
### Further Simplifying Equations
$$
\max_{\mathbf{a}} \min_{w, b} L(a, w, b) = \frac{1}{2} \|w\|^2 + \sum_{n=1}^{N} a^{(n)} \left( 1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \right)
$$

- To eliminate b:
$$
\sum_{n=1}^{N} a^{(n)} y^{(n)} = 0
$$
- To eliminate w:
$$
w = \sum_{n=1}^{N} a^{(n)} y^{(n)} \phi(x^{(n)})
$$
### Dual Representation

$$
\max_{\mathbf{a}} \min_{w, b} L(a, w, b) = \frac{1}{2} \|w\|^2 + \sum_{n=1}^{N} a^{(n)} \left( 1 - y^{(n)} \left( w^T \phi(x^{(n)}) + b \right) \right)
$$
Transformed to:
$$
\arg\max_{\mathbf{a}} \tilde{L}(\mathbf{a}) = \sum_{n=1}^{N} a^{(n)} - \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a^{(n)} a^{(m)} y^{(n)} y^{(m)} k(x^{(n)}, x^{(m)})
$$
Where:
$$
k(x^{(n)}, x^{(m)}) = \phi(x^{(n)})^T \phi(x^{(m)})
$$
Constraints:
$$
a^{(n)} \geq 0, \quad \forall n \in \{1, \dots, N\}
$$
$$
\sum_{n=1}^{N} a^{(n)} y^{(n)} = 0
$$
## Kernel Trick
![[Pasted image 20241113123756.png]]

- This calculation can be generalizes to basis expansions composed of all terms of order up to p
$$
k(\mathbf{x},\mathbf{z}) = (1 + \mathbf{x}^T\mathbf{z})^p
$$
