## Linearly Separable Problems

- Example setup:
    - Decision boundary defined by $w^T x = 0$

## Nonlinear Transformation / Basis Expansion

- A linear decision boundary would lead to under fitting
![[Pasted image 20241112223219.png]]

- **Example Transformation**:
    - Given $\phi(x) = (x_1, x_1^2)^T$, we create a higher-dimensional embedding / feature space.
- Enables learning a linear model in higher-dimensional space that can separate data in original space.

![[Pasted image 20241112223517.png]]

- Higher-dimensional embedding / feature space:
	- Feature transform/ basis expansion $\phi(x)$
	- Basis functions $(x_1,x_2)^T$

## Decision Boundaries and Polynomial Order

For a polynomial decision boundary of degree p in the original space, create a feature transform that includes all terms of order $\leq$ p that can be created based on the input variable x.

- Polynomial order 2 and a problem with 1 input variable
$$
x = (1, x_1) \rightarrow \phi(x) = (1, x_1, x_1^2)^T
$$

- Polynomial of order 2 and a problem with 2 input variables:
$$
x = (1, x_1, x_2) \rightarrow \phi(x) = (1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)^T
$$
Any decision boundary that is a polynomial of order p in $\mathbf{x}$ is linear in $\phi(x)$, with this logic we can adopt linear models in the higher dimensional embedding formed by $\phi(x)$, to learn decision boundaries corresponding to polynomials of order p in $\mathbf{x}$
### Example: Quadratic Decision Boundary

We need a quadratic decision boundary for problem with 1 input variable:
 $$
 w_0 x_0 + w_1 x_1 + w_2 x_1^2 = 0
$$
Non linear transformation:
$$
\mathbf{x} = (1,x_1)^T \rightarrow  \phi(x) = (1, x_1, x_1^2)^T
$$
Linear decision boundary in the feature space corresponds to a quadratic decision boundary in the original space
$$
\mathbf{w}^T\phi(\mathbf{x}) = 0  \quad \mathbf{w}^T = (w_0,w_1,w_2)
$$
$$
w_0 \cdot1 + w_1 x_1 + w_2 x_1^2 = 0 \rightarrow w_0 \cdot1 + w_1\phi_1(\mathbf{x}) + w_2\phi_2(\mathbf{x})
$$

![[Pasted image 20241112231813.png]]
![[Pasted image 20241112231823.png]]

## Other Nonlinear transformations

Besides polynomial expansions, non-polynomial transformations can be used.

$$
x = (1, x_1) \rightarrow \phi(x) = (1, x_1, e^{x_1})^T

$$
## Dimensionality of Feature Space

Most of the time, we will be transforming the problem into a higher dimensional space, but this is not always the case, if some terms are not needed in the decision boundary in the original space they can be removed or not included in the nonlinear transformation.

$$
x = (1, x_1, x_2) \rightarrow \phi(x) = (1, x_1, x_2^2)^T

$$

## Adopting Nonlinear Transformations in Logistic Regression

$$
logit(p_1)= \mathbf{w}^T\mathbf{x}

\quad 

p_1=p(1|\mathbf{x,w})= \frac{e^{\mathbf{w}^T \mathbf{x}}}{1+e^{\mathbf{w}^T \mathbf{x}}}

$$
$$
\downarrow
$$
$$
logit(p_1)=\mathbf{w}^T\phi(x) \quad 
p_1=p(1|\mathbf{\phi(x),w})= \frac{e^{\mathbf{w}^T \mathbf{\phi(x)}}}{1+e^{\mathbf{w}^T \mathbf{\phi(x)}}}
$$
With this changes our objective function can be viewed as:

$$
E(w) = -\sum_{i=1}^{N} \left[ y^{(i)} \ln p(1 | \phi(x^{(i)}), w) + (1 - y^{(i)}) \ln (1 - p(1 | \phi(x^{(i)}), w)) \right]

$$
The Gradient changes to this:
$$
\nabla_{E}(\mathbf{w}) = \sum_{i=1}^{N} \left( p(1 | \phi(\mathbf{x}^{(i)}), \mathbf{w}) - y^{(i)} \right) \phi(\mathbf{x}^{(i)})
$$
And the Hessian Matrix changes to this:
$$
H_{E}(\mathbf{w}) = \sum_{i=1}^{N} p(1 | \phi(\mathbf{x}^{(i)}), \mathbf{w}) \left( 1 - p(1 | \phi(\mathbf{x}^{(i)}), \mathbf{w}) \right) \phi(\mathbf{x}^{(i)}) \phi(\mathbf{x}^{(i)})^T
$$

### Steps for Adopting Nonlinear Transformations

1. Choose a nonlinear transformation.
$$
\mathbf{x} = (1, x_1) \rightarrow \phi(\mathbf{x}) = (1, x_1, x_1^2)^T
$$
2. Apply it to training examples, they need to be $(\phi(\mathbf{x}),y)$
$$
\mathcal{T} = \{ (\phi(\mathbf{x}^{(1)}), y^{(1)}), (\phi(\mathbf{x}^{(2)}), y^{(2)}), \cdots, (\phi(\mathbf{x}^{(N)}), y^{(N)}) \}
$$
3. Create a linear model based on transformed examples, same algorithm as others.
$$
\text{Given } \mathcal{T}, \quad \arg \min_{\mathbf{w}} E(\mathbf{w})
$$
5. Determine the nonlinear model by replacing $\phi_i(\mathbf{x})$ with the corresponding value that depends on $\mathbf{x}$
$$
w_0 \times 1 + w_1 \phi_1(\mathbf{x}) + w_2 \phi_2(\mathbf{x}) = 0 \rightarrow w_0 \times 1 + w_1 x_1 + w_2 x_1^2 = 0
$$
## Is Logistic Regression Still a Linear Model?

- Logistic regression remains linear in terms of its parameters even with nonlinear transformations, as these affect the input space but not the parameter linearity.
## Advantages of Linear Models

- Efficient learning algorithms
- Good generalization properties
## Caveats of Nonlinear Transforms

- Number of dimensions can become too high.
- Risk of overfitting due to complexity in the transformed space

![[Pasted image 20241113000716.png]]