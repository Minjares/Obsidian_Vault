
## Problems of gradient descent

- Can get stuck in local minima.
	- Not a problem with Cross-Entropy Loss, the function is strictly convex as seen below.
- Big $\eta$ = difficult to find optimum
- Small $\eta$ = long time to find optimum

### Illustrative Loss Function
![[Pasted image 20241108135858.png]]


### Differential Curve

![[Pasted image 20241108153033.png]]

Each line corresponds to the point in the weight space where the loss is the same.

![[Pasted image 20241108153230.png]]

Steepest descent forms a 90 degrees angle with the line, in the figure in the right, $w_1$ and $w_2$ axis have different magnitudes.
$$
\frac{\partial E}{\partial w_2} > \frac{\partial E}{\partial w_1}

\text{ depending on the location}
$$

The path of steepest descent in a lot of functions is only an instantaneous direction of best movement but it may not be the best direction in the longer term.

![[Pasted image 20241108153506.png]]

Changes in w$_1$ are smaller, changes in w$_2$ are larger, but get undone in the next iteration, which the optimization process is slow. 

### Why may the loss function have different gradients along different dimensions

The loss is a function of w, as w is what we are varying during the learning process.

$$
E(\mathbf{w}) = -\sum_{i=1}^N \left[ y^{(i)} \ln p(1 | \mathbf{x}^{(i)}, \mathbf{w}) + (1 - y^{(i)}) \ln (1 - p(1 | \mathbf{x}^{(i)}, \mathbf{w})) \right]


$$
Sigmoid function
$$
p_1 = \frac{e^{\mathbf{w}^\top \mathbf{x}}}{1 + e^{\mathbf{w}^\top \mathbf{x}}}
$$
The larger effect w$_2$ may be a result of the training examples having larger values for the feature x$_2$ than for x$_1$ 

$$
E(\mathbf{w}) = w_1^2 + 4w_2^2
$$
#### Standardization

Different partial derivatives are a result of different input variables having different scales and variance, this affect the loss function to a different extent.

- Standardizing input variables is recommended to help with this.
	- This may not be enough to solve the issue of overshooting along certain axis.

If we could make use of information about the curvature itself, this may help to improve on this issue, we can make use of second-order derivatives to tell if the gradient is changing too much, with this we can make smaller weight updates accordingly.

**Solution**: Newton-Raphson Method: Uni-variate Case for Illustration Purposes

## Newton-Raphson: Uni variate Weight Update Rule


We still move in the opposite direction of the gradient, but we will reduce the size of the update if the curvature is high.
$$
\downarrow w = w - \frac{E'(w)}{E''(w)}\uparrow
$$
$$
\text{**There's an absence of learning rate {$\eta$} here}
$$


### Origin of the Rule

The optimum of a quadratic function can be easily found in closed form, being potentially easier to find than the optimum of the original function.
![[Pasted image 20241108160510.png]]

### Taylor Polynomial for a Local Approximation of $E(\mathbf{w})$

Can be used to approximate a function $E(\mathbf{w})$ at $\mathbf{w}_0$
$$
T_n(\mathbf{w}) = \sum_{k=0}^n \frac{E^{(k)}(\mathbf{w}_0)}{k!}(\mathbf{w}-\mathbf{w}_0)^k
$$
where $E^{(k)}(\mathbf{w}_0)$ is the k-th order derivative of E at w$_0$ 

Important to note that w$_0$ is not the first element inside of $\mathbf{w}$, it is commonly used with the entirety of the vector.

#### Illustration
![[Pasted image 20241108161510.png]]


### Using it for a local approximation of $E(\mathbf{w})$

$$
T_n(\mathbf{w}) = \sum_{k=0}^n \frac{E^{(k)}(\mathbf{w}_0)}{k!}(\mathbf{w}-\mathbf{w}_0)^k
$$
This example is using Taylor polynomial of degree 2
$$
T_n(w) = \frac{E^{(0)}(w_0)}{0!}(w - w_0)^0 + \frac{E^{(1)}(w_0)}{1!}(w - w_0)^1 + \frac{E^{(2)}(w_0)}{2!}(w - w_0)^2
$$
$$
= E(w_0)(w - w_0)^0 + E'(w_0)(w - w_0)^1 + \frac{E''(w_0)}{2}(w - w_0)^2
$$
$$
= E(w_0) + (w - w_0)E'(w_0) + \frac{(w - w_0)^2}{2}E''(w_0)
$$
This results in 
$$
T_n(w)= E(w_0) + (w - w_0)E'(w_0) + \frac{(w - w_0)^2}{2}E''(w_0)
$$
In here, w$_0$ is the current value of the coefficient and w is the new value after adjustment, in here we are approximating E(w) around the current value of the coefficient.

**Objective**: find a new value w that leads to the minimum or maximum of the quadratic approximation in an attempt to move to the minimum or maximum of the original function.

### Weight Update Rule

Solving for 

$$
\frac{d}{dw}() E(w_0) + (w - w_0)E'(w_0) + \frac{(w - w_0)^2}{2}E''(w_0))=0
$$
will result in
$$
w = w - \frac{E'(w)}{E''(w)}
$$

## Newton-Raphson Method: Multivariate Case

### Second Order Partial Derivatives and The Hessian

The Hessian contains all second-order partial derivatives, capturing the curvature along multiple direction.

$$
H(f(x)) = H_f(x) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_0^2} & \frac{\partial^2 f}{\partial x_0 \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_0 \partial x_d} \\
\frac{\partial^2 f}{\partial x_1 \partial x_0} & \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_0} & \frac{\partial^2 f}{\partial x_d \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_d^2}
\end{pmatrix}
$$
### Weight Update rule

- Uni-variate update rule
$$
w = w - \frac{E'(w)}{E''(w)}
$$
- Multi-variate update rule
$$
\mathbf{w} = \mathbf{w}- H_E^{-1}(\mathbf{w}) \nabla E(\mathbf{w})
$$

$H_E^{-1}(\mathbf{w})$ is the inverse of the Hessian at the old w and $\nabla E(\mathbf{w})$ is the gradient at the old w.

### Steepest vs Newton-Raphson Weight Update Rule on a Quadratic Function

![[Pasted image 20241108172645.png]]

### Weight Update Rule for Non-Quadratic Loss Functions

$$
\mathbf{w} = \mathbf{w}- H_E^{-1}(\mathbf{w}) \nabla E(\mathbf{w})
$$
![[Pasted image 20241108172912.png]]
will take us to the optimal of the quadratic approximation in a single step, needs to be applied iteratively because that's not the true loss function.


### For Logistic Regression — Iterative  Reweighed Least Squares

$$
\mathbf{w} = \mathbf{w}- H_E^{-1}(\mathbf{w}) \nabla E(\mathbf{w})
$$
$$
H_E(\mathbf{w})= \sum_{i=1}^N p(1 | \mathbf{x}^{(i)}, \mathbf{w}) (1 - p(1 | \mathbf{x}^{(i)}, \mathbf{w})) \mathbf{x}^{(i)} \mathbf{x}^{(i)^T}
$$
$$
\nabla E(\mathbf{w}) = \sum_{i=1}^N \left( p(1 | \mathbf{x}^{(i)}, \mathbf{w}) - y^{(i)} \right) \mathbf{x}^{(i)}
$$

