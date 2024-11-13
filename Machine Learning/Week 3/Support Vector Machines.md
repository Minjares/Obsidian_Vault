
## General idea

- The perpendicular distance between the decision boundary and the closest of the training examples on the left or right is too small, this is called **margin** $\gamma$.

- The decision boundary can be chosen as to maximize the margin, which can help to avoid overfitting.
- Training examples that are exactly on the margin are called **support vectors**.

## Perpendicular distance from a Point $\mathbf{x}^{(n)}$ to a Hyperplane $h(\mathbf{x})=0$ 

$$
h(\mathbf{x})=w^T x^{(n)} + b \quad {dist}(h, x^{(n)}) = \frac{h(\mathbf{x}^{(n)})}{\|w\|}

$$
where 

$$
||\mathbf{w}|| = \sqrt{\mathbf{w^{T}w}}
$$
is the Euclidean norm or the length of the vector 

- It's important to find the $\mathbf{w}$ and b that maximizes the margin between the hyperplane and the closes training example.

$$
\min_n dist(h,\mathbf{x}^{(n)})
$$
- Optimization objective:  
$$
\arg \max_{\mathbf{w}, b} \left\{ \min_n \, \textcolor{red}{\mathrm{dist}(h, \mathbf{x}^{(n)})} \right\}
$$
- Constraint: all training examples must be classified correctly

$$
y^{(n)}h(\mathbf{x}^{(n)}) > 0
$$

## Optimization Problem

Find w and b that maximize the margin while all training examples are correctly classified.

### Rewriting the problem in a simpler format

$$
\arg\min_{w, b} \left\{ \frac{1}{2} \|w\|^2 \right\}

$$
Subject to:
$$
y^{(n)}h(\mathbf{x}^{(n)})  \geq 1, \, \forall (\mathbf{x}^{(n)}, y^{(n)}) \in \mathcal{T}

$$

## Maximum Margin Classifier

![[Pasted image 20241113031727.png]]
## Maximum Margin Classifier for Nonlinear Problems

![[Pasted image 20241113031802.png]]