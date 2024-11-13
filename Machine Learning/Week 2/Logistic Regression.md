## Loss function and Gradient Descent

### It's time to learn **w**

$$
  
\text{If } \text{logit}(p_1) = \mathbf{w}^T \mathbf{x} \geq 0, \text{ predict class 1.}

$$
$$
\text{If } \text{logit}(p_1) = \mathbf{w}^T \mathbf{x} < 0, \text{ predict class 0.}
$$
Given a training set 

$$
\mathcal{T} = \left\{ (\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots, (\mathbf{x}^{(N)}, y^{(N)}) \right\}

$$
The most reasonable values for $w$ are the ones for which the "probability" of the observed examples is the largest.

It's important to find the values of ***w*** that maximize the probability quantity of $p_1=p(1|x,w),p_0=p(0|x,w)$ 

### Likelihood

How likely would we get the output $y$ for the input $\mathbf{x}$ for  our training examples if the target distribution is really as the formula as above.
$$
\text{For each example } (\mathbf{x}^{(i)}, y^{(i)}) \in \mathcal{T}, \text{ this is captured as } 
$$
$$
p_{y^{(i)}} = p(y^{(i)} | \mathbf{x}^{(i)}, \mathbf{w}), \text{ where } y^{(i)} \in \{0, 1\} \quad \text{(conditional likelihood)}
$$
$$
\text{As examples are drawn i.i.d., jointly we have } \prod_{i=1}^N p_{y^{(i)}}. \quad \text{(joint conditional likelihood)}

$$
The term probability is usually when we assume the model's parameters are reliable. The term likelihood is for when we're trying to determine whether the parameters in a model are good given the data.

### Notation and Maximum Likelihood Estimation

$$
\prod_{i=1}^N p_{y^{(i)}} = \prod_{i=1}^N p(y^{(i)}|x^{i},w) = p(y|\mathbf{X},w) = \mathcal{L}(w)
$$
$$
\mathcal{T} = \left\{ (\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots, (\mathbf{x}^{(N)}, y^{(N)}) \right\}
$$
Training set is mapped into two things, a matrix with all inputs and the a vector of the output for each classified input.
$$
\text{Design matrix: }
\mathbf{X} = \begin{pmatrix}
x_1^{(1)} & x_2^{(1)} & \dots & x_d^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \dots & x_d^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
x_1^{(N)} & x_2^{(N)} & \dots & x_d^{(N)} \\
\end{pmatrix}

\text{Vector of outputs: }
\mathbf{y} = \begin{pmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(N)} \\
\end{pmatrix}


$$
The problem here is the need to find **w** that maximizes the likelihood:
$$
\text{argmax }  \mathcal{L}(w)
$$
#### Log-Likelihood and Loss Function


$$
\mathcal{L}(w) = \prod_{i=1}^N p_{y^{(i)}}

\text{ can be numerically unstable due to its low digits}
$$

We solve this by adding a natural logarithm to it.

$$
ln(\mathcal{L}(w)) = ln\prod_{i=1}^N p_{y^{(i)}} = \sum_{i=1}^N \ln p_{y^{(i)}}

$$
This is equivalent to finding **w** that minimizes the loss argmin E(w)

$$
E(\mathbf{w}) = -ln(\mathcal{L}(\mathbf{w}))=-\sum_{i=1}^N \ln p_{y^{(i)}}
$$
#### Understanding the Loss


Learning  $\mathbf{w}$ can be achieved by finding the  $\mathbf{w}$  that minimises  $E(\mathbf{w})$,  calculated based on the training set.
$$

\arg \min_{\mathbf{w}} E(\mathbf{w})
$$
$$
E(\mathbf{w}) = -\ln(\mathcal{L}(\mathbf{w})) = - \sum_{i=1}^N \ln p_{y^{(i)}}
$$

![[Pasted image 20241108010307.png]]

#### Cross Entropy Loss
$$
E(\mathbf{w}) = -\ln(\mathcal{L}(\mathbf{w})) = - \sum_{i=1}^N \ln p_{y^{(i)}}
$$
equivalent to

$$
E(w) = -\sum_{i=1}^N y^{)i)}\ln p(1|x^{(i)},w)+(1-y^{(i)})\ln(1-p(1|x^{(i)},w)) 

$$

Is a measure of dissimilarity between to probability distributions, in here its used to measure the dissimilarity between the true target P(y|X) and learned distribution p(y|x,w),estimated based on the training examples.

## Solving the Optimization Problem
h s
### General Idea of Gradient Descent

Adjusts **w** iteratively in the direction that leads to the biggest decrease (steepest descent) in E(w)
![[Pasted image 20241108012321.png]]
*E(w) is not a quadratic function*
$$
E(\mathbf{w})
$$
$$
w_i = w_i - \eta \frac{\partial E}{\partial w_i} \quad \text{where } \eta > 0
$$

#### Adjusting w in the Direction that reduces $E(w)$
$$
\begin{pmatrix}
w_0 \\
w_1 \\
\vdots \\
w_d \\
\end{pmatrix} 
=
\begin{pmatrix}
w_0 \\
w_1 \\
\vdots \\
w_d \\
\end{pmatrix} 
-
\eta \begin{pmatrix}
\frac{\partial E}{\partial w_0} \\
\frac{\partial E}{\partial w_1} \\
\vdots \\
\frac{\partial E}{\partial w_d}
\end{pmatrix}
$$
$$
\mathbf{w} = \mathbf{w} - \eta \nabla E(\mathbf{w}) 
$$
That's where *gradient* comes from

#### Batch version

- Initialise **w** with zeros or random numbers near to zero
- Repeat for a given number of iterations  or until $\nabla$E(**w**) is a vector of zeros 
$$
\mathbf{w} = \mathbf{w} - \eta \nabla E(\mathbf{w}) 
$$
$\eta$ is the learning rate

#### Applying it to Logistic Regression(Batch Version)

$$
E(\mathbf{w}) = - \sum_{i=1}^N \left( y^{(i)} \ln p(1 | \mathbf{x}^{(i)}, \mathbf{w}) + (1 - y^{(i)}) \ln (1 - p(1 | \mathbf{x}^{(i)}, \mathbf{w})) \right)
$$
The partial derivative with respect to **w** gives us the following:
$$
\nabla E(\mathbf{w}) = \sum_{i=1}^N \left( p(1 | \mathbf{x}^{(i)}, \mathbf{w}) - y^{(i)} \right) \mathbf{x}^{(i)}

$$
### Steepest Descent

Changes coefficients of w in the direction that gives the steepest descent, the direction that causes the largest reduction in E(w)

### Effect of $\eta$

- Large values of $\eta$ may result in large jumps, lacking stability
- Small values of $\eta$ result in longer converge to the optimum
It's important to find an optimal value for $\eta$



Volver a revisar el paginas 88-92 del libro

