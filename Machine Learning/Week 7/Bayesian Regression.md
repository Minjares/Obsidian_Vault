
Used for noisy targets or in other words a noisy distribution.

## What we are trying to learn

An unknown target distribution given by an unknown target function, the input distribution can also be unknown. 

![[Pasted image 20241204220307.png]]

### Probabilistic Machine Learning

Key Differences: Frequentist vs Bayesian Paradigms

-Frequentist
- **Model parameters  $\theta$ Fixed values.
- **Computation:** Parameters are computed using an estimator (e.g., Maximum Likelihood Estimation, **MLE**).
- **Confidence in estimates:** Derived through multiple experiments or datasets (e.g., cross-validation).
-Bayesian
- **Model parameters  $\theta$:** Treated as random variables.
- **Dataset:** Only one observed dataset is used.
- **Uncertainty in  $\theta$ :** Expressed as a probability distribution over  $\theta$.

### Probabilistic Machine Learning: Bayes' Law

- Using the sum and product rules of probability for discrete values of $\theta$ (or events):
  $$
  P(\theta \mid X) = \frac{P(X \mid \theta) P(\theta)}{P(X)}
  $$
  - **Posterior**: $P(\theta \mid X)$ — The updated probability of $\theta$ after observing data $X$.
  - **Likelihood**: $P(X \mid \theta)$ — The probability of observing $X$ given $\theta$.
  - **Prior**: $P(\theta)$ — The initial belief about $\theta$, before observing data.
  - **Marginal Probability**: $P(X)$ — The total probability of observing the data $X$, summed over all possible values of $\theta$:
    $$
    P(X) = \sum_{\theta} P(X \mid \theta) P(\theta)
    $$

- For continuous variables (i.e., all possible values of $\theta$), we can do the same:
  $$
  p(\theta \mid X) = \frac{p(X \mid \theta)p(\theta)}{p(X)}
  $$
  - **Posterior**: $p(\theta \mid X)$
  - **Likelihood**: $p(X \mid \theta)$
  - **Prior**: $p(\theta)$

- The marginal probability density function $p(X)$ is:
  $$
  p(X) = \int p(X \mid \theta)p(\theta) d\theta
  $$

- **Key takeaway**:
  $$
  \text{Posterior} \propto \text{Likelihood} \times \text{Prior}
  $$

### Linear Regression: Probabilistic View

- Consider the linear relationship between the target $y$ and predictor $x$:
  $$
  y = \hat{y}(x, \omega) + \epsilon
  $$
  - Where $\epsilon$ is some **unexplained noise**.
  - $\hat{y}(\cdot)$ is a linear combination of basis functions.

- Assume $\epsilon$ is a univariate Gaussian variable with:
  - Zero mean.
  - Precision $\beta = \frac{1}{\sigma^2}$ (where $\sigma^2$ is the variance).
  - $\beta$ is used for convenience of notation.

- The probability distribution of the target $y$, conditioned on $x$ and $\omega$, is given by:
  $$
  p(y \mid x, \omega, \beta) = \mathcal{N}(y \mid \hat{y}(x, \omega), \beta^{-1})
  $$

### Univariate Gaussian Distribution

- **Occurs often in Nature:** Examples include height, IQ.
- **Central Limit Theorem:** Fixed-length sums of variables from any distribution approximately follow a Gaussian distribution.
- **Parameters:**
  - $\mu$: Position of the center (mean).
  - $\sigma$: Width of the distribution (standard deviation).

- **Probability Density Function (PDF):**
  $$
  \mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{1}{2\sigma^2}(x - \mu)^2}
  $$

- **Properties:**
  $$
  \mathbb{E}[\mathcal{N}(x \mid \mu, \sigma^2)] = \mu
  $$
  $$
  \text{Var}[\mathcal{N}(x \mid \mu, \sigma^2)] = \sigma^2
  $$
### Multivariate Gaussian Distribution

- **For an $N$-dimensional vector $\mathbf{x}$**, the multivariate Gaussian distribution is given by:
  $$
  \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
  \frac{1}{(2\pi)^{N/2}} 
  \frac{1}{|\det(\boldsymbol{\Sigma})|^{1/2}} 
  \exp\left\{ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\}
  $$

- **Parameters:**
  - $\boldsymbol{\mu}$: An $N$-dimensional mean vector.
  - $\boldsymbol{\Sigma}$: An $N \times N$ covariance matrix.

- **Example: Bivariate (2D) Gaussian**:
  $$
  \boldsymbol{\Sigma} = 
  \begin{bmatrix}
  \sigma_{xx}^2 & \sigma_{xy} \\
  \sigma_{yx} & \sigma_{yy}^2
  \end{bmatrix}
  $$
### MLE (Maximum Likelihood Estimation)

- **Parameters to estimate** when fitting a multivariate Gaussian to data are:
  - $\boldsymbol{\mu}$ (mean vector)
  - $\boldsymbol{\Sigma}$ (covariance matrix)

- **Deriving MLE for $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$:**
  - Given a dataset $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)^\top$ of $N$ i.i.d. data points, where each $\mathbf{x}_i$ is a row in the matrix.

  - **Likelihood Function:**
    Using the product rule, the likelihood function is defined as:
    $$
    p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \prod_{i=1}^N p(\mathbf{x}_i \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})
    $$

  - **For a single data point $\mathbf{x}_i$:**
    The probability density is given by the multivariate Gaussian:
    $$
    p(\mathbf{x}_i \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
    \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
    \frac{1}{(2\pi)^{N/2}} 
    \frac{1}{|\det(\boldsymbol{\Sigma})|^{1/2}} 
    \exp\left\{ -\frac{1}{2} (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu}) \right\}.
    $$

- **To derive MLE for $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$:**
  - We need to **maximize the likelihood function** with respect to each parameter.

- **Solve the following equations:**
  $$
  \frac{\partial}{\partial \boldsymbol{\mu}} p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) \Big|_{\boldsymbol{\mu} = \boldsymbol{\mu}_{ML}} = 0
  $$
  $$
  \frac{\partial}{\partial \boldsymbol{\Sigma}} p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) \Big|_{\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_{ML}} = 0
  $$

- **Log-likelihood Function ($\mathcal{L}$):**
  - To simplify computations, we work with the log-likelihood function:
    $$
    \mathcal{L} = \ln p(\mathbf{X} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})
    $$
  - Substituting the multivariate Gaussian likelihood:
    $$
    \mathcal{L} = -\frac{ND}{2} \ln(2\pi) 
    - \frac{N}{2} \ln |\det(\boldsymbol{\Sigma})| 
    - \frac{1}{2} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
    $$
    - Where:
      - $N$: Number of data points.
      - $D$: Dimensionality of the data.
      - $\boldsymbol{\Sigma}$: Covariance matrix.

- **Considering just the terms in $\mathcal{L}$ dependent on $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$:**

  $$
  \mathcal{L}(\boldsymbol{\mu}) = \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
  $$

  $$
  \mathcal{L}(\boldsymbol{\Sigma}) = -\frac{N}{2} \ln \det(\boldsymbol{\Sigma}) - \frac{1}{2} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
  $$

- **Solving** $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = 0$ **and** $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}} = 0$ **gives:**

  $$
  \boldsymbol{\mu}_{ML} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i
  $$

  $$
  \boldsymbol{\Sigma}_{ML} = \frac{1}{N} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^\top
  $$
### Linear Regression: MLE & OLS

- **Given a dataset** $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^N$ and targets $\mathbf{y} = \{y_i\}_{i=1}^N$, assuming data are **i.i.d.**

- **Likelihood Function**:
  We can formulate the likelihood function as a function of $\boldsymbol{\omega}$ and $\beta$:
  $$
  p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta) = \prod_{i=1}^N \mathcal{N}(y_i \mid \boldsymbol{\omega}^\top \phi(\mathbf{x}_i), \beta^{-1})
  $$

  - **Using the previous result**:
    $$
    \hat{y}(\mathbf{x}_i, \boldsymbol{\omega}) = \boldsymbol{\omega}^\top \phi(\mathbf{x}_i)
    $$

  - This form matches the one we’ve seen for **MLE of multivariate Gaussians**.

- **Log-Likelihood Function**:
  The log-likelihood $\mathcal{L}$ is expressed as:
  $$
  \mathcal{L} = \ln p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta)
  $$
  Expanding it:
  $$
  \mathcal{L} = -\frac{ND}{2} \ln(2\pi) 
  - \frac{N}{2} \ln \beta 
  - \frac{N}{2} 
  - \frac{\beta}{2} \sum_{i=1}^N (y_i - \boldsymbol{\omega}^\top \phi(\mathbf{x}_i))^2
  $$

- **Where**:
  - $\mathbf{X}$: Input data.
  - $\mathbf{y}$: Target values.
  - $\phi(\mathbf{x}_i)$: Basis function for input $\mathbf{x}_i$.
  - $\boldsymbol{\omega}$: Weight parameters.
  - $\beta$: Precision parameter (inverse of variance).


- **Looking at the quadratic term in $\mathcal{L}$:**
  $$
  \mathcal{L} = \ln p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta) = -\frac{ND}{2} \ln(2\pi) - \frac{N}{2} \ln \beta - \frac{\beta}{2} R(\boldsymbol{\omega})
  $$

- **Residual Term** $R(\boldsymbol{\omega})$:
  $$
  R(\boldsymbol{\omega}) = \sum_{i=1}^N (y_i - \boldsymbol{\omega}^\top \phi(\mathbf{x}_i))^2 = \sum_{i=1}^N (y_i - \hat{y}(\mathbf{x}_i, \boldsymbol{\omega}))^2
  $$

- **Key Insight**:
  - This is the **same** as the sum of squared residuals in OLS (Ordinary Least Squares)!

- **Optimization Relationship**:
  - $\boldsymbol{\omega}$ which **maximizes** $\mathcal{L}$ also **minimizes** $R(\boldsymbol{\omega})$:
    $$
    \arg \max_{\boldsymbol{\omega}} \mathcal{L} \equiv \arg \min_{\boldsymbol{\omega}} (-\mathcal{L}) = \arg \min_{\boldsymbol{\omega}} \sum_{i=1}^N (y_i - \hat{y}(\mathbf{x}_i, \boldsymbol{\omega}))^2 = \arg \min_{\boldsymbol{\omega}} R(\boldsymbol{\omega})
    $$

- **MLE Solution**:
  $$
  \boldsymbol{\omega}_{MLE} = (\Phi^\top \Phi)^{-1} \Phi^\top \mathbf{y}
  $$

- **OLS Solution**:
  $$
  \boldsymbol{\omega}_{OLS} = (\Phi^\top \Phi)^{-1} \Phi^\top \mathbf{y}
  $$

- **Conclusion**:
  - Under the **additive Gaussian noise assumption**, **MLE = OLS**.

So what's the difference?
![[Pasted image 20241204224548.png]]
There is an assumption needed for MLE

## Linear Regression: Bayesian Perspective 

### Case 1: Exact Fit with Two Predictors
- **Linear regression with a polynomial of degree 1**:
  $$
  \hat{y}_i = \omega_0 + \omega_1 x_i
  $$
- **Two predictors and targets**:
  $\{(x_i, y_i)\}_{i=1,2}$
  - The line can be fit exactly.

- **Given two unknowns and two observations**:
  - We can determine the slope $\omega_1$ and intercept $\omega_0$ using the observations:
    $$
    \omega_1 = \frac{y_2 - y_1}{x_2 - x_1}
    $$
  - Then estimate $\omega_0$.

---

### Case 2: Overdetermined System
- **More than two points**:
  - The system is overdetermined.
  - We can use:
    - **OLS** to minimize squared residuals.
    - Assume a **noise model** to fit the data.

- **System of linear equations**:
  $$
  y_i = \omega_0 + \omega_1 x_i + \epsilon_i
  $$
  - Where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (Gaussian noise).
  - This leads to:
    $$
    \boldsymbol{\omega}_{MLE} \equiv \boldsymbol{\omega}_{OLS}
    $$

---

### Case 3: Underdetermined System
- **Single observation**:
  - With just one observation $(x_1, y_1)$:
    - Two unknowns $\omega_0$ and $\omega_1$.
    - One observation.

- **Linear equation**:
  $$
  y_1 = \omega_0 + \omega_1 x_1 + \epsilon_1
  $$
  - The system is underdetermined.
  - We cannot uniquely solve for $\omega_0$ and $\omega_1$.
  - **Infinite possible solutions**.

- **Visualizing a Family of Solutions**:
  - A family of $\omega_0$ values can represent different valid solutions:
    - Example figures (a), (b), (c), (d) show variations of $\omega_0$:
      - $\omega_0 = 1.2, 0.4, 1.8, 0.8$.



## Introduction to Bayesian Inference
- **Assumption**: We assume a distribution for the unknown parameter $\omega_0$.
- This forms the basis of **Bayesian Inference**.
- **Dataset**: Given $\mathbf{X} = \{x_i\}_{i=1}^N$ and $\mathbf{y} = \{y_i\}_{i=1}^N$, assuming data are **i.i.d.**
- **Likelihood Function**:
  $$
  p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta) = \prod_{i=1}^N \mathcal{N}(y_i \mid \boldsymbol{\omega}^\top \phi(x_i), \sigma^2)
  $$
- In **MLE**, we maximized the likelihood with respect to $\boldsymbol{\omega}$.
- In the **Bayesian perspective**:
  - We introduce **a prior belief** about $\boldsymbol{\omega}$.
  - Instead of a fixed $\boldsymbol{\omega}$, we integrate over all plausible $\boldsymbol{\omega}$.

---

## Adding a Prior Distribution
- **Prior**: Represents our belief about $\boldsymbol{\omega}$ **before seeing the data**.
- Assume a **Gaussian prior**:
  $$
  p(\omega_0) \sim \mathcal{N}(0, \alpha)
  $$
- **Posterior Distribution**:
  Given $N$ observations $\{x_i, y_i\}$, the posterior for $\omega_0$ is:
  $$
  p(\omega_0 \mid \mathbf{y}, \mathbf{x}, \boldsymbol{\omega_1}, \sigma^2) \propto \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \omega_1 x_i - \omega_0)^2 - \frac{\omega_0^2}{2\alpha}\right)
  $$
- **Log-posterior**:
  $$
  \ln p(\omega_0 \mid \mathbf{y}, \mathbf{x}, \boldsymbol{\omega_1}, \sigma^2) = -\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \omega_1 x_i - \omega_0)^2 - \frac{\omega_0^2}{2\alpha} + \text{const}
  $$

---

## Completing the Square
- By completing the square, we find:
  $$
  \ln p(\omega_0 \mid \mathbf{y}, \mathbf{x}, \boldsymbol{\omega_1}, \sigma^2) = -\frac{1}{2\sigma^2_{post}} (\omega_0 - \mu_{post})^2 + \text{const}
  $$
- **Posterior Distribution**:
  $$
  p(\omega_0 \mid \mathbf{y}, \mathbf{x}, \boldsymbol{\omega_1}, \sigma^2) \sim \mathcal{N}(\mu_{post}, \sigma^2_{post})
  $$
- Where:
  - $\mu_{post} = \frac{N\alpha}{N\alpha + \sigma^2} \mu_{ML}$
  - $\sigma^2_{post} = \frac{\sigma^2 \alpha}{N\alpha + \sigma^2}$

---

## Extending to All Weights
- Previously, we assumed a prior for $\omega_0$. Now, consider a **prior over all weights** $\boldsymbol{\omega}$.
- **Likelihood Function**:
  $$
  p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta) = \prod_{i=1}^N \mathcal{N}(y_i \mid \boldsymbol{\omega}^\top \phi(x_i), \beta^{-1})
  $$
- **Prior**:
  $$
  p(\boldsymbol{\omega}) = \mathcal{N}(\boldsymbol{\omega} \mid 0, \alpha^{-1} \mathbf{I})
  $$
- **Posterior Distribution**:
  $$
  p(\boldsymbol{\omega} \mid \mathbf{y}, \mathbf{X}, \beta) \propto p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\omega}, \beta) p(\boldsymbol{\omega})
  $$

---

## Completing the Square for Posterior
- If $p(\boldsymbol{\omega})$ is a **conjugate prior**, then $p(\boldsymbol{\omega} \mid \mathbf{y}, \mathbf{X})$ is Gaussian.
- **Log-posterior**:
  $$
  \ln p(\boldsymbol{\omega} \mid \mathbf{y}, \mathbf{X}) = -\frac{\beta}{2} \sum_{i=1}^N (y_i - \boldsymbol{\omega}^\top \phi(x_i))^2 - \frac{\alpha}{2} \boldsymbol{\omega}^\top \boldsymbol{\omega} + \text{const}
  $$
- Completing the square gives:
  $$
  p(\boldsymbol{\omega} \mid \mathbf{y}, \mathbf{X}) \sim \mathcal{N}(\mathbf{m}_{post}, \mathbf{S}_{post})
  $$
  Where:
  - $\mathbf{m}_{post} = \beta \mathbf{S}_{post} \Phi^\top \mathbf{y}$
  - $\mathbf{S}_{post}^{-1} = \alpha \mathbf{I} + \beta \Phi^\top \Phi$

---

## Predictive Distribution
- **MLE for Linear Regression**:
  $$
  \boldsymbol{\omega}_{MLE} = \Phi^\dagger
  $$
- Predictive distribution for $y$ is:
  $$
  \mathcal{N}(y \mid (\omega_0 + \omega_1 x), \beta^{-1})
  $$
- **Uncertainty Behavior**:
  - Model uncertainty decreases as the number of observations increases.
  - Predictive distribution narrows with more data.

