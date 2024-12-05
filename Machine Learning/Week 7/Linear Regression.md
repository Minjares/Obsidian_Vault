## Info

- Let's say we have training data comprising $\{x_i, y_i\}_{i=1..N}$ pairs  
- $x_i$: BMI & $y_i$: cardiac outputs (ejection fraction)  
- We want to fit a polynomial of **degree 1** to the data such that  

$$\hat{y}_i = \omega_0 + \omega_1 x_i \tag{1}$$  

- Where, $\hat{y}_i$ is the **predicted value** for $y_i$  
- $\omega_0$: **intercept** and $\omega_1$: **slope**  
- So how do we estimate $\omega_0, \omega_1$?


## Ordinary Least Squares (OLS)

- First define the **residual** for a single data point $i$ as:  

$$r_i = (y_i - (\omega_0 + \omega_1 x_i)) = (y_i - \hat{y}_i) \tag{2}$$  

- Key idea in OLS - find $\omega_0, \omega_1$ that minimizes the **sum of squared residuals**:  

$$R(\omega) = \sum_{i=1}^{N} r_i^2 = \sum_{i=1}^{N} (y_i - \hat{y}_i(x_i, \omega))^2 \tag{3}$$  

- So how do we minimize $R(\omega)$?
### Minimization

- Setting derivatives of $R$ with respect to $\omega$ to 0: $$\frac{\partial R(\omega_0, \omega_1)}{\partial \omega_0} = 0$$ $$\frac{\partial R(\omega_0, \omega_1)}{\partial \omega_1} = 0$$ - This results in a linear system of two equations with two unknowns $(\omega_0, \omega_1)$: ### The Normal Equations: 1. For $\omega_1$ (slope): $$\sum_{i=1}^N x_i y_i = \omega_0 \sum_{i=1}^N x_i + \omega_1 \sum_{i=1}^N x_i^2$$ 2. For $\omega_0$ (intercept): $$\sum_{i=1}^N y_i = \omega_0 N + \omega_1 \sum_{i=1}^N x_i$$
- Expressing the system of equations in **matrix form**:

#### Matrix Form of the Normal Equations:

1. Standard form :  
   $$
   \begin{pmatrix}
   \sum_{i=1}^N x_i y_i \\
   \sum_{i=1}^N y_i
   \end{pmatrix}
   =
   \begin{pmatrix}
   \sum_{i=1}^N x_i & \sum_{i=1}^N x_i^2 \\
   N & \sum_{i=1}^N x_i
   \end{pmatrix}
   \begin{pmatrix}
   \omega_0 \\
   \omega_1
   \end{pmatrix}
   $$

2. Solving for $\omega_0, \omega_1$ :  
   $$
   \begin{pmatrix}
   \omega_0 \\
   \omega_1
   \end{pmatrix}
   =
   \begin{pmatrix}
   \sum_{i=1}^N x_i & \sum_{i=1}^N x_i^2 \\
   N & \sum_{i=1}^N x_i
   \end{pmatrix}^{-1}
   \begin{pmatrix}
   \sum_{i=1}^N x_i y_i \\
   \sum_{i=1}^N y_i
   \end{pmatrix}
   $$

- **Interpretation**:
  - The system solves for $\omega_0$ (intercept) and $\omega_1$ (slope) using matrix operations.
  - This is the **ordinary least squares (OLS) solution** for a degree-1 polynomial regression function.
#### Arbitrary 

- Similarly, for polynomials of **arbitrary degree**, we have:  

$$\hat{y}_i(x_i, \omega) = \sum_{j=0}^M \omega_j \phi_j(x_i) = \omega^T \phi(x_i) \tag{9a}$$  

$$
\omega = 
\begin{pmatrix}
\omega_0 \\
\omega_1 \\
\vdots \\
\omega_M
\end{pmatrix}
,
\quad
\phi = 
\begin{pmatrix}
\phi_0(x_i) \\
\phi_1(x_i) \\
\vdots \\
\phi_M(x_i)
\end{pmatrix}
\tag{9b}
$$  

- As before, $\omega$ is the weights vector; $\phi$ is the **basis functions** vector.  
- Each $\phi_j(x_i)$ is a **basis function**; $\phi_0(x_i) = 1$ is a dummy basis function for $\omega_0$ (intercept).  
- This form is generalisable to **other types of basis functions** as well.  

##### Design Matrix

- Fitting an $M$-degree polynomial requires estimation of $M + 1$ parameters in $\omega$.
- Using $\omega$, $\phi$ we can define $R(\omega)$ as:  

$$R(\omega) = \sum_{i=1}^N (y_i - \omega^T \phi(x_i))^2$$  

- As before, we can estimate $\omega_{OLS}$ by minimising $R(\omega)$ with respect to $\omega$:  

$$\frac{\partial R(\omega)}{\partial \omega} = \sum_{i=1}^N (y_i - \omega^T \phi(x_i)) \phi(x_i)^T = 0$$  

$$\sum_{i=1}^N y_i \phi^T(x_i) = \omega^T \left(\sum_{i=1}^N \phi(x_i) \phi^T(x_i)\right)$$  

- Equation can be expressed in matrix-vector form by defining the **design matrix** $\Phi$.
- $\Phi$ is an $N \times (M+1)$ matrix where each $x_i$ has an associated basis function vector $\phi(x_i)$.
##### OLS: Estimating Regression Weights

- By defining the design matrix $\Phi$ as:  

$$
\Phi =
\begin{pmatrix}
\phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_M(x_1) \\
\phi_0(x_2) & \phi_1(x_2) & \cdots & \phi_M(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_0(x_N) & \phi_1(x_N) & \cdots & \phi_M(x_N)
\end{pmatrix}
$$  

- $\omega_{OLS}$ is given by solving:  

$$\Phi^T y = \Phi^T \Phi \omega$$
$$\text{normal solution}$$

- $\omega_{OLS}$:  

$$\omega_{OLS} = (\Phi^T \Phi)^{-1} \Phi^T y$$  

- $(\Phi^T \Phi)^{-1} \Phi^T$ is also known as the **Moore-Penrose pseudoinverse** of $\Phi$.

---

- The solution can be obtained by this set of linear equations (the normal equation).
- However, numerical inversion of matrices can be troublesome, especially if the matrix is large.
- Numerical inversion of matrices can be troublesome, especially if the matrix is large. OLS may not be best option then.
### Examples
![[Pasted image 20241203225908.png]]

How do we know how to choose?

#### Evaluating Regression Models


- **Coefficient of determination** or $R^2$:  

$$R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2},$$  
where $\bar{y}$ is the mean of the observed targets.

- **Mean Absolute Error (MAE)**:  

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$  

- **Mean Squared Error (MSE)**:  

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$  

- **Root Mean Squared Error (RMSE)**:  

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$  

- … and **several others**!


## Learning from data
