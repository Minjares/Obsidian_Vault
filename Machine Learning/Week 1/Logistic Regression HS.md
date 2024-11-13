
# Hypothesis Set

Questions that need to be solved?
- What kind of function can logistic regression model?
- What parameters need to be learned?

## General Idea

Despite the name,==Logistic Regression== is an approach for classification problems. In LR we we will model the probability of an instance to belong to a given class as linear combination of the inputs.

### Focus

We will focus on binary classification problems, problems where "y" is a set containing two possible categorical classes .

$$  y = \{c_0,c_1\} = {0,1} $$
We assume numeric inputs.
### Need for a the logit function

Consider that we wish to model *P*(y = 1|x) = P(1|x) as a function of the input variables. 

$$ p(1|x,w) = w_0 + w_1x_1 + ... + w_dx_d $$
$$  p(1|x,w) = w_0 = w_1x_1 + ... + w_dx_d, where x_0 = 1 $$
$$ p(1|x,w) = w^Tx $$
$$ p_1 = w^Tx $$
With this we would be able to deal with classification problems being p$_1$ $\geq$ 0.5
and 0 otherwise. Important to note that p$_1$ should be in [0,1].

We might think using a natural logarithm as a possible solution, but this would limit p$_1$ from 0 to infinite or vice versa. That's why there's a need for a logit function.

#### Solution

Create a model:
$$ logit(p_1) = ln(\frac{p_1}{1-p_1})$$
This model allows us to map from \[0,1] to \[-$\infty$,$\infty$],  since logit(p$_1$) is in \[-$\infty$,$\infty$],
and w$^T$x is in \[-$\infty$,$\infty$] we can model logit(p$_1$) = w$^T$x 


### The Odds


$$ logit(p_1) = ln(\frac{p_1}{1-p_1}) = w^Tx$$
The odds are the ratio of probabilities of two possible outcomes

$$ o_1 = \frac{p_1}{p_0} = \frac{p_1}{1-p_1} $$
So if o$_1$ $\geq$  1, we can say that the class is 1
otherwise the class is 0.

## Logit

In simple words the logit is the logarithm of the odds.

$$ logit(p_1) = ln(\frac{p_1}{1-p_1})$$
$$  
\text{If } \text{logit}(p_1) = \mathbf{w}^T \mathbf{x} \geq 0, \text{ predict class 1.}

$$
$$
\text{If } \text{logit}(p_1) = \mathbf{w}^T \mathbf{x} < 0, \text{ predict class 0.}

$$
Important to remember that the **w** are parameters of the function needed to be learn based on training examples

## A Linear Classifier

The equation **w**$^T$**x**  = 0 is the equation of the hyperplane in the input space.
Lets use a 2-dimensional input space, this is the equation of the line.

### Let's get p$_1$ and p$_2$

If we solve logit for p$_1$ we get:
$$  
{logit}(p_1) = \mathbf{w}^T \mathbf{x}
$$
$$

p_1 =\frac{e^{\mathbf{w}^T \mathbf{x}}}{1+e^{\mathbf{w}^T \mathbf{x}}}

$$
$$

p_0 = 1- p_1 =\frac{1}{1+e^{\mathbf{w}^T \mathbf{x}}}

$$
With this we can get that p$_1$ $\geq$ 0.5 -> class 1 

*Illustration graph*
![[Pasted image 20241107230022.png]]

### Relationship between the distance to the decision boundary and $p_1$

The larger **w**$^T$**x** the higher p$_1$, and the more negative **w**$^T$**x**, the smaller the p$_1$

![[Pasted image 20241107231128.png]]


### Hypothesis Set

![[Pasted image 20241107231441.png]]