
## Types of ML

### Unsupervised Learning
There's no label, and we don't really know what we are training. We can see this in both in clustering and dimensional reduction.
- Create an internal representation of the input capturing pattern/structure in data.
- Clustering algorithm tries to detect similar groups.
- Learn interesting patterns from dataset with no labels
- Dimensionality reduction tries to simplify the data without losing too much information.
![[Pasted image 20241001181739.png]]
### Supervised Learning
Classification and Regression
- Correct out know for each training example
- Learn a function that maps an input to an output based on examples.
- Classification learns to predict discrete values(class labels.
- Regression learns to predict continuous values.
![[Pasted image 20241001181713.png]]

### Reinforcement Learning
- An agent learns how to interact with an environment more efficiently
- In each step
	- The agent receives observation which give agent information about the state.
- Agent receives a reward periodically.
![[Pasted image 20241001181833.png]]

## Training and testing
### Example:
- Input: House information
- Output: Prince
- Aim: Find model to predict price
- Training dataset: a sequence of {features,price}
- Prediction/testing: given the information does it predicts price correctly?
- Performance: difference between predicted price and true price.
### Loss function
Scores how far off a prediction is from desired "target" output.

### Underfitting and overfitting

- Under fits when training performance is poor
- Model overfits when training performance is good but the performance is poor
![[Pasted image 20241001182823.png]]

Overfitting means that the model performs good on training data, but does not generalize the testing data, underfitting is the opposite, occurs when model is to simple.

In general the training error decreases as we add complexity to our model with additional features or more complex prediction mechanisms,
Test error, decreases up to a certain amount of complexity then increases again as the model complexity increases again. 

`#Note: Important to find optimum spot. `

### Data split

It is common to use 80% for training and 20% for testing.

