# Details
### Lecturers
*Leandro Minku*
*Jian Liu*
*Dr. Jiangu*

### Lectures and Tutorials
There's a tutorial once per week, and each session is a 50 minutes session.
**Tutorial date is on Canvas**

## Quizzes and Exams
20% two quizzes, 10% percent each
80% Exam, on May/June

# Motivation for machine learning

- Many types of data that need to be analyzed, processed and used.
- Use of data to automatically predict and prevent manual loss of time
# Definition
- Approaches that help us define patterns in data.
- Field of study that gives computers the ability to learn without being explicitly programmed. *Wrong because a machine that has not been programmed can't do anything*
- Learn from experience **E** with respect from some task **T** and some performance measure **P**, if its performance on **T**, as measured by **P**, improves with experience **E**.

# Artificial Intelligence
- Rusell and Norvig,
	- AI is the area of CS which studies rational agents.
	- **Rational Agents** are computer programs that perceive and take actions that maximize their potential outcomes.

# Supervised Learning
Learns a mapping from inputs to outputs, given a training set of input-output pairs, given a training set of input-outputs pair.

### Active learning
Where the data set is acquired through queries that we make.

Basically having all inputs and outputs all at once.

### Online learning

Online learning, where the data set is given to the algorithm one example at a time. This happens when we have streaming data that the algorithm has to process 'on the run'.

## Components
- Unknown target function  $$ f : X \to y $$
- Training set composed of n examples, inputs and outputs.
$$  T = \{(x^i, y^i)\}_{i=1}^{n}$$

- Hypothesis Set, space of all possible function for any value of a and b
$$ h(x) a^Tx + b $$
The learning Algorithm outputs a final hypothesis which is a good approximation of the unknown target function. It will search for the hypothesis that it believes best approximates the function to the Training example.

- There is also an unknown Target Distribution, and an unknown input distribution. (Need to investigate more.)

## Problem
Given a set of training examples, where the training set are drawn independently and identically distributed from a fixed albeit unknown joint probability distribution 
$$ p(x,y) = p(y|x)p(x) $$

# The input Space x

- d-dimensional space, where each dimension can usually be:
	- Numeric
	- Ordinal:
		- Expertise(low, median high)
	- Categorical
		- Car(fiat,Toyota, etc)
Possible to have dimensions of different times, but many ML models assume that all of inputs are numeric.
`#Note: Convert other dimension categories that are not numeric, to numbers, using maybe an embedding model. Theres many ways to convert them to numbers. `

# Regression vs Classification
### Regression

Input to output, or to estimate output given inputs.
### Classification

Many inputs, to find a function that separates classes given different inputs.
