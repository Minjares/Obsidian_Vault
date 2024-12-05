## How to solve the problem?

### Sequential minimal optimization
- Breaks the quadratic programming problem into smaller quadratic programming problems that can be solved analytically one at a time.
	-A subset of a
- Uses heuristics to decide which of these smaller problems to solve at each step.
	-Which  subset of a to solve

#### How many $a^{n}$ to solve

- Suppose we have **a** values for which the constrains are satisfied
- Suppose we pick $a^{(m)}$ to update, while keeping the others fixed.
- Would we be able to change the values of $a^{(m)}$ ?
Constraints:
![[Pasted image 20241202225259.png]]
- Any changes in $a^{(m)}$ would result in the summation constraint being violated.
![[Pasted image 20241202225753.png]]
> What is the smallest number of Lagrange multipliers that can be updated in each step such that we can remain satisfying this constraint?

[[5c-SMO.pdf#page=6&selection=44,0,46,72|5c-SMO, page 6]]

The smallest number would be two.

### Sequential Minimal Optimization (SMO)

![[Pasted image 20241202230108.png]]

#### Initialize a

- Initialize to any value that satisfies the constraints.
![[Pasted image 20241202230219.png]]

- Which value of the langrange multiplier can satisfy the constraints of all the training examples?
- The answer is 0
$$
a^{(n)}=0
$$
#### $\text{Optimising } \tilde{L}(\mathbf{a}) \text{ With Respect to } a^{(i)} \text{ and } a^{(j)} \text{ While Dealing With Constraints}$


![[Pasted image 20241202231101.png]]

We replace the sum to a variable, since we are taking it as a constant, its a linear function.
![[Pasted image 20241202231248.png]]

Minimum variables for both are 0, and maximum is C

- When changing one variable, the other one should also change, for example if ai is changed aj also need to be changed accordingly, following the constraints.
![[Pasted image 20241202231616.png]]

Values that can be assumed depending on the output
![[Pasted image 20241202232519.png]]

#### Writing $a^{(i)}$ as a function of $a^{(j)}$ 

![[Pasted image 20241202232815.png]]

Now we are maximizing $a^{(j)}$ 
![[Pasted image 20241202232857.png]]

#### Optimizing the value for $a^{(j)}$ 

We are trying to solve a(j) in close form
![[Pasted image 20241202233000.png]]

Where the Error is given by
$$
E^{(i)}=h(\mathbf{x}^{(i)})-y^{(i)}
$$

#### Clipping the value of $a^{(j)}$ 

The update rule may lead to violations in the box constrains.

rules for Clipping

![[Pasted image 20241202233432.png]]

With this, we get the value of $a^{(j)}$ 

##### Obtaining $a^{(i)}$ 
![[Pasted image 20241202233626.png]]


#### Select a pair of Lagrange multipliers to update next

##### KKT Conditions Euristics

![[Pasted image 20241202234411.png]]

#### How to select $a^{(i)}$ 
![[Pasted image 20241202235254.png]]

#### How to select $a^{(j)}$

We follow the following strategies until a positive improvement is viewed.
![[Pasted image 20241202235808.png]]

Insead of using that formula to calculate the new aj, we go with the largest absolute value for the difference of Errors.

