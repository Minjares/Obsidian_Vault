
### Origin

Comes from the problem of overfitting, basically when we overfill to the training data so well that we are also including noise.

SVM try to prevent overfitting by, maximizing the margin.

### Problem

Overfitting 

#### Solution

Slack Variables

![[Pasted image 20241202201409.png]]

The variable allow examples to be misclassified, and within the margin. Slack variable tell us how much an example can violate the margin.

If the variable is zero, the example is on the margin or outside of it, if its (0-1), its inside of the margin. If its one is on the decision boundary and more than one means that its misclassified.

##### Effect

![[Pasted image 20241202201903.png]]


There are some changes to the margin too.

![[Pasted image 20241202202729.png]]

The margin is 1 divided by the norm of w

###### Optimization Function
![[Pasted image 20241202202934.png]]

Now we need to find also the value of all slack variables. We are trying, to minimize the value of the slack, meaning the amount of violations of he margin.

The C, is the importance we give to minimizing Slack variables, small value place importance to maximizing margin, larger C more emphasis in correct classification, since we are minimizing the slack. Smaller values of C, allow less overfiitting, since we allow more Slack.

![[Pasted image 20241202203500.png]]
Larger C has more unique decision boundaries, as seen on the isolated blue dot on the left.

![[Pasted image 20241202204224.png]]

### Making Predictions using Soft SVM
![[Pasted image 20241202204330.png]]
Same as normal SVM
#### From Primal to Dual
##### Using Lagrange Relaxation

![[Pasted image 20241202204516.png]]

The constraints need to be fixed to comply with origin $f(x)\leq0$  
![[Pasted image 20241202204714.png]]
Both $a$ and $\beta$  $\geq$  than 0

Lagrange Relaxation, is larger since we also need to include the Slack function.

##### Dual 
![[Pasted image 20241202205138.png]]

Its time to simplify the equation using KKT stationary.
![[Pasted image 20241202205646.png]]

![[Pasted image 20241202205914.png]]

betta can be easily replaced, using KKT stationary

![[Pasted image 20241202210028.png]]

Since we got rid of w, we need to make predictions with the following, 
![[Pasted image 20241202210953.png]]

### Difference with Hard SVM

In hard svm when $y^nh(x^{n})=1$ meant that the point was exactly on the margin, but on soft SVM, it doesnt necesarilly means that.
![[Pasted image 20241202211140.png]]

This complicates things a little, on soft SVM, support vectors can be on the margin, within the margin or on the wrong side of the margin, 
![[Pasted image 20241202211636.png]]

### Calculation of b

#### Previously
![[Pasted image 20241202211856.png]]
#### Now

Since there is a chance of not having a svm on the margin, and since b is highly dependant of the slack, we can choose any b.


