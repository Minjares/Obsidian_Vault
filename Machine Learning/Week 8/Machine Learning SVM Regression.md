
### Recap

- To find the best fit, we minimize the sum of squared errors:
	- Least square estimation
$$
min \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{m} \left( y_i - (\mathbf{w} \cdot \mathbf{x}_i + b) \right)^2
$$

![[Pasted image 20241118110133.png]]

#### Evaluating Regression Models

1. Coefficient of Determination (R²)
$$
R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}
$$
Where:
- $\bar{y}$ is the mean of the observed targets.

2. Mean Absolute Error (MAE)
$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$
 3. Mean Squared Error (MSE)
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$
 4. Root Mean Squared Error (RMSE)
$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$
 Notes:
- There are several other metrics available for regression evaluation.

## Support Vector Regression

![[Pasted image 20241118110957.png]]

- Tube, is a margin, so the distance of arrows is reduced. No arrow for points inside the tube.
Will consider tube regression:
- Within tube, no error,
- Outside a tube, error by distance of the tube
Error measure:
### Error Measure

Formula:
$$
\text{err}(y, s) = \max(0, |s - y| - \epsilon)
$$



Conditions:
- If $|s - y| \leq \epsilon )$:  
  $$ 0 $$
- If  $|s - y| > \epsilon$ :  
  $$ |s - y| - \epsilon $$

---

##### Note:
This is usually called **$\epsilon$-insensitive error**, where $epsilon > 0$.

