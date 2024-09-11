
#### 2. **Logistic Regression**

- **Type**: Supervised Learning (Classification)


The `LogisticRegression` and `MultiClassLogisticRegression` performs classification tasks. Logistic Regression is used for binary classification, while Multiclass Logistic Regression extends this approach to handle multiple classes. These models are useful for tasks object detection, image classification

### 1. Logistic Regression


**Logistic Regression** is a linear model used for binary classification problems. It predicts the probability that a given input belongs to a particular class using the logistic (sigmoid) function maps any input to either 0 or 1


#### Sigmoid Function

The **sigmoid** function is used to model the probability of a binary outcome.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$



- **Binary Cross-Entropy Loss**:

  $$
  J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right]
  $$


- **Gradient Descent**:

  The parameters are updated by taking steps proportional to the negative gradient of the loss function:

  $$
  \theta_j = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
  $$


- **Gradient of the Cost Function**:

  The gradient of the cost function with respect to each parameter \( \theta_j \) is given by:

  $$
  \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
  $$

### Multi-Class Logistic Regression

#### Softmax Function

The **softmax** function is used to model the probability of multiple classes. For class \(j\), the softmax function is defined as:

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$


#### Training the Model

- **Gradient Calculation**:

  The gradient for multiclass logistic regression is computed similarly to binary logistic regression but for each class. The gradient of the cost function is:

  $$
  \text{gradient} = \frac{1}{m} \sum_{i=1}^{m} x^T \cdot (y_{\text{pred}} - y)
  $$

  Where:
  -  y_pred  are the predicted probabilities for each class.
  - y  are true labels.

#### Evaluation

- **Accuracy**:

  The model's accuracy is calculated by comparing the predicted classes with the true classes:

  $$
  accuracy = \left(\frac{\text{number of correct predictions}}{\text{total predictions}}\right) \times 100
  $$

