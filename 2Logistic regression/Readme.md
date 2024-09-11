
#### 2. **Logistic Regression**

- **Type**: Supervised Learning (Classification)


The `LogisticRegression` and `MultiClassLogisticRegression` performs classification tasks. Logistic Regression is used for binary classification, while Multiclass Logistic Regression extends this approach to handle multiple classes. These models are useful for tasks object detection, image classification

### 1. Logistic Regression


**Logistic Regression** is a linear model used for binary classification problems. It predicts the probability that a given input belongs to a particular class using the logistic (sigmoid) function maps any input to either 0 or 1


### Sigmoid Function

The sigmoid function is used to model the probability of a binary outcome. It is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


### Binary Cross-Entropy Loss

The cost function for logistic regression is the binary cross-entropy loss, which measures the performance of the classification model. It is defined as:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right]
$$


### Gradient Descent

The model parameters \( \theta \) are updated using gradient descent. The update rule is:

$$
\theta_j = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}
$$


### Gradient of the Cost Function

The gradient of the cost function with respect to each parameter \( \theta_j \) is:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

## Multiclass Logistic Regression

**Type:** Supervised Learning (Classification)

Multiclass Logistic Regression extends logistic regression to handle multiple classes by using the softmax function.

### Softmax Function

The softmax function is used to model the probabilities of multiple classes. For class \( j \), it is defined as:

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$

Where:
-  z_j is the score for class \( j \).
  - K  is the number of classes.

### Gradient Calculation

The gradient of the cost function for multiclass logistic regression is computed for each class and is used to update the parameters:

$$
\text{gradient} = \frac{1}{m} \sum_{i=1}^{m} x^T \cdot (y_{\text{pred}} - y)
$$

Where:
- y_pred are the predicted probabilities for each class.
- y are the true labels.

### Evaluation

The accuracy of the model is computed by comparing the predicted classes with the true classes:

$$
\text{accuracy} = \left(\frac{\text{number of correct predictions}}{\text{total predictions}}\right) \times 100
$$


