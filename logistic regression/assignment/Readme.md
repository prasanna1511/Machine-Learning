## Logistic Regression: Binary and Multi-Class Classification

- Overview
- Mathematical Background
- Binary Logistic Regression
- Multi-Class Logistic Regression

## Overview

- #### Logistic regression is a classification algorithm in machine learning

Binary Logistic Regression: 
Sigmoid function will approximate the given input to either 0 or 1

Multi-Class Logistic Regression
Used for classifying instances into more than two classes using the softmax function. Here the probabilty of each input is calculated indivaidually and then approximated to 1.



## Mathematical Equations

### Binary Logistic Regression

#### Sigmoid Function

The **sigmoid** function is used to model the probability of a binary outcome. It is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### Gradient of the Sigmoid Function

The gradient of the binary cross-entropy loss with respect to each parameter \(\theta_j\) is given by:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

### Multi-Class Logistic Regression

#### Softmax Function

The **softmax** function is used to model the probability of multiple classes. For class \(j\), the softmax function is defined as:

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$



#### Gradient of the Softmax Function

The gradient of the cross-entropy loss with respect to each parameter \(\theta_j\) is given by:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})_j - y^{(i)}_j) x^{(i)}
$$
