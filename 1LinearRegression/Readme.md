#### 1. **Linear Regression**

- **Type**: Supervised Learning (Regression)

The `LinearRegression` class is designed to implement a simple linear regression model. The primary goal is to find the best-fitting line through a set of data points by calculating the parameters (intercept and slope) that minimize the error between the predicted and actual values.

1. **Parameter Calculation**:
   - The class calculates the intercept (`θ₀`) and slope (`θ₁`) using input data (`x` and `y`).
   - The parameters are determined using the following equations:
     - **Mean Calculation**: Compute the mean of the input and output data:
       - `x_mean = (1/n) * Σ xᵢ`
       - `y_mean = (1/n) * Σ yᵢ`
     - **Slope (θ₁) Calculation**: Find the slope by dividing the covariance of `x` and `y` by the variance of `x`:
       - `θ₁ = Σ((xᵢ - x_mean) * (yᵢ - y_mean)) / Σ((xᵢ - x_mean)²)`
     - **Intercept (θ₀) Calculation**: Determine the intercept using the mean values:
       - `θ₀ = y_mean - θ₁ * x_mean`

2. **Prediction**:
   - The `predict_y` method uses the calculated parameters to predict the output `y` for a given input `x` using the linear equation:
     - `y = θ₀ + θ₁ * x`

## Equations Used

- **Linear Equation**: 
  - `y = θ₀ + θ₁ * x`
  
- **Slope Calculation**:
  - `θ₁ = Σ((xᵢ - x_mean) * (yᵢ - y_mean)) / Σ((xᵢ - x_mean)²)`

- **Intercept Calculation**:
  - `θ₀ = y_mean - θ₁ * x_mean`

## Summary

This implementation provides a straightforward way to fit a line to a set of data points and make predictions based on the fitted line. It uses basic statistical methods to compute the best-fit parameters and can be applied to various regression tasks in machine learning, computer vision, and robotics.

- In computer vision, Linear Regression can be used for tasks like depth estimation and predicting continuous values from image features. In robotics, it aids in motion modeling and simple predictive control.
