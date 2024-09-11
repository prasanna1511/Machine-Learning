import numpy as np
from typing import Optional
from mlrcv.utils import *

class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line

        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        x_mean =np.mean(x)
        y_mean = np.mean(y)

        theta_1 =np.sum((x-x_mean)*(y-y_mean) )/np.sum(x-x_mean)
        theta_0 = y_mean - theta_1*x_mean

        self.theta_0 = theta_0
        self.theta_1 = theta_1

        

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """
        
        y_pred = self.theta_0 + self.theta_1*x

        return y_pred

class NonLinearRegression:
    def __init__(self):
        self.theta = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
        X = np.ones((len(x), degree+1))
        for i in range(1, degree+1):
            X[:, i] = x**i
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """
        
        y_pred = np.zeros(len(x))
        for i in range(len(self.theta)):
            y_pred = y_pred + self.theta[i] * x**i

        return y_pred
