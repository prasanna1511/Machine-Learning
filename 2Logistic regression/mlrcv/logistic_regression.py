import numpy as np
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.utils import *
import typing


class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        return sigmoid(np.dot(x, self.theta))

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der (np.ndarray): first derivative value
        """
        error = y_pred - y
        gradient = np.dot(x.T, error) / len(y)
        return gradient

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta parameters that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """

        x = np.hstack((np.ones((x.shape[0], 1)), x)) 
        n_features = x.shape[1]
        self.theta = np.zeros((n_features, 1))  

        for i in range(self.epochs):
            y_pred = self.predict_y(x)
            gradient = self.first_derivative(x, y_pred, y)
            self.theta -= self.learning_rate * gradient  

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """
        
        x = np.hstack((np.ones((x.shape[0], 1)), x))  
        y_pred = self.predict_y(x)
        y_pred_class = (y_pred >= 0.5).astype(int) 
        accuracy = np.mean(y_pred_class == y) * 100 
        return accuracy

class MultiClassLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.theta_class = None 

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
This function should use the parameters theta to predict the y class given an input x

Args:
    - x (np.ndarray): input data to predict y classes

Returns:
    - y_pred (np.ndarray): the model prediction of the input x
"""
       
        if x.shape[1] + 1 == self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x)) 
        y_pred = softmax(np.dot(x, self.theta))
        print("Xshape", x.shape, " | theta shape", self.theta.shape, " | y_pred shape", y_pred.shape)
        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:

        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y,
        for each possible class.

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der: first derivative value
        """
        error = y_pred - y
        gradient = np.dot(x.T, error) / len(x)
        print("gradient shape", gradient.shape)
        return gradient

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta_class parameters (multiclass) that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """

        n_samples, n_features = x.shape
        num_classes = y.shape[1]

        # Add a column of ones to x for the bias term
        x = np.hstack((np.ones((n_samples, 1)), x))

        if self.theta is None:
            self.theta = np.zeros((n_features + 1, num_classes)) 

        self.theta_class = self.theta 

        print("Xshape", x.shape, " | theta shape", self.theta.shape, " | y shape", y.shape)
        for i in range(self.epochs):
            y_pred = self.predict_y(x)
            gradient = self.first_derivative(x, y_pred, y)
            self.theta -= self.learning_rate * gradient

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:

        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """

       
        if x.shape[1] + 1 == self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))  

        y_pred_probs = self.predict_y(x)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true) * 100
        print("Accuracy:", accuracy)
        return accuracy
