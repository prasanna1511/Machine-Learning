import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
from mlrcv.core import *

class TreeNode:
    def __init__(self, node_id: str, max_degree: int):
        """
        This function initializes the TreeNode class (already implemented):

        Args:
            - node_id (str): node id to identiy the current node
            - max_degree (int): max degree of the tree

        Returns:
        """
        self.max_attr_gain = -np.inf
        self.attr_split = None
        self.attr_split_val = None
        self.node_class = None
        self.children = {}
        self.leaf = False
        self.node_id = node_id
        self.max_degree = max_degree

    def infer_node(self, x: np.ndarray) -> float:
        """
        This function goes over the tree given the input data x and return the 
        respective leaf class (only returns node_class after reaching a leaf):

        Args:
            - x (np.ndarray): input data x to be checked over the tree

        Returns:
            - node_class (float): respective leaf class given input x
        """

        if self.leaf:
            return self.node_class
        
       
        
        if x[self.attr_split] <= self.attr_split_val:
            return self.children['left'].infer_node(x)
        else:
            return self.children['right'].infer_node(x)
        # return node_class

    def split_node(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        This function uses the current x and y data to split the tree nodes
          (left and right) given the information_gain

        calculated over the possible splits. Recursion stop condition
          will be when the current degree arrives at
        maximum degree (setting it as leaf):

        Args:
            - x (np.ndarray): input data x to be splited
            - y (np.ndarray): class labels of the input data x
            - degree (int): current node degree

        Returns:
        """
        if degree == self.max_degree or len(y) == 0:
            self.leaf = True
            self.node_class = np.argmax(np.bincount(y.astype(int))) if len(y) > 0 else 0  # Handle empty y gracefully
            return

        if np.unique(y).size == 1:
            self.leaf = True
            self.node_class = y[0] 
            return

        best_attr, best_split, best_gain = -1, None, -np.inf

        for attr in range(x.shape[1]):
            split, gain = self.attr_gain(x[:, attr], y)
            if gain > best_gain:
                best_gain = gain
                best_attr, best_split = attr, split

        if best_attr == -1 or best_split is None:
            self.leaf = True
            self.node_class = np.argmax(np.bincount(y.astype(int)))
            return

        self.attr_split = best_attr
        self.attr_split_val = best_split

        x_l = x[x[:, best_attr] <= best_split]
        y_l = y[x[:, best_attr] <= best_split]
        x_r = x[x[:, best_attr] > best_split]
        y_r = y[x[:, best_attr] > best_split]

        if len(y_l) > 0 and len(y_r) > 0:
            self.children['left'] = TreeNode(self.node_id + 'l', self.max_degree)
            self.children['right'] = TreeNode(self.node_id + 'r', self.max_degree)
            self.children['left'].split_node(x_l, y_l, degree + 1)
            self.children['right'].split_node(x_r, y_r, degree + 1)
        else:
            self.leaf = True
            self.node_class = np.argmax(np.bincount(y.astype(int)))


    def attr_gain(self, x_attr: np.ndarray, y: np.ndarray) -> (float, float):
        """
        This function calculates the attribute gain:

        Args:
            - x_attr (np.ndarray): input data x[attr] to be splitted
            - y (np.ndarray): labels of the input data x

        Returns:
            - split_gain (float): highest gain from the possible attributes splits
            - split_value (float): split value selected for x_attr attribute
        """
        split_gain = None
        split_value = None
        
        unique_values = np.unique(x_attr)
        best_gain = -np.inf
        best_value = None
        for value in unique_values:
            y_l = y[x_attr <= value]
            y_r = y[x_attr > value]
            gain = self.information_gain(y, y_l, y_r)
            if gain > best_gain:
                best_gain = gain
                best_value = value
        split_gain = best_gain
        split_value = best_value

        return split_gain, split_value
    
    def information_gain(self, y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> float:
        """
        This function calculates the attribute gain from the candidate splits y_l and y_r:

        Args:
            - y (np.ndarray): the full labels of the current node
            - y_l (np.ndarray): labels of the candidate left node split
            - y_r (np.ndarray): labels of the candidate right node split

        Returns:
            - I (float): information gain from the candidate splits y_l and y_r
        """
        
        pl = len(y_l) / len(y)
        pr = len(y_r) / len(y)
        I = self.entropy(y) - pl * self.entropy(y_l) - pr * self.entropy(y_r)
        
        # I = None

        return I

    def entropy(self, y: np.ndarray) -> float:
        """
        This function calculates the entropy from the input labels set y:

        Args:
            - y (np.ndarray): the labels set to calculate the entropy

        Returns:
            - H (float): the entropy of the input labels set y
        """
        H = 0
        for c in np.unique(y):
            p = len(y[y == c]) / len(y)
            H -= p * np.log2(p)
        # H = None
        return H

class DecisionTree:
    def __init__(self, num_class: int, max_degree: int):
        """
        This function initializes the DecisionTree class (already implemented):

        Args:
            - num_class (int): number of class from your data
            - max_degree (int): max degree of the tree

        Returns:
        """
        self.root = None
        self.max_degree = 5
        self.num_class = num_class
        
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function fits the decision tree with the training data x and labels y.
        Starting from root tree node,
        and iterate recursively over the nodes, split on left and right nodes:

        Args:
            - x (np.ndarray): the full labels of the current node
            - y (np.ndarray): labels of the candidate left node split

        Returns:

        """
        self.root = TreeNode('root', self.max_degree)
        self.root.split_node(x, y, 0)      
        
        # pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts y_pred class from the input x:

        Args:
            - x (np.ndarray): input data to be predicted by the tree

        Returns:
            - y_pred (np.ndarray): tree predictions over input x
        """
        # y_pred = None
        y_pred = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            y_pred[i] = self.root.infer_node(x[i])

        
        return y_pred

    def eval(self, x: np.ndarray, y: np.ndarray):
        """
        This function evaluate the model predicting y_pred from input x and calculating teh accuracy between y_pred and y:

        Args:
            - x (np.ndarray): input data to be predicted by the tree
            - y (np.ndarray): input class labels

        Returns:
            - acc (float): accuracy of the model
        """

        return accuracy(y, self.predict_y(x))
