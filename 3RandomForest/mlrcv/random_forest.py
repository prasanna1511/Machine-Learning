import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.decision_tree import *
from typing import Optional

class RandomTreeNode:
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
        This function goes over the tree given the input data x and return the respective leaf class:

        Args:
            - x (np.ndarray): input data x to be checked over the tree

        Returns:
            - node_class (float): respective leaf class given input x
        """
        # node_class = None

        if self.leaf:
            return self.node_class
        
        if self.attr_split is None or self.attr_split_val is None:
            raise ValueError(f"TreeNode with ID {self.node_id} has no valid split defined.")
        
        if x[self.attr_split] <= self.attr_split_val:
            if 'left' in self.children:
                return self.children['left'].infer_node(x)
            else:
                return self.node_class  
        else:
            if 'right' in self.children:
                return self.children['right'].infer_node(x)
            else:
                return self.node_class  


        # return node_class

    def split_node(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        This function uses the current x and y data to split the tree nodes (left and right) given the information_gain
        calculated over the possible splits. Recursion stop condition will be when the current degree arrives at
        maximum degree (setting it as leaf):

        Args:
            - x (np.ndarray): input data x to be splited
            - y (np.ndarray): class labels of the input data x
            - degree (int): current node degree

        Returns:
        """
        if degree == self.max_degree or len(y) == 0 or len(np.unique(y)) == 1:
            self.leaf = True
            y_int = y.astype(int)
            self.node_class = np.argmax(np.bincount(y_int)) if len(y) > 0 else 0
            return

        best_attr, best_split, best_gain = -1, None, -np.inf
        for attr in range(x.shape[1]):
            split, gain = self.attr_gain(x[:, attr], y)
            if gain > best_gain:
                best_gain = gain
                best_attr, best_split = attr, split

        if best_attr == -1 or best_split is None:
            self.leaf = True
            y_int = y.astype(int)
            self.node_class = np.argmax(np.bincount(y_int))
            return

        self.attr_split = best_attr
        self.attr_split_val = best_split

        x_l = x[x[:, best_attr] <= best_split]
        y_l = y[x[:, best_attr] <= best_split]
        x_r = x[x[:, best_attr] > best_split]
        y_r = y[x[:, best_attr] > best_split]

        if len(y_l) > 0 and len(y_r) > 0:
            self.children['left'] = RandomTreeNode(self.node_id + 'L', self.max_degree)
            self.children['left'].split_node(x_l, y_l, degree + 1)
            self.children['right'] = RandomTreeNode(self.node_id + 'R', self.max_degree)
            self.children['right'].split_node(x_r, y_r, degree + 1)
        else:
            self.leaf = True
            y_int = y.astype(int)
            self.node_class = np.argmax(np.bincount(y_int))
        # return

    def attr_gain(self, x_attr: np.ndarray, y: np.ndarray) -> (float, float):
        """
        This function calculates the attribute gain. For the random tree case, the attr splits should be divided
        as in the decision tree, but then a subset should be random selected from it:

        Args:
            - x_attr (np.ndarray): input data x[attr] to be splitted
            - y (np.ndarray): labels of the input data x

        Returns:
            - split_gain (float): highest gain from the possible attributes splits
            - split_value (float): split value selected for x_attr attribute
        """
        split_gain = None
        split_value = None

        possible_splits = np.unique(x_attr)
        if len(possible_splits) < 2:
            return -np.inf, None

        random_splits = np.random.choice(possible_splits, size=min(10, len(possible_splits)), replace=False)
        best_split, best_gain = None, -np.inf
        for split in random_splits:
            y_l = y[x_attr <= split]
            y_r = y[x_attr > split]
            gain = self.information_gain(y, y_l, y_r)
            if gain > best_gain:
                best_gain = gain
                best_split = split

        return best_gain, best_split


        # return split_gain, split_value

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
        # I = None
        I = self.entropy(y) - (len(y_l) / len(y)) * self.entropy(y_l) - (len(y_r) / len(y)) * self.entropy(y_r)
        return I

    def entropy(self, y: np.ndarray) -> float:
        """
        This function calculates the entropy from the input labels set y:

        Args:
            - y (np.ndarray): the labels set to calculate the entropy

        Returns:
            - H (float): the entropy of the input labels set y
        """
        # H = None
        if len(y) == 0:
            return 0
        counts = np.bincount(y.astype(int))
        probabilities = counts / len(y)
        H = -np.sum(probabilities * np.log2(probabilities + 1e-9))
 
        return H

class RandomTree:
    def __init__(self, num_class: int, max_degree: int):
        """
        This function initializes the RandomTree class (already implemented):

        Args:
            - num_class (int): number of class from your data
            - max_degree (int): max degree of the tree

        Returns:
        """
        self.root = None
        self.max_degree = max_degree
        self.num_class = num_class
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function fits the random tree with the training data x and labels y. Starting from root tree node,
        and iterate recursively over the nodes, split on left and right nodes:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): labels of the input data x

        Returns:

        """
        # pass
        self.root = RandomTreeNode('0', self.max_degree)
        self.root.split_node(x, y, 0)

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts y_pred class from the input x:

        Args:
            - x (np.ndarray): input data to be predicted by the tree

        Returns:
            - y_pred (np.ndarray): tree predictions over input x
        """
        # y_pred = None
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
             y_pred[i] = self.root.infer_node(x[i])
        return y_pred

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function evaluate the model predicting y_pred from input x and calculating teh accuracy between y_pred and y:

        Args:
            - x (np.ndarray): input data to be predicted by the tree
            - y (np.ndarray): input class labels

        Returns:
            - acc (float): accuracy of the model
        """
        return accuracy(y, self.predict_y(x))

class RandomForest:
    def __init__(self, num_class: int, max_degree: int, trees_num: Optional[int] = 10, random_rho: Optional[float] = 1.0):
        """
        This function initializes the RandomForest class (already implemented):

        Args:
            - num_class (int): number of class from your data
            - max_degree (int): max degree of the tree
            - trees_num (int): number of random trees to be generated
            - random_rho (float): rho attribute to generate the random subset from the input data

        Returns:
        """
        self.max_degree = max_degree
        self.trees_num = trees_num
        self.random_rho = random_rho
        self.d_trees = [ RandomTree(num_class, self.max_degree) for _ in range(trees_num) ]
        self.num_class = num_class

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        This function fits the random forest with the training data x and labels y. For each random tree fits the 
        data x and y:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): labels of the input data x

        Returns:

        """
        # pass
        for i in range(self.trees_num):
            x_s, y_s = self.random_tree_data(x, y)
            self.d_trees[i].fit(x_s, y_s)


    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts y_pred from input data x. For each random tree get the predicted class,
        then, sum the predicts over a voting matrix, returning a data distribution:

        Args:
            - x (np.ndarray): the input data x to be predicted
        Returns:
            - y_pred (np.ndarray): the prediction y_pred from the input x
        """
        
        y_pred = np.zeros((x.shape[0], self.num_class))
        for tree in self.d_trees:
            y_pred += np.eye(self.num_class)[tree.predict_y(x).astype(int)]
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def random_tree_data(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        This function generates a random subset x_s of samples from the input data x and y where
        len(x_s) / len(x) = self.rho:

        Args:
            - x (np.ndarray): the input data x
            - y (np.ndarray): the labels of the input data x
        Returns:
            - x_s (np.ndarray): the subset of the input data x
            - y_s (np.ndarray): the correspondent subset of labels y
        """
        num_samples = int(len(x) * self.random_rho)
        indices = np.random.choice(len(x), num_samples, replace=False)
        return x[indices], y[indices]
        # return x_s, y_s

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function evaluate the model predicting y_pred from input x and calculating teh accuracy between y_pred and y:

        Args:
            - x (np.ndarray): input data to be predicted by the tree
            - y (np.ndarray): input class labels

        Returns:
            - acc (float): accuracy of the model
        """
        return accuracy(y, self.predict_y(x))
