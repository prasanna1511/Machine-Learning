#### 3. **Random Forests**

- **Type**: Supervised Learning (Classification and Regression)
  
- **Description**: Random Forests are an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their classifications (for classification tasks) or the mean prediction (for regression tasks)

## Decision Tree

**Type:** Supervised Learning (Classification/Regression)

A Decision Tree is a flowchart
- Each internal node represents a test on an attribute.
- Each branch represents an outcome of the test.
- Each leaf node represents a class label (in classification) or a continuous value (in regression).


**Entropy:**

Entropy measures the impurity of a dataset

$$
H(D) = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$
Where:

c is the number of classes.
p_i is the proportion of examples in class i

**Information Gain:**

$$
Gain(D,A) = H(D) - \left( \sum_{|D_v|}{|D|} H(S_1) + \frac{|S_2|}{|S|} H(S_2) \right)
$$


Information Gain helps in choosing the best attribute to split a node. It is given by:


  
**Split Node Criteria**

To split a node, calculate the gain for each attribute and select the one with the highest gain:
$$
\text{BestGain} = max(Gain(D, attr))
$$

**Stopping Criteria**

Stop splitting a node if:

The node's depth reaches the maximum degree of the tree.
All examples at the node are of the same class.
There are no more attributes to split on.
The subset of examples is empty.

**Prediction**

To predict the class of a given input x, traverse the tree from the root to a leaf node:
$$
Prediction(x)= Class(LeafNodeReachedbyx)$$


**Accuracy**

Accuracy if correctly classified 
$$
\text{accuracy} = \left(\frac{\text{number of correct predictions}}{\text{total predictions}}\right) \times 100
$$

