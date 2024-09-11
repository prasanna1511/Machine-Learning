
## KMeans Clustering

**Type:** Unsupervised Learning (Clustering)

KMeans is a popular clustering algorithm that divides a dataset into \( k \) clusters, where each cluster is defined by its centroid, the mean of all data points assigned to that cluster.

### Initialization Methods

1. **Random Initialization**

   Randomly selects k  data points from the dataset as the initial cluster centroids. 

2. **KMeans++ Initialization**

   KMeans++ improves clustering results by choosing initial centroids that are farther apart. The first centroid is chosen randomly, and subsequent centroids are selected based on their distance from existing centroids. The probability \( p_i \) of selecting a data point \( x_i \) as a centroid is given by:

   $$
   p_i = \frac{\min_{j} \| x_i - \text{center}_j \|^2}{\sum_{k} \min_{j} \| x_k - \text{center}_j \|^2}
   $$

 

### Distance Calculation

The Euclidean distance between data point \( x_i \) and  cluster centroid 

$$
\text{dist}_{ij} = \| x_i - \text{center}_j \|^2
$$



### Objective Function

The KMeans algorithm aims to minimize variance, also known as inertia. The objective function \( J \) to minimize is:

$$
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \text{center}_j \|^2
$$


### Iterative Process

The KMeans algorithm involves the following iterative steps:

1. **Assign Points to Clusters**

   Each data point \( x_i \) is assigned to the nearest centroid:

   $$
   c_i = \arg \min_{j} \| x_i - \text{center}_j \|^2
   $$

 

2. **Update Centroids**

   The centroid of each cluster is updated as the mean of all points assigned to that cluster:

   $$
   \text{center}_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
   $$

  

3. **Convergence**

   The algorithm repeats the above steps until convergence, which occurs when the centroids no longer change significantly or the maximum number of iterations is reached
