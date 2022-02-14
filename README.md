# Collaborative Filtering via Gaussian Mixtures

## Problems
We are given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user (matrix rows) has rated only a small fraction of the movies (matrix columns) so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix. We will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have K possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step).

## Part I: Comparing Clustering via K-means and (soft) Clustering by EM Algorithm
The 2D toy dataset used in this part is loaded from "toy_data.txt". The objective is to minimize the cost function, which is the sum squared of errors (SSE) of prediction matrix and the original matrix element-wise, while finding a reasonable number of clusters (a trivial solution would be number of clusters equals to the number of users). The SSE only includes non-zero elements of the original matrix.

### K-means Clustering
In this part, we try to calculate the SSE using K = {1, 2, 3, 4} and (random) seed {0, 1, 2, 3, 4}. Varying the seed is necessary, since the k-means algorithm is sensitive to the initial location of the means of each cluster. The results which generate the lowest SSE (for varying seed) are as follows:
- Cost (K=1): 5462.30 (best seed: 0)
- Cost (K=2): 1684.91 (best seed: 0)
- Cost (K=3): 1329.59 (best seed: 3)
- Cost (K=4): 1035.50 (best seed: 4)

For example, take K=3 and seed=3, if we visualize the data we have:
![Figure_3](https://user-images.githubusercontent.com/87055709/153839059-d0056fe9-0b2b-4491-8450-e8e15de90ea1.png)

### EM Algorithm
The procedur in this part is similar to the previous part. The difference is mainly in soft cluster, which means that one data point can be clustered in more than one clusters, but each with corresponding probability.

For example, take K=4 and seed=0, if we visualize the data we have:
![Figure_4_EM](https://user-images.githubusercontent.com/87055709/153839824-c05cc127-0407-4fe0-a5d0-63c810feae0c.png)

### Bayesian Information Criterion (BIC)
One way to find the best number of cluster is using the BIC, which is defined as follows:
![image](https://user-images.githubusercontent.com/87055709/153840209-65747cff-f6df-48ec-aaf4-72e3a007c05e.png)

where:
l = log-likelihood of the data under the current model
p = number of free parameters
n = number of data points

Using this criterion, we have that the best K is 3 with BIC value -1169.26.

### Mixture models for matrix completion (netflix data)
![image](https://user-images.githubusercontent.com/87055709/153840893-e248e0ea-a5f1-4467-af0c-bf00c7ee9e5f.png)
![image](https://user-images.githubusercontent.com/87055709/153840940-ae8edeb0-7fc6-40df-9000-d1a7058f7c5a.png)
![image](https://user-images.githubusercontent.com/87055709/153841015-b5e45ac4-7ed2-4eee-a312-c041ac8037ef.png)
![image](https://user-images.githubusercontent.com/87055709/153841067-001c701e-96a0-4a82-bbd8-f8bcd209fdd3.png)
![image](https://user-images.githubusercontent.com/87055709/153841234-ce4200fa-405c-4009-b909-25e94dbd805f.png)
![image](https://user-images.githubusercontent.com/87055709/153841314-cabeec4e-6337-414e-bc2b-05b9f972e66d.png)
![image](https://user-images.githubusercontent.com/87055709/153841406-174bfc5a-277c-4899-b458-08c761d368b6.png)
![image](https://user-images.githubusercontent.com/87055709/153841473-6fbda50f-f14b-4180-a4ec-b431ba3cf806.png)
![image](https://user-images.githubusercontent.com/87055709/153841534-5b98fb26-b506-4e9c-b939-2322699d377c.png)

Using K=12, comparing the "netflix_complete.txt" and the prediction from the model we have RMSE value 0.4805.
