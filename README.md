# kmeans-clustering-python


## K-Means Clustering Algorithm in Python

This project implements the K-Means clustering algorithm from scratch in Python. It offers functionality to visualize each step of the clustering process and even allows viewing a specific iteration upon request. The algorithm is robust and uses a fixed random seed to ensure consistent outputs, making it suitable for learning, experimentation, and showcasing the K-Means algorithm.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Project Challenges and Solutions](#project-challenges-and-solutions)
- [Example Usage](#example-usage)
- [Installation and Usage](#installation-and-usage)
- [Motivation](#motivation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

# Features

## Custom Implementation:
K-Means algorithm implemented without using high-level libraries like scikit-learn.

## Interactive Visualization:
Visualizes each iteration of clustering, from initialization to final convergence.

## Specific Iteration Display:
Allows users to focus on a particular iteration and its clustering state.

## Fixed Random Seed:
Ensures consistent outputs for reproducibility.

## Convergence Check:
Automatically detects when the centroids stabilize.


# Dataset

The dataset is generated using the make_blobs function from scikit-learn, which creates synthetic data with a known number of clusters. This ensures simplicity and allows clear visualization of the algorithm's behavior.


# Project Challenges and Solutions:

## 1. Inconsistent Outputs
   
 Problem: The algorithm initially produced different outputs each time it ran due to the randomness in centroid initialization.

 Solution: A fixed random seed (random_seed=42) was added to ensure consistent results across runs.



## 2. Specific Iteration Visualization
   
Problem: Displaying specific iterations was initially unavailable.

Solution: Added a plot_specific_iteration function to display clusters and centroids at a user-specified iteration. This feature provides flexibility in analyzing the clustering process.


# Example Usage

## Initial Dataset Visualization

The following image shows the initial dataset with randomly generated data points distributed into four distinct clusters:

![image](https://github.com/user-attachments/assets/adfd49bd-4608-434d-a0ef-ebc079dca6ca)



## Iteration Progress

The algorithm visualizes each iteration, displaying the movement of centroids and the grouping of points into clusters:

![image](https://github.com/user-attachments/assets/5f8bc1ed-fc97-4396-9338-a1aae16dcec6)

(this image only includes the view after the first (0) iteration.)


**Note**: The iterations in this implementation start at index 0 because the clustering process is iteratively implemented using Python's `range()` function. This means:
- Iteration 0 corresponds to the initial clustering with randomly chosen centroids.
- Subsequent iterations (1, 2, etc.) correspond to updates to centroids and clusters.
  

## Final Clustering

After convergence, the final clusters are displayed:

![image](https://github.com/user-attachments/assets/7c414fda-a5a6-4863-8b18-78bc37b65069)

## Specific Iteration

You can also view a specific iteration, such as iteration 3:

![image](https://github.com/user-attachments/assets/289ff76e-3926-486c-9387-603da93a9a38)


To analyze a different iteration, simply modify the `specific_iteration` variable in the script. For example, setting `specific_iteration = 2` will display the clustering process at the second iteration.


# Installation and Usage

## Clone this repository:

git clone https://github.com/ipekucr/kmeans-clustering-python.git
cd kmeans-clustering-python

## Install dependencies:
Ensure you have Python 3.x and the required libraries (numpy, matplotlib, scikit-learn) installed.

## Run the script:
python kmeans_clustering.py

# Motivation
K-Means is one of the most widely used clustering algorithms in data science and machine learning. This project aims to:
- Demystify how K-Means works by implementing it from scratch.
- Provide a tool for visualizing the clustering process step-by-step.
- Help users understand clustering principles through interactive and customizable features.

# Acknowledgments

This project was inspired by the article ["Create Your Own K-Means Clustering Algorithm in Python"] (https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670) on Towards Data Science. We expanded upon the foundation laid by this article to include advanced features such as reproducibility, empty cluster handling, and specific iteration visualization.


# License

This project is licensed under the MIT License.
















































