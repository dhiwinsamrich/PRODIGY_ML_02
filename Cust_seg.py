# Importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Data Collection and Analysis
#Laod the csv file to DataFrame

customer_data = pd.read_csv('D:\\Internship\\K_means Clustering\\New DataSet\\Mall_Customers.csv')

#print(customer_data.head())
##Finding the no.of rows and columns
#print(customer_data.shape)
##Getting information about the DataSet
#print(customer_data.info())
##Getting information about the missing values
#print(customer_data.isnull().sum())

#Choosing annual income and spending score columns
X = customer_data.iloc[:,[3,4]].values

#choosing the correct no.of clusters
#WCSS -> Within Clusers Sum of Squares
#Elbow Method : Plotting WCSS for different number of clusters & choosing the point where there is an increase in WCSS value 

wcss=[]

# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
#     kmeans.fit(X)

#     wcss.append(kmeans.inertia_)
    
    #plot an Elbow graph to find min WCSS value
    #sns.set()
    # plt.plot(range(1,11), wcss)
    # plt.title("Elbow Point Graph")
    # plt.xlabel("No.of Clusters")
    # plt.ylabel('WCSS Value')
    # plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

#print(Y)

# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()