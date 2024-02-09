import numpy as np
from sklearn.cluster import KMeans
import csv

def read_customer_data(file_path):
    customer_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            row = [int(row[0]), row[1], int(row[2]), float(row[3]), int(row[4])]
            customer_data.append(row)
    return np.array(customer_data)

file_path = './Dataset/Mall_Customers.csv'

customer_data = read_customer_data(file_path)

features = customer_data[:, 2:]

num_clusters = 5 

kmeans = KMeans(n_clusters=num_clusters)

kmeans.fit(features)

cluster_labels = kmeans.labels_

for i, label in enumerate(cluster_labels):
    print(f"CustomerID {customer_data[i, 0]} is in cluster {label+1}")
