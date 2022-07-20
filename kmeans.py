

import random
import math
import matplotlib.pyplot as plt
import numpy as np
# Euclidian Distance between two d-dimensional points
def eucldist(p0, p1):
    dist = 0.0
    for i in range(0, len(p0)):
        dist += (p0[i] - p1[i]) ** 2
    return math.sqrt(dist)

cluster_centers = []
# K-Means Algorithm
def kmeans(k, datapoints):
    # d - Dimensionality of Datapoints
    d = len(datapoints[0])
    print("d:")
    print(d)
    print("len",len(datapoints))
    # Limit our iterations

    Max_Iterations = 10000
    i = 0

    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    cluster_centers = []

    print(cluster)
    print(prev_cluster)
    # Randomly Choose Centers for the Clusters
    for i in range(0, k):
        new_cluster = []
        # for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
        cluster_centers += [random.choice(datapoints)]
        force_recalculation = False


    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation):

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        # Update Point's Cluster Alligiance
        for p in range(0, len(datapoints)):
            min_dist = float("inf")#represent an infinite number

            # Check min_distance against all centers
            for c in range(0, len(cluster_centers)):

                dist = eucldist(datapoints[p], cluster_centers[c])
                # print(dist,min_dist)
                if (dist < min_dist):
                    min_dist = dist
                    cluster[p] = c  # Reassign Point to new Cluster

        # print(cluster_centers)
        # print(len(cluster_centers))
        # print("__DEBUG__",cluster)
        # Update Cluster's Position
        for k in range(0, len(cluster_centers)):
            new_center = [0] * d
            members = 0

            for p in range(0, len(datapoints)):
                # print(cluster[p])
                if (cluster[p] == k):  # If this point belongs to the cluster
                    for j in range(0, d):

                        new_center[j] += datapoints[p][j]
                        # print(new_center[j])
                    members += 1

            for j in range(0, d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)
                    # print(new_center[j])
                    print(members)
                    # This means that our initial random assignment was poorly chosen
                # Change it to a new datapoint to actually force k clusters
                else:
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print("重新計算...")
            print(new_center)
            cluster_centers[k] = new_center

    print("======== Results ========")
    print("Clusters", cluster_centers)
    print("Iterations", i)
    print("Assignments", cluster)
    return cluster_centers

# TESTING THE PROGRAM#
if __name__ == "__main__":
    # 2D - Datapoints List of n d-dimensional vectors. (For this example I already set up 2D Tuples)
    # Feel free to change to whatever size tuples you want...

    k = 2  # K - Number of Clusters

    X = -2 * np.random.rand(100, 2)
    X1 = 1 + 2 * np.random.rand(50, 2)
    X[50:100, :] = X1
    # print(X)
    plt.scatter(X[:, 0], X[:, 1], s=50, c="b")
    plt.scatter(X[:, 0], X[:, 1], s=50, c="b")
    print("Data Point: ", X)

    clu_cen = kmeans(k, X)
    print("K Clusters: ", clu_cen)
    plt.scatter(clu_cen[0][0],clu_cen[0][1], s=200, c="g", marker ="s")
    plt.scatter(clu_cen[1][0], clu_cen[1][1], s=200, c="r", marker ="s")
    plt.show()