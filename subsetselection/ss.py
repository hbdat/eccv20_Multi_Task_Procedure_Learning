import numpy as np
import torch
import pandas as pd
import scipy
from sklearn.cluster import KMeans

#%%
# Funtion that takes the similarity matrix between Kmeans centers and segment features
#         and resturns the set of representative centers and assignments to representatives
# S: similarity matrix between X and Y
# repNum: number of representatives from X
def run_ss(S,repNum):
	N = S.shape[0]
	active_set = np.empty(0)
	remaining_set = np.array(list(set(range(N)) - set(active_set)))
	cost1 = -float('inf')
	best_cost = -float('inf')
	assignment = np.array([0, N])
	for iter in range(repNum):
		for i in range(len(remaining_set)):
			element = remaining_set[i]
			[cost2, assignment2] = ss_cost(S, np.append(active_set,element).astype(int))
			if (cost2 > best_cost):
				best_cost = cost2
				best_index = element
				best_assignment = assignment2
		if (best_cost > cost1):
			active_set = np.append(active_set, best_index)
			remaining_set = np.array(list( set(range(N)) - set(active_set) ))
			cost1 = best_cost
			assignment = best_assignment
		else:
			break
	return active_set.astype(int), assignment.astype(int)


# Function to compute the best assignment for a given active set
# S: similarity matrix between X and Y
# aset: subset of indices from X
def ss_cost(S, aset):
	N = S.shape[0]
	#[v, assgn] = torch.max(S[aset,:],0)
	v = np.ndarray.max(S[aset,:], 0)
	assgn = np.ndarray.argmax(S[aset,:], 0)
	#cost = sum(v).detach().numpy()
	cost = sum(v)
	return cost, assgn

#%%
################   Main Part of Code #############

# Take list of K arrays, each item corresponding to T_i x d array of d-dimensional features of T_i segments
K = 5
d = 2
T = np.array([50, 45, 50, 40, 60])
F = np.empty(K, dtype=object)
Y = np.array([0, d]) # sum(T) x d dimensional array of all features of all videos
for i in range(K):
	F[i] = np.random.randn(T[i],d)
	if (i == 0):
		Y = F[0]
	else:
		Y = np.append(Y, F[i], axis = 0)

# Apply kmeans to data to get centroids
M = 30 # number of clusters
kmeans = KMeans(n_clusters=M, init='k-means++', max_iter=1000, n_init=50, random_state=0)
kmeans.fit(Y)
X = kmeans.cluster_centers_ # X is the M x d array of M centers in d-dimension

# Compute similarity between X and Y
S = -scipy.spatial.distance.cdist(X, Y, metric='euclidean')

# Run subset selection
# repNum: number of representative centers
# reps: representative centers
# assignments: assignments of segments to representative centers
repNum = 10
[reps, assignments] = run_ss(S,repNum)

print(reps)
print(assignments)

print(np.unique(reps).shape)
print(np.unique(assignments).shape)

