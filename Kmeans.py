from sklearn.cluster import KMeans
import numpy as np
X= np.array([
    [25, 80],
    [30, 75],   # Person 2: Age=30, Spending Score=75
    [22, 90],   # Person 3: Age=22, Spending Score=90
    [45, 20],   # Person 4: Age=45, Spending Score=20
    [50, 15],   # Person 5: Age=50, Spending Score=15
    [48, 25],   # Person 6: Age=48, Spending Score=25
    [35, 50],   # Person 7: Age=35, Spending Score=50
    [32, 55],   # Person 8: Age=32, Spending Score=55
    [38, 45]    # Person 9: Age=38, Spending Score=45
])
model = KMeans(n_clusters=3)
model.fit(X)
print(model.labels_)
print(model.cluster_centers_)
