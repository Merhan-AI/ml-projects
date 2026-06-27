import numpy as np
from sklearn.neighbors import KNeighborsClassifier
x = np.array([[15, 6],
              [17, 5],
              [16, 7],
              [45, 2],
              [50, 1],
              [48, 2],
              [19, 6],
              [42, 3]])
likes = np.array([0, 0, 0, 1, 1, 1, 0, 1])
model = KNeighborsClassifier()
model.fit(x,likes)
print(model.predict([[29, 3.75]]))
from sklearn.metrics import accuracy_score
print(accuracy_score(likes,model.predict(x)))