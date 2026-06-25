import numpy as np
from sklearn.linear_model import LogisticRegression
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = LogisticRegression()
model.fit(hours,result)
print(model.predict([[14.5]]))
from sklearn.metrics import accuracy_score
print(accuracy_score(result,model.predict(hours)))
print(model.predict([[1]]))   # 1 hour - should be?
print(model.predict([[3]]))   # 3 hours - should be?
print(model.predict([[5]]))   # 5 hours - should be?