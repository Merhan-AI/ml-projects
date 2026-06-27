import numpy as np
from sklearn.svm import SVC
x = np.array([[2,300],
              [4,500],
              [6,600],
              [8,700],
              [10,800],
              [3,400],
              [7,750],
              [5,550]])
y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
model = SVC()
model.fit(x,y)
print(model.predict([[6, 650]]))
from sklearn.metrics import accuracy_score
print(accuracy_score(y,model.predict(x)))