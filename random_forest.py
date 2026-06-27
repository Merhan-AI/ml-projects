import numpy as np 
from sklearn.ensemble import RandomForestClassifier
x = np.array([[2, 300],
              [4, 500],
              [6, 600],
              [8, 700],
              [10, 800],
              [3, 400],
              [7, 750],
              [5, 550]])
Loan_approved = np.array([0, 0, 1, 1, 1, 0, 1, 0])
model = RandomForestClassifier()
model.fit(x,Loan_approved)
print(model.predict([[4.5, 700]]))
from sklearn.metrics import accuracy_score
print(accuracy_score(Loan_approved, model.predict(x)))
