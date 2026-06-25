import numpy as np
from sklearn.linear_model import LinearRegression
Experience =np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1,1)
Salary = np.array([30000, 35000, 40000, 50000, 60000, 70000, 80000, 90000])
model = LinearRegression()
model.fit(Experience,Salary)
prediction = model.predict([[9]])
print(prediction)
print(model.coef_)
print(model.intercept_)
from sklearn.metrics import r2_score
print(r2_score(Salary, model.predict(Experience)))
