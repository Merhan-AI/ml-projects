import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
print(df.head())
print(df.shape)
# Drop useless columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing Age with median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert Sex to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numbers
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(df.isnull().sum())
print(df.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x = df.drop('Survived',axis = 1)
y = df['Survived']

model = RandomForestClassifier()
model.fit(x,y)
print("Accuracy:", accuracy_score(y, model.predict(x)))
# New passenger: 
# Pclass=3, Sex=male(0), Age=22, SibSp=1, Parch=0, Fare=7.25, Embarked=S(0)
new_passenger = np.array([[3, 0, 22, 1, 0, 7.25, 0]])
prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Passenger Survived ✅")
else:
    print("Passenger Did Not Survive ❌")

# Pclass=1, Sex=female(1), Age=30, SibSp=1, Parch=0, Fare=71.28, Embarked=C(1)
new_passenger2 = np.array([[1, 1, 30, 1, 0, 71.28, 1]])
prediction2 = model.predict(new_passenger2)

if prediction2[0] == 1:
    print("Passenger Survived ✅")
else:
    print("Passenger Did Not Survive ❌")