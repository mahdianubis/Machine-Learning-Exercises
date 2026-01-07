import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Day-05-1.csv")

x = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)

print(f"{accuracy:.2f}%")

new_wine = [[13.7, 1.7, 2.4, 15.6, 110, 2.8, 3.0, 0.3, 2.2, 5.6, 1.0, 3.9, 1000]]

prediction = model.predict(new_wine)
print(prediction[0])