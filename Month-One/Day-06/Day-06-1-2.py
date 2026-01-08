import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Day-06-1.csv")
df["Species"] = df["Species"].map({"Iris-setosa" : 0, "Iris-versicolor" : 1, "Iris-virginica" : 2})

x = df.drop(["SepalLengthCm", "Id"], axis=1)
y = df["SepalLengthCm"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = mean_absolute_error(y_pred, y_test)
print(score)