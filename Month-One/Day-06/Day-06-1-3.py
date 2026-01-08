import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("Day-06-1.csv")
df["Species"] = df["Species"].map({"Iris-setosa" : 0, "Iris-versicolor" : 1, "Iris-virginica" : 2})

x = df.drop(["SepalLengthCm", "Id"], axis=1)
y = df["SepalLengthCm"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=5)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

score = mean_absolute_error(y_pred, y_test)
print(score)