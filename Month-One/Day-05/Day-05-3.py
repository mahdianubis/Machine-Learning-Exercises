import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("Day-05-3.csv")

x = df.drop(["price", "country", "city", "street", "date", "statezip"], axis=1)
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn_model = KNeighborsRegressor(n_neighbors=6)
knn_model.fit(x_train_scaled, y_train)
knn_pred = knn_model.predict(x_test_scaled)
knn_error = mean_absolute_error(y_test, knn_pred)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(x_train, y_train)
tree_pred = tree_model.predict(x_test)
tree_error = mean_absolute_error(y_test, tree_pred)

print(f"KNN MAE: {knn_error:,.0f}")
print(f"Decision Tree MAE: {tree_error:,.0f}")