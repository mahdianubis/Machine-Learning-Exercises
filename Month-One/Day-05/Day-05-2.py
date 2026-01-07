import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Day-05-2.csv")

# df["size"] = df["size"].map({"large" : 2, "medium" : 1, "small" : 0})

# x = df.drop(["size", "price_total_billion", "price_per_meter_num"], axis=1)
# y = df["size"]

# x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 ,random_state=42)

# model = DecisionTreeClassifier()

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# accuracy = accuracy_score(y_pred, y_test)

# new_home = [[60, 1402, 1, 3, 1, 1, 1, 0, 0]]
# new_home_pred = model.predict(new_home)
# # print(new_home_pred)

# importances = model.feature_importances_
# feature_names = x.columns

# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# # print(importance_df.sort_values(by='Importance', ascending=False))

# -------------------------------------------------------------------------------------------------------------------------------

# x = df.drop(["size", "price_per_meter_num"], axis=1)
# y = df["price_per_meter_num"]

# x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 ,random_state=42)

# model = DecisionTreeRegressor()

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# error = mean_absolute_error(y_test, y_pred)

# # print(f"{error:,.0f}")

# importances = model.feature_importances_
# feature_names = x.columns

# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# # print(importance_df.sort_values(by='Importance', ascending=False))

# -------------------------------------------------------------------------------------------------------------------------------

x_class = df.drop(["size", "price_total_billion", "price_per_meter_num"], axis=1)
y_class = df["size"]

x_train, x_test, y_train, y_test = train_test_split(x_class, y_class, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

print(f"{acc_knn:.2f}")

