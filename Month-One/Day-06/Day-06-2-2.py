import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("Day-06-2.csv")


df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"female" : 1, "male" : 0})
df["Embarked"] = df["Embarked"].map({"C" : 2, "S" : 1, "Q" : 0})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df = df.drop("Cabin", axis=1)
df = df.dropna()

x = df.drop(["Survived", "Name", "Ticket", "PassengerId", "Embarked"], axis=1)
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = model = RandomForestClassifier(n_estimators=50,  max_depth=5, min_samples_split=10, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"{accuracy * 100:.2f}%")



# from sklearn.model_selection import GridSearchCV

# rf = RandomForestClassifier(random_state=42)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# grid_search.fit(x_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Accuracy:", grid_search.best_score_)