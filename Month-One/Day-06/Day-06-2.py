import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)


print(f"{accuracy * 100:.2f}%")



# importances = model.feature_importances_
# feature_names = x.columns

# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# # print(importance_df.sort_values(by='Importance', ascending=False))
