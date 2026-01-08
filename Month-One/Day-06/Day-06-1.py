import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Day-06-1.csv")

x = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)

# print(score)


new = [[4.1, 2.1, 2.5, 1.2]]
new_pred = model.predict(new)
# print(new_pred)


# importances = model.feature_importances_
# feature_names = x.columns

# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# # print(importance_df.sort_values(by='Importance', ascending=False))
