import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("Day-01-1.csv")

df = df.drop(["Cabin", "Ticket"], axis=1)
df = df.dropna(subset="Fare")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Gender_Num"] = np.where(df["Sex"] == "male", 0, 1)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["Legal_Age"] = df.apply(lambda x: "Adult" if x["Age"] >= 18 else "Child", axis=1)

Fare_mean = df.groupby("Pclass")["Fare"].mean()
plt.bar(Fare_mean.index, Fare_mean.values)
plt.title("Average ticket price for each class")
plt.show()

Embarked_m = df.groupby("Embarked")["PassengerId"].sum()
plt.pie(Embarked_m.values, labels=Embarked_m.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title("Percentage of passengers based on boarding location")
plt.show()

# ---------------------------------new---------------------------------------
# # style : white , darkgrid, whitegrid, ticks, dark
# sns.set_theme(style="darkgrid")

# sns.scatterplot(data=df, x="Age", y="Fare", hue="Pclass", palette="viridis")
# plt.title("Relationship between Age, Fare and Class")
# plt.show()

# sns.barplot(data=df, x="Pclass", y="Age", hue="Sex")
# plt.title("Average Age by Class and Gender")
# plt.show()

# sns.histplot(data=df, x="Age", kde=True, color="purple")
# plt.title("Age Distribution of Passengers")
# plt.show()

# # We understand which columns are related to each other.
# numeric_df = df.select_dtypes(include=[np.number])
# corr = numeric_df.corr()
# sns.heatmap(corr, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x="Embarked", palette="Set2")
# plt.title("Number of Passengers per Embarkation Point")
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x="Pclass", y="Fare", palette="pastel")
# plt.title("Fare Distribution by Ticket Class")
# plt.show()


# sns.pairplot(df, hue="Pclass", palette="bright")
# plt.show()

# numeric_df = df.select_dtypes(include=[np.number])
# corr = numeric_df.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
# plt.show()