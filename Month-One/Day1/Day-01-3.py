import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Load dataset

df = pd.read_csv("Day-01-3.csv")


# --- Data Preprocessing & Feature Engineering ---


df = df.drop(["Id"], axis=1)

df["Sepal_Area"] = df["SepalLengthCm"] * df["SepalWidthCm"]
df["Petal_Area"] = df["PetalLengthCm"] * df["PetalWidthCm"]

df["PetalLength_Normalized"] = (df["PetalLengthCm"] - df["PetalLengthCm"].min()) / (df["PetalLengthCm"].max() - df["PetalLengthCm"].min())

species_stats = df.groupby("Species")[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].describe()

print(len(df[(df["PetalLengthCm"] > df["PetalLengthCm"].mean()) & (df["PetalWidthCm"] < df["PetalWidthCm"].mean())]))
 
print("Species list:", df["Species"].unique())

pivot_report = df.pivot_table(index="Species", values=["Sepal_Area", "Petal_Area"], aggfunc="mean")


# --- Visualization Section ---

# sns.boxplot(data=df, x="SepalLengthCm", y="Species", palette="muted")
# plt.title("Distribution of Sepal Length by Species")
# plt.show()

# sns.scatterplot(data=df, x="PetalWidthCm", y="PetalLengthCm", hue="Species", style="Species")
# plt.title("Petal Length vs Width (Species Differentiation)")
# plt.show()

# sns.pairplot(data=df, hue="Species", diag_kind="kde")
# plt.suptitle("Pairwise Relationships of Iris Features")
# plt.show()

# sns.violinplot(data=df, x="Species", y="PetalLengthCm", inner="quartile")
# plt.title("Violin Plot: Petal Length Density by Species")
# plt.show()

# plt.figure(figsize=(10, 7))
# sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="magma")
# plt.title("Final Correlation Map")
# plt.show()
