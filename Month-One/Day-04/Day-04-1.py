import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv("divar_dataset.csv")

df[["has_elevator", "has_balcony", "has_storage", "has_parking"]] = np.where(df[["has_elevator", "has_balcony", "has_storage", "has_parking"]] == True, 1, 0)

df["price_total_billion"] = df["price_total"] / 1_000_000_000

df["is_luxury"] = df["description"].str.contains("مستر|روف گاردن|برند", na=False).astype(int)

df["price_per_meter_num"] = df["price_per_meter_text"].str.replace(r'[^0-9]', '', regex=True)
df["price_per_meter_num"] = pd.to_numeric(df["price_per_meter_num"], errors='coerce')
df = df.drop("price_per_meter_text", axis=1)

# print(df["price_per_meter_num"])

# fig = px.scatter(df, x="build_year", y="price_per_meter_num")
# fig.show()
# -------------------------------------------------------------------------------------
df = df[(df['price_per_meter_num'] > 20_000_000) & (df['price_per_meter_num'] < 600_000_000)]

a = df.groupby('has_parking')['price_per_meter_num'].mean()
b = df.groupby('has_balcony')['price_per_meter_num'].mean()
c = df.groupby('has_storage')['price_per_meter_num'].mean()

d = df.groupby('build_year')['price_per_meter_num'].mean()

df["urgent"] = df["description"].str.contains("زیر قیمت|پول لازم|فوری", na=False).astype(int)

bins = [0, 60, 120, df["area_m2"].max()]
labels = ["small", "medium", "large"]
df["size"] = pd.cut(df["area_m2"], bins=bins, labels=labels)

df["floor"] = df["floor"].astype(str)
df["floor"] = df["floor"].apply(lambda x: x.split(" از ")[0] if " از " in x else x)
df["floor"] = df["floor"].replace({"همکف": "0", "زیرزمین": "-1"})
df["floor"] = df["floor"].str.extract('(\d+)').fillna(0)
df["floor"] = df["floor"].astype(int)

# fig = px.line(df, x="build_year", y="price_per_meter_num")
# fig.show()

# fig = px.box(df, x="district_persian", y="price_per_meter_num")
# fig.show()

# sns.barplot(data=df, x="urgent", y="price_per_meter_num")
# plt.show()

# sns.barplot(data=df, x="floor", y="price_per_meter_num", hue="is_luxury")
# plt.show()

# sns.barplot(data=df, x="floor", y="price_per_meter_num", hue="urgent")
# plt.show()

# sns.scatterplot(df, x="area_m2", y="price_per_meter_num", size="rooms", hue="build_year")
# plt.show()

# plt.figure(figsize=(10, 6))
# numeric_cols = df.select_dtypes(include=[np.number])
# sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.show()