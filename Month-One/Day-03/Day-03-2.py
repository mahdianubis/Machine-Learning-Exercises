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

print(df["price_per_meter_num"])

# fig = px.scatter(df, x="build_year", y="price_per_meter_text")
# fig.show()
