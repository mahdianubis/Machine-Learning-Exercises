import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


df = pd.read_csv("Day-04-3.csv")

def clean_recoverable(value):
    value = str(value)
    if '-' in value:
        parts = value.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    return float(value)

def s_years(value):
    value = str(value)
    if 's' in value:
        return value
    else:
        return value[:-1] + '0s'

df['Estimated_Recoverable_Reserves_Billion_Barrels'] = df['Estimated_Recoverable_Reserves_Billion_Barrels'].apply(clean_recoverable)
df["Rosneft"] = df["Major_Operators"].str.contains("Rosneft", na=False).astype(int)
df["CNPC"] = df["Major_Operators"].str.contains("CNPC", na=False).astype(int)
df["Chevron"] = df["Major_Operators"].str.contains("Chevron", na=False).astype(int)

a = df.groupby("Oil_Type_Grade")["Production_Capacity_Barrels_Day"].mean().sort_values(ascending=False)

df["Recovery_Rate_Percentage"] = df["Estimated_Recoverable_Reserves_Billion_Barrels"] / df["Proven_Reserves_Billion_Barrels"] * 100
df["Year_Discovered"] = df["Year_Discovered"].apply(s_years)

b = df.groupby("Production_Capacity_Barrels_Day")[["CNPC", "Rosneft", "Chevron"]].sum()

df["Has_Issues"] = df["Notes"].str.contains("infrastructure|aging|severe", na=False).astype(int)

# fig = px.treemap(df, path=["Reservoir_Name", "Basin_Region"])
# fig.show()

# sns.lineplot(data=df, x="Year_Discovered", y="Estimated_Recoverable_Reserves_Billion_Barrels")
# plt.show()

# sns.barplot(data=df, x="Status", y="Recovery_Rate_Percentage")
# plt.show()

# sns.scatterplot(data=df, x="Proven_Reserves_Billion_Barrels", y="Production_Capacity_Barrels_Day", size="Estimated_Recoverable_Reserves_Billion_Barrels")
# plt.show()
