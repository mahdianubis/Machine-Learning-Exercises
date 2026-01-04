import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Day-02-1.csv")
df = df.dropna()


# --- Data Preprocessing & Feature Engineering ---

top_5_delay_carriers = df.groupby("carrier_name")["arr_delay"].sum().sort_values(ascending=False).head()
top_5_cancal_carriers = df.groupby("carrier_name")["arr_cancelled"].sum().sort_values(ascending=False).head()


df["z_score"] = (df["arr_delay"] - df["arr_delay"].mean()) / df["arr_delay"].std()
outliers = df[(df["z_score"] > 3) | (df["z_score"] < -3)]
bad_counts = outliers["carrier_name"].value_counts()

# --- Visualization Section ---

top_5_weather_carriers = df.groupby("carrier_name")["weather_delay"].mean().sort_values(ascending=False).head(5).index
df_top5 = df[df["carrier_name"].isin(top_5_weather_carriers)]
# sns.boxplot(data=df_top5, x="carrier_name", y="weather_delay")
# plt.xticks(rotation=45)
# plt.title("Weather Delay Distribution for Top 5 Carriers")
# plt.show()

# sns.lineplot(data=df, x="month", y="arr_delay", marker='o')
# plt.title("Average Monthly Arrival Delays")
# plt.show()

# sns.scatterplot(data=df, x="arr_flights", y="arr_delay", hue="carrier_name", palette="rocket")
# plt.title("Flights Count vs Total Delay Minutes")
# plt.show()