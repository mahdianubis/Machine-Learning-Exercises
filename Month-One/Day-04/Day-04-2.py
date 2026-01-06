import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


df = pd.read_csv("Day-04-2.csv")

con = {"Male" : 0, "Female" : 1, "Other" : 2}
df["gender"] = df["gender"].map(con)

bins = [0, 20, 35, 100]
labels = ['Teen', 'Young', 'Adult']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

churn_report = df.groupby('is_churned')[['age', 'listening_time', 'ads_listened_per_week', 'skip_rate']].mean()

loyalty = df.groupby('subscription_type')['listening_time'].mean().sort_values(ascending=False)

offline_impact = df.groupby('offline_listening')['songs_played_per_day'].mean()

df['ads_per_hour'] = df['ads_listened_per_week'] / (df['listening_time']) 
print(df.head())

# sns.barplot(data=df, x='country', y='listening_time', hue='is_churned')
# plt.show()

# sns.scatterplot(data=df, x="listening_time", y="skip_rate")
# plt.show()

# fig = px.sunburst(df, path=["country", "subscription_type", "gender"])
# fig.show()
