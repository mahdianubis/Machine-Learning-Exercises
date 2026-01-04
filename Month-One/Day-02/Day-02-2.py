import pandas as pd 
import plotly.express as px

df = pd.read_csv("Day-02-2.csv")

df[["director", "cast"]] = df[["director", "cast"]].fillna("Unknown")
df["country"] = df["country"].fillna("Not Specified")

df["main_genre"] = df["listed_in"].str.split(",").str[0]
df = df.drop(["listed_in", "date_added", "description"], axis=1)

df['duration_num'] = df['duration'].str.split(' ').str[0].fillna(0).astype(int)
df['duration_type'] = df['duration'].str.split(' ').str[1] 


# print(df.head())

# fig1 = px.sunburst(df,path=['type', 'main_genre'], values='release_year')
# fig1.show()


# country_counts = df['country'].str.split(', ').explode().value_counts().reset_index()
# country_counts.columns = ['country', 'count']
# fig2 = px.choropleth(country_counts, locations="country", locationmode='country names', color="count", hover_name="country")
# fig2.show()
