import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

# Load dataset
df = pd.read_csv("Day-01-2.csv")
# --- Data Preprocessing & Feature Engineering ---

df["date"] = pd.to_datetime(df["date"]).dt.year
df["renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)
df["Price_per_sqft"] = df["price"] / df["sqft_living"]
df['house_age_at_sale'] = df['date'] - df['yr_built']

# Binning: Categorize houses by size

bins = [0, 1500, 3000, df['sqft_living'].max()]
labels = ['Small', 'Medium', 'Large']
df['house_size_cat'] = pd.cut(df['sqft_living'], bins=bins, labels=labels)

# --- Outlier Detection ---

mean_price = df['price'].mean()
std_price = df['price'].std()
outliers = df[np.abs(df['price'] - mean_price) > (3 * std_price)]
print(f"Number of outliers detected: {len(outliers)}")

Z_Score = (df["price"] - df["price"].mean()) / df["price"].std()
outliers = df[(Z_Score > 3) | (Z_Score < -3)]
print(f"Number of outliers detected: {len(outliers)}")

# Large houses and prices lower than the city average
lucky_houses = df[(df["price"] < df["price"].mean()) & (df["house_size_cat"] == "Large")]

# Renovated homes with more than two bathrooms and prices below the city average
m_houses = df[(df["price"] < df["price"].mean()) & (df["renovated"] == 1) & (df["bathrooms"] > 2)]

# --- Visualization Section ---


# cols = ['price', 'sqft_living', 'floors', 'house_age_at_sale']
# sns.pairplot(data=df, vars=cols, hue="house_size_cat", palette="viridis")
# plt.show()

# sns.kdeplot(data=df, x="price", hue="waterfront", fill=True, common_norm=False)
# plt.title("Price Density: Waterfront vs Normal Houses")
# plt.show()

# a = df[df["yr_built"] >= 2010]
# sns.boxplot(data=a, x="floors", y="price", palette="viridis")
# plt.title("Price Distribution based on number of floors (Post-2010)")
# plt.show()

# sns.barplot(data=df, x="view", y="price")
# plt.show()

# sns.boxplot(data=df, x='house_size_cat', y='price', palette="Set2")
# plt.title("Price Distribution by House Size Category")
# plt.show()

# g = sns.FacetGrid(df, col="renovated", height=5)
# g.map(sns.histplot, "price")
# plt.show()


# sns.regplot(data=df, x="sqft_living", y="price", scatter_kws={'alpha':0.3})
# plt.title("The relationship between volume and price")
# plt.show()

# sns.violinplot(data=df, x="bedrooms", y="price", inner="quartile")
# plt.title("Price distribution based on the number of bedrooms")
# plt.show()

# g = sns.FacetGrid(df, col="renovated", height=5)
# g.map_dataframe(sns.histplot, x="price", hue="waterfront", kde=True)
# g.add_legend()
# plt.show()


# --- Final storage ---
df.to_csv("King_County_Cleaned.csv", index=False)
