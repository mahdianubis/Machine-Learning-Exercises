import pandas as pd 
import plotly.express as px

df = pd.read_csv("Day-03-1.csv")

df["Profit_Margin"] = df["Profit"] / df["Sales"]
df["Efficiency"] = df["Profit"] / df["Quantity"]

ship_mode_map = {'Standard Class': 0, 'Second Class': 1, 'First Class': 2, 'Same Day': 3}
df['Ship_Mode_Code'] = df['Ship Mode'].map(ship_mode_map)

segment_map = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}
df['Segment_Code'] = df['Segment'].map(segment_map)

region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
category_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False).reset_index()

# fig_state = px.bar(df, x="State", y="Profit", color="Region", title="State-wise Profit/Loss Analysis")
# fig_state.show()

# loss_df = df[df['Profit'] < 0]
# fig_loss = px.sunburst(loss_df, path=['Region', 'Category', 'Sub-Category'], values='Sales', title="Deep Dive into Loss-Making Sectors")
# fig_loss.show()

# fig_margin = px.scatter(df, x="Sales", y="Profit_Margin")
# fig_margin.show()

