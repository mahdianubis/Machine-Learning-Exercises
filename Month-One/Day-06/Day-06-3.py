import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Day-06-3.csv", encoding='latin1')

df = df.replace(',', '', regex=True) 

df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])

df = df.drop(["in_shazam_charts", "track_name", "key", "artist(s)_name"], axis=1)
df = df.dropna()

df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})

df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])

x = df.drop("mode", axis=1).astype(float)
y = df["mode"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)

print(accuracy)

