import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

model = KNeighborsClassifier(n_neighbors=3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_pred, y_test)

print(accuracy)

