import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('heart.csv')

# 2. Preprocessing
X = df.drop(columns=['target'])
y = df['target']

# Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi
y_pred = model.predict(X_test)
print(f'Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# 5. Simpan Model
joblib.dump(model, 'model_jantung.sav')
print("Model disimpan sebagai 'model_jantung.sav'")