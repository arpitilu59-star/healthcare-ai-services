import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Chhota sa dataset banate hain (Symptoms: Fever, Cough, Headache, Fatigue)
# 1 = Yes, 0 = No
data = {
    'Fever':    [1, 0, 1, 0, 1, 0],
    'Cough':    [1, 1, 0, 0, 1, 0],
    'Headache': [0, 1, 1, 0, 0, 1],
    'Fatigue':  [1, 1, 1, 0, 0, 0],
    'Disease':  ['Flu', 'Cold', 'Migraine', 'Healthy', 'Flu', 'Healthy']
}

df = pd.DataFrame(data)

# 2. X = Symptoms, y = Disease
X = df.drop('Disease', axis=1)
y = df['Disease']

# 3. Model train karte hain
model = RandomForestClassifier()
model.fit(X, y)

# 4. Model ko save karte hain taaki app.py use kar sake
joblib.dump(model, 'medical_model.pkl')
print("Model trained and saved as medical_model.pkl!")