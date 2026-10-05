import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Dataset
data = {
    'team1': ['CSK', 'MI', 'RCB', 'KKR', 'CSK', 'MI', 'RCB', 'DC', 'CSK', 'KKR', 'GT', 'RR', 'CSK', 'MI'],
    'team2': ['MI', 'RCB', 'KKR', 'DC', 'RCB', 'KKR', 'CSK', 'MI', 'GT', 'RR', 'RR', 'CSK', 'DC', 'GT'],
    'toss_winner': ['CSK', 'MI', 'RCB', 'DC', 'CSK', 'KKR', 'CSK', 'DC', 'GT', 'KKR', 'GT', 'CSK', 'CSK', 'MI'],
    'toss_decision': ['bat', 'field', 'field', 'bat', 'bat', 'field', 'field', 'bat', 'field', 'field', 'bat', 'field', 'bat', 'field'],
    'venue': ['Chennai', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 'Mumbai', 'Bengaluru', 'Delhi', 'Ahmedabad', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Chennai', 'Mumbai'],
    'winner': ['CSK', 'MI', 'KKR', 'DC', 'CSK', 'MI', 'CSK', 'DC', 'GT', 'KKR', 'GT', 'CSK', 'CSK', 'MI']
}

df = pd.DataFrame(data)

# 2. Encode features
encoders = {}
features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']

for col in features:
    le = LabelEncoder()
    if col in ['team1', 'team2', 'toss_winner']:
        unique_teams = pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique()
        le.fit(unique_teams)
    else:
        le.fit(df[col])
    encoders[col] = le
    df[col + '_enc'] = le.transform(df[col])

target_encoder = LabelEncoder()
df['winner_enc'] = target_encoder.fit_transform(df['winner'])

X = df[[col + '_enc' for col in features]]
y = df['winner_enc']

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict Function
def predict_match(team1, team2, toss_winner, toss_decision, venue):
    sample = pd.DataFrame([{
        'team1_enc': encoders['team1'].transform([team1])[0],
        'team2_enc': encoders['team2'].transform([team2])[0],
        'toss_winner_enc': encoders['toss_winner'].transform([toss_winner])[0],
        'toss_decision_enc': encoders['toss_decision'].transform([toss_decision])[0],
        'venue_enc': encoders['venue'].transform([venue])[0]
    }])
    pred_class = clf.predict(sample)[0]
    return target_encoder.inverse_transform([pred_class])[0]

print("Predicted Winner (CSK vs MI in Chennai):", predict_match('CSK', 'MI', 'CSK', 'bat', 'Chennai'))
