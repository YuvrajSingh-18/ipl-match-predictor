import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏")
st.title("🏏 IPL Match Outcome Predictor")

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

teams = sorted(pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique().tolist())
venues = sorted(df['venue'].unique().tolist())
decisions = sorted(df['toss_decision'].unique().tolist())

for col in features:
    le = LabelEncoder()
    if col in ['team1', 'team2', 'toss_winner']:
        le.fit(teams)
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

# 4. Streamlit UI
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", teams, index=0)

team2_options = [t for t in teams if t != team1]
with col2:
    team2 = st.selectbox("Select Team 2", team2_options, index=0)

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", decisions)
venue = st.selectbox("Match Venue", venues)

if st.button("Predict Match Winner"):
    sample = pd.DataFrame([{
        'team1_enc': encoders['team1'].transform([team1])[0],
        'team2_enc': encoders['team2'].transform([team2])[0],
        'toss_winner_enc': encoders['toss_winner'].transform([toss_winner])[0],
        'toss_decision_enc': encoders['toss_decision'].transform([toss_decision])[0],
        'venue_enc': encoders['venue'].transform([venue])[0],
    }])

    probs = clf.predict_proba(sample)[0]
    classes = list(target_encoder.classes_)

    t1_prob = probs[classes.index(team1)] if team1 in classes else 0.0
    t2_prob = probs[classes.index(team2)] if team2 in classes else 0.0

    total_prob = t1_prob + t2_prob
    if total_prob > 0:
        win_pct1 = round((t1_prob / total_prob) * 100, 1)
        win_pct2 = round((t2_prob / total_prob) * 100, 1)
    else:
        win_pct1 = 50.0
        win_pct2 = 50.0

    if win_pct1 >= win_pct2:
        winner = team1
        chance = win_pct1
    else:
        winner = team2
        chance = win_pct2

    st.success(f"🏆 Predicted Winner: **{winner}** ({chance}% win chance)")
