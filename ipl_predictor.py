if st.button("Predict Match Winner"):
  sample = pd.DataFrame([{
      'team1_enc': encoders['team1'].transform([team1])[0],
      'team2_enc': encoders['team2'].transform([team2])[0],
      'toss_winner_enc': encoders['toss_winner'].transform([toss_winner])[0],
      'toss_decision_enc': encoders['toss_decision'].transform([toss_decision])[0],
      'venue_enc': encoders['venue'].transform([venue])[0],
  }])

  # Get probability distribution across all classes
  probs = clf.predict_proba(sample)[0]
  classes = target_encoder.classes_

  # Find probabilities strictly for the two playing teams
  t1_prob = probs[list(classes).index(team1)] if team1 in classes else 0.0
  t2_prob = probs[list(classes).index(team2)] if team2 in classes else 0.0

  # Winner is strictly the team with the higher score
  if t1_prob >= t2_prob:
    winner = team1
    win_pct = (
        round((t1_prob / (t1_prob + t2_prob)) * 100, 1)
        if (t1_prob + t2_prob) > 0
        else 50.0
    )
  else:
    winner = team2
    win_pct = (
        round((t2_prob / (t1_prob + t2_prob)) * 100, 1)
        if (t1_prob + t2_prob) > 0
        else 50.0
    )

  st.success(f"🏆 Predicted Winner: **{winner}** ({win_pct}% win chance)")
