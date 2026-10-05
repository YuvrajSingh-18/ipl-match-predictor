# 🏏 IPL Match Outcome Predictor & Analytics

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project designed to analyze historical Indian Premier League (IPL) fixtures and predict match winners using fixture conditions, venue trends, and toss decisions.

---

## 📌 Project Overview
- **Data Preprocessing & Encoding:** Cleans categorical match variables and encodes team names, venues, and toss outcomes into numerical formats suitable for machine learning.
- **Exploratory Data Analysis:** Evaluates key winning indicators such as toss impact and venue bias.
- **Classification Modeling:** Employs a Scikit-Learn **Random Forest Classifier** to assess feature importance and predict winning probabilities.
- **Custom Inference Function:** Provides a reusable function (`predict_match`) to simulate upcoming head-to-head fixtures.

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YuvrajSingh-18/ipl-match-predictor.git
   cd ipl-match-predictor
   ```

2. **Install required dependencies:**
```bash
pip install pandas scikit-learn
```

3. **Run the prediction model:**
```bash
python ipl_predictor.py
```

---

## 📊 Sample Inference
```python
predict_match(team1='CSK', team2='MI', toss_winner='CSK', toss_decision='bat', venue='Chennai')
# Output: Predicted Winner: CSK
```
