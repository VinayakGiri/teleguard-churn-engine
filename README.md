# 📡 TeleGuard — AI Churn Intelligence Engine
### *Predict. Explain. Retain.*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini API](https://img.shields.io/badge/Gemini%201.5%20Flash-AI%20Insights-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-00c9a7?style=flat-square&logo=rocket&logoColor=white)](https://YOUR-DEMO-LINK-HERE)

</div>

---

> "A machine learning system that identifies which telecom customers are about to leave — and uses Gemini AI to explain why and what to do about it. Built entirely from scratch as a self-directed learning project."

---

## 🚀 Live Demo

👉 **[Launch TeleGuard Dashboard](https://YOUR-DEMO-LINK-HERE)**

Upload any Telco churn CSV and get real-time predictions, risk scores, and an AI-generated retention strategy in under 5 seconds.

---

## 📸 Screenshots

![TeleGuard dashboard showing batch churn predictions](screenshots/dashboard_batch.png)

---

## 🔍 The Problem

Telecom companies lose somewhere between 15% and 25% of their customers every year to churn. That sounds like a statistic until you calculate what it actually means: a mid-sized operator with a million subscribers and $65 average monthly revenue is haemorrhaging over $150 million annually to customers who quietly cancel and move on. Most of that loss is predictable — and therefore preventable.

The tools that exist to solve this are either expensive enterprise platforms locked behind six-figure contracts, or generic dashboards that tell you churn happened without telling you *why* or *who's next*. I was studying this space as part of my coursework at NMIMS Mumbai and kept hitting the same wall: there was a clear gap between what the research said was possible and what a small team or startup could actually build and deploy.

So I built TeleGuard. Not as an academic exercise, but as an attempt to answer a real question: what does a functional, end-to-end churn intelligence system look like when you build every piece yourself — the data pipeline, the models, the AI narrative layer, and the dashboard — from scratch? The answer is this repository.

---

## ✅ What It Does

- **Batch scoring** — Upload a CSV of customer records and score all of them in one pass, with a risk distribution chart and a ranked list of your highest-risk accounts
- **Individual profiling** — Enter any customer's profile and get an instant churn probability with a color-coded risk card and progress bar
- **Revenue exposure quantification** — Converts high-risk customer counts into actual monthly revenue at risk in rupees, not just percentages
- **Gemini AI retention strategy** — Feeds the model's top feature importances into Gemini 1.5 Flash, which returns a 3-point, plain-English retention strategy written for a CFO — no bullet symbols, no jargon
- **Model interpretability** — A dedicated performance tab shows accuracy and AUC for all three trained models alongside a teal-scale feature importance chart so the predictions are never a black box

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Best Model | Random Forest |
| Accuracy | **84.2%** |
| AUC Score | **0.847** |
| Dataset Size | 7,043 customers |
| Month-to-month churn rate | 42.7% |
| Two-year contract churn rate | 2.8% |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| ML Models | Scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Visualization | Plotly, Seaborn, Matplotlib |
| AI Narrative | Google Gemini 1.5 Flash via `google-genai` SDK |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud (or local via `streamlit run`) |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/teleguard-churn-engine.git
cd teleguard-churn-engine

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key (free at aistudio.google.com)
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the preprocessing pipeline
python3 src/preprocess.py

# 6. Train the models
python3 src/model.py

# 7. Launch the dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501** and upload `data/telco_raw.csv` to begin.

---

## 📁 Project Structure

```
teleguard-churn-engine/
│
├── data/
│   ├── telco_raw.csv              # Original Kaggle dataset (7,043 rows, 21 columns)
│   └── telco_cleaned.csv          # Preprocessed & one-hot encoded dataset
│
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis with Seaborn plots
│
├── src/
│   ├── preprocess.py              # Cleaning, encoding, and feature engineering
│   ├── model.py                   # Training pipeline — 3 models, AUC selection, artefact export
│   └── gemini_insights.py         # Gemini AI integration for retention strategy + customer explanation
│
├── dashboard/
│   └── app.py                     # Streamlit web app — 3 tabs, Plotly charts, real-time scoring
│
├── models/
│   ├── churn_model.pkl            # Best model (saved with joblib)
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── feature_names.pkl          # Training feature list for column alignment at inference
│   ├── top_features.json          # Top churn drivers with importance scores
│   └── model_results.json         # Accuracy and AUC for all 3 models
│
├── screenshots/                   # Dashboard screenshots for README and LinkedIn
├── requirements.txt
└── README.md
```

---

## 🧠 What I Learned

- **TotalCharges** was stored as a string with blank spaces for new customers — not NaN, not zero, just `" "`. Standard `.dropna()` missed it entirely. The fix was a `pd.to_numeric(..., errors='coerce')` pass before any other cleaning, which became a permanent part of the preprocessing pipeline. That one bug taught more about real-world data quality than a semester of clean toy datasets.

- **Accuracy is a misleading metric for imbalanced classification.** The Telco dataset is ~73% non-churners, which means a model that predicts "No Churn" for everyone scores 73% accuracy without learning anything. AUC-ROC measures how well the model separates the two classes across all decision thresholds — it's the number that actually tells you whether the model is useful. This is why the project tracks AUC as the primary evaluation metric, not accuracy.

- **Bridging a traditional ML pipeline to an LLM requires you to think carefully about what the model actually produced, not just what it predicted.** Feeding raw probabilities into Gemini isn't useful. The insight came from extracting feature importances, ranking them, and constructing a structured prompt that gave Gemini the business context it needed to generate something actionable — rather than something generic. That prompt engineering process took longer than the model training.

- **The gap between a model's output and a business decision is where most ML projects quietly fail.** A probability score of 0.73 means nothing to a retention manager. The design decision to always show a color-coded risk label, a revenue figure in rupees, and a Gemini-written plain-English recommendation alongside the raw score was the most important UX choice in the project — not any hyperparameter.

---

## 📦 Dataset

**IBM Telco Customer Churn Dataset** — Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

7,043 customer records across 21 features including contract type, tenure, internet service, monthly charges, and churn label. No license restrictions for educational and portfolio use.

---

<div align="center">

Built by **Vinayak Giri** · AI & Data Science Student, NMIMS Mumbai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR-LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/YOUR-GITHUB)

</div>
