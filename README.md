# 🧠 Early Detection of Chronic Kidney Disease (CKD)
## 📍 Project Overview
Chronic Kidney Disease (CKD) affects **10% of the global population**, often going undetected until its later stages—where treatment is less effective and more costly. 
This project provides an end-to-end solution for **early detection and monitoring** of CKD using machine learning and an interactive dashboard, empowering health professionals and planners to **act earlier, 
allocate smarter, and save lives**.

## ❓ Why This Project?
This project was developed in response to the growing burden of Chronic Kidney Disease (CKD) and the need for proactive, data-driven healthcare solutions. Key motivations include:
- **Combat Delayed Diagnosis:** CKD is often asymptomatic in early stages, resulting in late detection and worsened outcomes. Our solution identifies risks before symptoms manifest.
- **Enable Proactive Intervention:** AI-driven predictive modeling flags at-risk patients earlier, allowing timely care to slow disease progression.
- **Reduce Systemic Strain:** Early detection cuts costly emergency interventions, alleviating pressure on hospitals and overburdened systems like the NHS.
- **Bridge Clinical Practice with AI:** Designed for healthcare professionals—this tool integrates seamlessly into workflows, enhancing diagnostic confidence.

## 📦 Data Source
- **Dataset:** [Kaggle Chronic Kidney Disease Dataset](https://www.kaggle.com/datasets/yasserhessein/chronic-kidney-disease)
- **Records:** 400 patient samples (250 with CKD, 150 without)
- **Features:** 26 clinical variables including age, blood pressure, albumin, hemoglobin, serum creatinine, hypertension, diabetes, etc.
- **Imbalance:** Handled via visualization awareness and model weighting strategies.

## 🧰 Actions Taken
**1. Data Cleaning & Preprocessing**
- Whitespace stripping, missing value imputation (mean for numerical, mode for categorical)
- Feature encoding (categorical to binary or ordinal)
- Exploratory data analysis (univariate, bivariate)
  
**2. Model Development**
- Trained and compared Random Forest and XGBoost classifiers
- Evaluated using accuracy, classification report, and ROC curves
- Addressed class imbalance with class_weight and stratified validation

**3. Dashboard Development**
- Built an interactive dashboard with Dash, Plotly, and Dash Bootstrap Components
- Included live predictions, data insights, and model performances views
- Integrated a FastAPI backend for real-time scoring

**4. Deployment-Ready Setup**
- Modular, scalable Python codebase
- Visual and interpretive elements for health professionals

**Top predictors:** Albumin, Serum Creatinine, Hypertension, Hemoglobin
Balanced performance with interpretability and speed

## 📊 Results

|Model	| Accuracy	| Precision	| Recall	|AUC|
| --- |--- |---| ----|
| Random Forest	| 94%	| 0.95	|0.93	|0.96|
| XGBoost |	92%	|0.92	|0.91	|0.94 |

## 🧠 Actionable Insights for Health Services
- **Early Flagging:** Patients with elevated serum creatinine or low hemoglobin can be prioritized for CKD follow-ups.
- **Triage Optimization:** Deploy in NHS 111-style systems for front-line risk scoring.
- **Rural Health Outreach:** Portable prediction tool for remote clinics without nephrology specialists.
- **Public Screening:** Focus on age 40–70 for effective population-level CKD surveillance.

## 📊 Dashboard Features 
[**DASHBOARD**](https://chronic-kidney-disease-dashboard.onrender.com/)
- **🧾 Prediction Tab:** Input patient data and get instant CKD risk prediction
- **🔍 Data Insights Tab:** Visualize trends, distributions, and bivariate analysis
- **📈 Model Performance:** ROC curves, feature importance plots, classification metrics
