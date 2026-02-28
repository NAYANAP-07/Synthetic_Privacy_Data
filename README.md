# 🔐 Synthetic Data Privacy Verification Engine

**Synthetic Data Privacy Verification Engine** is a web-based AI tool that evaluates the privacy risks of a dataset and generates safer synthetic datasets for sharing. It quantifies risks such as re-identification, membership inference, and statistical similarity, providing a final privacy score to guide data sharing decisions.

---

## 💡 Problem Statement

Organizations and researchers often need to share datasets but risk leaking sensitive information. This tool helps measure and minimize privacy risks while enabling safe data sharing.

---

## 🚀 Key Features

- Upload CSV or Excel datasets containing numeric columns.
- Automatically clean data (handle NaN and infinite values).
- Generate synthetic datasets using controlled noise levels.
- Measure:
  - **Statistical Similarity** between real and synthetic data.
  - **Re-identification Risk** of sensitive records.
  - **Membership Inference Risk** using AI-based models.
- Calculate an overall **Privacy Score** to assess dataset safety.
- Visualize **Privacy–Utility Tradeoff** using a horizontal chart.
- Highlight the **primary privacy vulnerability** in the dataset.

---

## 🏗️ Technology Stack

- **Backend & Web Framework:** Python, Flask  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning & AI:** scikit-learn (Random Forest Classifier, Pairwise Distances)  
- **Visualization:** Matplotlib  
- **Frontend:** HTML, CSS, Jinja2 templates  

---




