<<<<<<< HEAD
# 💼 Tech Salary Predictor

A Machine Learning web app that predicts **exact salaries** for tech jobs based on role, experience, education, and more — built with Python, Scikit-learn, and Streamlit.

---

## 🚀 Live Demo

> Deploy on Streamlit Cloud and paste your URL here:
> `https://yourname-salary-predictor.streamlit.app`

---

## 📸 Features

- 🎯 Predicts **exact annual salary** for 38+ tech job roles
- 📊 Shows **low / predicted / high** salary range
- 💵 Displays **monthly salary** breakdown
- ⚡ Instant predictions — no file upload needed
- 🌙 Clean dark UI built with Streamlit

---

## 🗂️ Project Structure

```
Salary-Prediction/
│
├── Salary_Data.csv          # Raw dataset (6,700+ rows)
├── salary_analysis.ipynb    # Full EDA + model training notebook (Steps 1–10)
├── tech_job_model.pkl       # Trained model bundle (auto-generated)
├── app.py                   # Streamlit web app
├── requirements.txt         # Python dependencies
└── README.md                # You are here
```

---

## 📓 Notebook Pipeline (salary_analysis.ipynb)

| Step | Description |
|------|-------------|
| Step 1 | Load Dataset |
| Step 2 | Basic EDA — shape, dtypes, missing values |
| Step 3 | Data Cleaning — duplicates, missing values, normalization |
| Step 4 | Filter Tech Jobs only |
| Step 5 | Visualizations — histograms, countplot, heatmap |
| Step 6 | Feature Engineering — encoding, scaling |
| Step 7 | Model Building — Random Forest Classifier |
| Step 8 | Evaluation — accuracy, F1, confusion matrix |
| Step 9 | Save cleaned CSV + model pkl |
| Step 10 | Retrain as **Regressor** + save full bundle for Streamlit |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Features | Age, Years of Experience, Gender, Education Level, Job Title |
| Target | Salary (USD) |
| Train/Test Split | 80% / 20% |
| Estimators | 200 |
| Max Depth | 10 |
| MAE | ~$8,000 |
| R² Score | ~0.92 |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/salary-predictor.git
cd salary-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `salary_analysis.ipynb` in VS Code or Jupyter and run all cells including **Step 10** to generate `tech_job_model.pkl`.

### 4. Launch the app
```bash
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Requirements

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
```

Install all at once:
```bash
pip install streamlit scikit-learn numpy pandas matplotlib seaborn
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New App** → select your repo → set main file to `app.py`
5. Click **Deploy** — you'll get a free public URL in minutes

> ⚠️ Make sure `tech_job_model.pkl` is committed and pushed to GitHub before deploying.

---

## 📊 Dataset

- **Source:** Salary_Data.csv
- **Total rows:** ~6,700
- **Columns:** Age, Gender, Education Level, Job Title, Years of Experience, Salary
- **Tech rows after filtering:** ~849

### Tech Job Roles Included
`Software Engineer` · `Data Scientist` · `Data Analyst` · `Machine Learning Engineer` · `AI Engineer` · `Data Engineer` · `DevOps Engineer` · `Cloud Engineer` · `Cybersecurity Analyst` · `Network Engineer` · `Web Developer` · `Full Stack Developer` · `Backend Developer` · `Frontend Developer` · `IT Manager` · `Chief Technology Officer` · `Chief Data Officer` · `Principal Engineer` · `Business Intelligence Analyst` · `UX Designer` · `UX Researcher` · and more...

---

## 👤 Author

**Zeesh** — [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
=======
# Tech-Salary-Predictor
Tech Salary Predictor is a machine learning project built in a Jupyter Notebook that predicts tech salaries using features like experience, role, and location. It covers data preprocessing, EDA, model training, and evaluation using Python and scikit-learn.
>>>>>>> 46417ce11e37f64539a1855fce5f82d07a49f078
