import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load & clean
df = pd.read_csv("Salary_Data.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
edu_map = {"Bachelor's Degree": "Bachelor's", "Master's Degree": "Master's"}
df["Education Level"] = df["Education Level"].replace(edu_map)

# Tech filter
TECH_KEYWORDS = [
    'software','developer','data scientist','data analyst','machine learning',
    'ml engineer','ai engineer','data engineer','devops','cloud','cybersecurity',
    'network engineer','systems engineer','database','it support','it manager',
    'web developer','backend','frontend','full stack','principal engineer',
    'senior engineer','chief technology','chief data','technical',
    'business intelligence','junior developer','junior software','junior web',
    'junior data','senior software','senior data','senior engineer',
    'software project manager','software manager','ux researcher','ux designer',
]
def is_tech(t):
    t = str(t).lower()
    return any(k in t for k in TECH_KEYWORDS)

tech_df = df[df["Job Title"].apply(is_tech)].copy().reset_index(drop=True)

# Encode
le_gender = LabelEncoder().fit(sorted(tech_df["Gender"].unique()))
le_edu    = LabelEncoder().fit(sorted(tech_df["Education Level"].unique()))
le_title  = LabelEncoder().fit(sorted(tech_df["Job Title"].unique()))

tech_df["Gender_enc"]    = le_gender.transform(tech_df["Gender"])
tech_df["Education_enc"] = le_edu.transform(tech_df["Education Level"])
tech_df["JobTitle_enc"]  = le_title.transform(tech_df["Job Title"])

FEATURES = ["Age", "Years of Experience", "Gender_enc", "Education_enc", "JobTitle_enc"]
X = tech_df[FEATURES]
y = tech_df["Salary"]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[["Age","Years of Experience"]] = scaler.fit_transform(X[["Age","Years of Experience"]])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE : ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R2  : {r2_score(y_test, y_pred):.4f}")

# Save everything needed by the app
bundle = {
    "model":     model,
    "scaler":    scaler,
    "le_gender": le_gender,
    "le_edu":    le_edu,
    "le_title":  le_title,
    "tech_jobs": sorted(tech_df["Job Title"].unique().tolist()),
    "genders":   sorted(tech_df["Gender"].unique().tolist()),
    "edu_levels":sorted(tech_df["Education Level"].unique().tolist()),
}
with open("tech_job_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved: tech_job_model.pkl")
print("Jobs in model:", bundle["tech_jobs"])
