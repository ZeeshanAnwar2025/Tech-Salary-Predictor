"""
============================================================
  Salary Dataset — Full EDA, Data Cleaning & ML Pipeline
  Focus: TECH Job Roles Classification
============================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import joblib

# ── Colour palette ───────────────────────────────────────────────────────────
PALETTE = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE)
BLUE   = "#4C72B0"
GREEN  = "#55A868"
RED    = "#C44E52"
ORANGE = "#DD8452"

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1  ─  LOAD & INITIAL INSPECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 1 — LOAD & INITIAL INSPECTION")
print("="*60)

df = pd.read_csv("/mnt/user-data/uploads/Salary_Data.csv")

print(f"\n▸ Dataset shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n▸ Column names   : {list(df.columns)}")
print("\n▸ Data types:\n", df.dtypes)
print("\n▸ First 5 rows:")
print(df.head().to_string())
print("\n▸ Last 5 rows:")
print(df.tail().to_string())

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2  ─  MISSING VALUES (BEFORE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 2 — MISSING VALUES")
print("="*60)

missing_before = df.isnull().sum()
missing_pct    = (missing_before / len(df) * 100).round(2)
missing_df     = pd.DataFrame({"Missing Count": missing_before,
                               "Missing %":     missing_pct})
print("\n▸ Before cleaning:\n", missing_df.to_string())

# Drop rows with any missing value (small fraction — < 0.1%)
df.dropna(inplace=True)

missing_after = df.isnull().sum()
print("\n▸ After dropping rows with missing values:\n",
      missing_after.to_string())
print(f"\n  Rows remaining: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3  ─  DUPLICATE REMOVAL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 3 — DUPLICATE REMOVAL")
print("="*60)

dup_count = df.duplicated().sum()
print(f"\n▸ Duplicate rows found: {dup_count}")
df.drop_duplicates(inplace=True)
print(f"  Rows after dedup      : {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4  ─  SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 4 — SUMMARY STATISTICS")
print("="*60)

print("\n▸ Numeric summary:\n", df.describe().round(2).to_string())
print("\n▸ Categorical summary:")
for col in ["Gender", "Education Level"]:
    print(f"\n  {col}:\n{df[col].value_counts()}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5  ─  OUTLIER DETECTION (IQR)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 5 — OUTLIER DETECTION (IQR)")
print("="*60)

num_cols = ["Age", "Years of Experience", "Salary"]
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"\n  {col}: Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f} "
          f"→ bounds [{lo:.0f}, {hi:.0f}] | outliers={n_out}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6  ─  EDA VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 6 — EDA VISUALISATIONS")
print("="*60)

# -- Fig 1: Histograms + KDE for numeric columns --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribution of Numeric Features", fontsize=15, fontweight="bold")
for ax, col, color in zip(axes, num_cols, [BLUE, GREEN, ORANGE]):
    df[col].plot.hist(ax=ax, bins=30, color=color, edgecolor="white",
                      alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("/home/claude/fig1_histograms.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig1_histograms.png")

# -- Fig 2: Boxplots for outlier visualisation --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Boxplots — Outlier View", fontsize=15, fontweight="bold")
for ax, col, color in zip(axes, num_cols, [BLUE, GREEN, ORANGE]):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_title(col)
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig("/home/claude/fig2_boxplots.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig2_boxplots.png")

# -- Fig 3: Countplots for categorical columns --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Categorical Feature Distributions", fontsize=15, fontweight="bold")
for ax, col in zip(axes, ["Gender", "Education Level"]):
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, palette=PALETTE, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/fig3_countplots.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig3_countplots.png")

# -- Fig 4: Salary by Education Level --
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("Education Level")["Salary"].median().sort_values().index
sns.boxplot(data=df, x="Education Level", y="Salary", order=order,
            palette=PALETTE, ax=ax)
ax.set_title("Salary Distribution by Education Level", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/fig4_salary_by_education.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig4_salary_by_education.png")

# -- Fig 5: Correlation heatmap (numeric only) --
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 12})
ax.set_title("Correlation Matrix (Numeric Features)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/fig5_correlation_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig5_correlation_heatmap.png")

# -- Fig 6: Salary vs Experience scatter --
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(df["Years of Experience"], df["Salary"],
                c=df["Age"], cmap="viridis", alpha=0.45, edgecolors="none", s=18)
plt.colorbar(sc, ax=ax, label="Age")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary ($)")
ax.set_title("Salary vs Experience (coloured by Age)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/fig6_salary_vs_experience.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig6_salary_vs_experience.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7  ─  DATA CLEANING: KEEP TECH ROLES ONLY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 7 — FILTER TECH JOB ROLES ONLY")
print("="*60)

# Comprehensive list of tech job keywords
TECH_KEYWORDS = [
    "software", "data scientist", "data analyst", "data engineer",
    "machine learning", "ml engineer", "ai engineer",
    "developer", "engineer", "full stack", "back end", "front end",
    "devops", "cloud", "network engineer", "it manager", "it support",
    "cybersecurity", "security engineer", "database", "systems",
    "web developer", "web designer", "ux designer", "ux researcher",
    "product designer", "technical support", "technical writer",
    "business intelligence", "bi analyst", "help desk",
    "junior software", "senior software", "principal engineer",
    "director of engineering", "senior data", "junior data",
    "junior developer", "senior developer", "research scientist",
    "junior research", "senior research"
]

def is_tech_role(title: str) -> bool:
    """Return True if any tech keyword appears in the lowercased job title."""
    t = str(title).lower()
    return any(kw in t for kw in TECH_KEYWORDS)

rows_before = len(df)
df_tech = df[df["Job Title"].apply(is_tech_role)].copy()
rows_after = len(df_tech)
rows_removed = rows_before - rows_after

print(f"\n▸ Rows BEFORE tech filter : {rows_before}")
print(f"  Rows REMOVED (non-tech)  : {rows_removed}")
print(f"  Rows REMAINING (tech)    : {rows_after}")
print(f"\n▸ Tech job titles retained ({df_tech['Job Title'].nunique()} unique):")
for t in sorted(df_tech["Job Title"].unique()):
    print(f"    • {t}")

# -- Fig 7: Top 15 Tech Job Roles --
fig, ax = plt.subplots(figsize=(11, 6))
top_roles = df_tech["Job Title"].value_counts().head(15)
top_roles.sort_values().plot.barh(ax=ax, color=BLUE, edgecolor="white")
ax.set_title("Top 15 Tech Job Roles (Count)", fontsize=13, fontweight="bold")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("/home/claude/fig7_top_tech_roles.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n  ✔ Saved fig7_top_tech_roles.png")

# -- Fig 8: Salary by top tech roles --
top10 = df_tech["Job Title"].value_counts().head(10).index.tolist()
df_top10 = df_tech[df_tech["Job Title"].isin(top10)]
fig, ax = plt.subplots(figsize=(12, 6))
order_sal = df_top10.groupby("Job Title")["Salary"].median().sort_values(ascending=False).index
sns.boxplot(data=df_top10, x="Job Title", y="Salary", order=order_sal,
            palette=PALETTE, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_title("Salary Distribution — Top 10 Tech Roles",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/fig8_salary_top_tech.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig8_salary_top_tech.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8  ─  FEATURE ENGINEERING & ENCODING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 8 — FEATURE ENGINEERING & ENCODING")
print("="*60)

# Group rare job titles (< 10 occurrences) to avoid too many tiny classes
min_samples = 10
counts = df_tech["Job Title"].value_counts()
valid_titles = counts[counts >= min_samples].index.tolist()
df_tech = df_tech[df_tech["Job Title"].isin(valid_titles)].copy()
print(f"\n▸ After removing classes with <{min_samples} samples:")
print(f"  Remaining rows       : {len(df_tech)}")
print(f"  Unique job titles    : {df_tech['Job Title'].nunique()}")

# Normalise Education Level variants
def normalise_edu(val):
    v = str(val).strip().lower()
    if "high" in v: return "High School"
    if "bachelor" in v: return "Bachelor's"
    if "master" in v: return "Master's"
    if "phd" in v or "ph.d" in v: return "PhD"
    return "Bachelor's"
df_tech["Education Level"] = df_tech["Education Level"].apply(normalise_edu)

# Ordinal encode Education Level
edu_order = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df_tech["Education_Encoded"] = df_tech["Education Level"].map(edu_order)
# Fallback for any unseen level
df_tech["Education_Encoded"] = df_tech["Education_Encoded"].fillna(1)

# Label encode Gender (binary)
le_gender = LabelEncoder()
df_tech["Gender_Encoded"] = le_gender.fit_transform(df_tech["Gender"])

# Target: label encode Job Title
le_target = LabelEncoder()
df_tech["Target"] = le_target.fit_transform(df_tech["Job Title"])

print("\n▸ Encoding map — Education Level:")
print(edu_order)
print("\n▸ Encoding map — Gender:")
for cls, idx in zip(le_gender.classes_, range(len(le_gender.classes_))):
    print(f"  {cls} → {idx}")
print(f"\n▸ Target classes ({len(le_target.classes_)}):")
for idx, cls in enumerate(le_target.classes_):
    print(f"  {idx} → {cls}")

# Feature matrix
FEATURES = ["Age", "Gender_Encoded", "Education_Encoded",
            "Years of Experience", "Salary"]
X = df_tech[FEATURES].values
y = df_tech["Target"].values

print(f"\n▸ Feature matrix shape : {X.shape}")
print(f"  Target vector shape  : {y.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9  ─  SAVE CLEANED DATASET
# ═══════════════════════════════════════════════════════════════════════════
out_cols = ["Age", "Gender", "Education Level", "Job Title",
            "Years of Experience", "Salary",
            "Gender_Encoded", "Education_Encoded", "Target"]
df_tech[out_cols].to_csv("/home/claude/tech_jobs_cleaned.csv", index=False)
print("\n  ✔ Saved tech_jobs_cleaned.csv")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10  ─  TRAIN / TEST SPLIT & SCALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 10 — TRAIN / TEST SPLIT & SCALING")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only
X_test_sc  = scaler.transform(X_test)        # transform test — no leakage

print(f"\n▸ Train set : {X_train.shape[0]} rows")
print(f"  Test set  : {X_test.shape[0]}  rows")
print("  Scaler fitted on train data only (no data leakage).")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11  ─  MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 11 — MODEL TRAINING & EVALUATION")
print("="*60)

# ── Model A: Random Forest (handles multi-class well, no scaling needed) ──
print("\n── Random Forest ──")
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average="weighted")
cv_rf  = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()

print(f"  Accuracy (test) : {acc_rf:.4f}")
print(f"  F1-score (wt.)  : {f1_rf:.4f}")
print(f"  CV Accuracy (5-fold): {cv_rf:.4f}")
print("\n  Classification Report (RF):")
print(classification_report(y_test, y_pred_rf,
                             target_names=le_target.classes_))

# ── Model B: Logistic Regression (scaled data) ──────────────────────────
print("\n── Logistic Regression ──")
lr = LogisticRegression(max_iter=2000, C=1.0,
                        solver="lbfgs", random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr  = f1_score(y_test, y_pred_lr, average="weighted")
cv_lr  = cross_val_score(lr, X_train_sc, y_train, cv=5,
                         scoring="accuracy").mean()

print(f"  Accuracy (test) : {acc_lr:.4f}")
print(f"  F1-score (wt.)  : {f1_lr:.4f}")
print(f"  CV Accuracy (5-fold): {cv_lr:.4f}")
print("\n  Classification Report (LR):")
print(classification_report(y_test, y_pred_lr,
                             target_names=le_target.classes_))

# ── Pick best model ──────────────────────────────────────────────────────
best_model_name = "Random Forest" if f1_rf >= f1_lr else "Logistic Regression"
best_model      = rf               if f1_rf >= f1_lr else lr
best_preds      = y_pred_rf        if f1_rf >= f1_lr else y_pred_lr
print(f"\n▸ Best model selected: {best_model_name} (F1={max(f1_rf, f1_lr):.4f})")

# ── Fig 9: Feature importance (RF) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(ax=ax, color=BLUE, edgecolor="white")
ax.set_title("Feature Importance — Random Forest",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/fig9_feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n  ✔ Saved fig9_feature_importance.png")

# ── Fig 10: Confusion matrix (best model, top-10 classes) ───────────────
# Show top-10 most frequent classes for readability
top10_idx = np.argsort(np.bincount(y_test))[-10:]
mask_test  = np.isin(y_test, top10_idx)
yt_sub     = y_test[mask_test]
yp_sub     = best_preds[mask_test]
label_names_sub = le_target.inverse_transform(top10_idx)

cm = confusion_matrix(yt_sub, yp_sub, labels=top10_idx)
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names_sub, yticklabels=label_names_sub, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix ({best_model_name}) — Top-10 Tech Roles",
             fontsize=13, fontweight="bold")
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("/home/claude/fig10_confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig10_confusion_matrix.png")

# ── Fig 11: Model accuracy comparison bar chart ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
models  = ["Random Forest", "Logistic Regression"]
accs    = [acc_rf, acc_lr]
colors  = [GREEN if a == max(accs) else BLUE for a in accs]
bars = ax.bar(models, accs, color=colors, edgecolor="white", width=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/fig11_model_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔ Saved fig11_model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12  ─  SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════════
model_bundle = {
    "model":         best_model,
    "scaler":        scaler,
    "label_encoder": le_target,
    "gender_encoder":le_gender,
    "features":      FEATURES,
    "model_name":    best_model_name,
}
joblib.dump(model_bundle, "/home/claude/tech_job_predictor.pkl")
print("\n  ✔ Saved tech_job_predictor.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13  ─  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"""
  Dataset (original)      : {df.shape[0]} rows × {df.shape[1]} cols
  After missing+dup removal: {len(df)} rows
  Tech roles only          : {len(df_tech)} rows
  Unique tech job titles   : {df_tech['Job Title'].nunique()}
  Train / Test split       : {X_train.shape[0]} / {X_test.shape[0]}

  ┌─────────────────────────┬────────────┬────────────┐
  │ Model                   │  Accuracy  │  F1 (wt.)  │
  ├─────────────────────────┼────────────┼────────────┤
  │ Random Forest           │  {acc_rf:.4f}   │  {f1_rf:.4f}   │
  │ Logistic Regression     │  {acc_lr:.4f}   │  {f1_lr:.4f}   │
  └─────────────────────────┴────────────┴────────────┘

  Best model : {best_model_name}
  Saved files:
    • tech_jobs_cleaned.csv
    • tech_job_predictor.pkl
    • fig1_histograms.png  … fig11_model_comparison.png
""")
print("  ✅ Pipeline complete.")
