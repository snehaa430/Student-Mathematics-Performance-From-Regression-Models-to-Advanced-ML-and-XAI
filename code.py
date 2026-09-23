"""
Interpretable and Explainable Machine Learning for Predicting
Student Mathematics Performance: From Regression Models to Advanced ML and XAI
--------------------------------------------------------------------------------

Dataset: UCI "Student Performance" dataset (Math subject), Cortez & Silva (2008).
Download page:  https://archive.ics.uci.edu/dataset/320/student+performance
File needed:    student-mat.csv  (place it next to this script)

If the file isn't found, a synthetic dataset with a similar structure is
generated automatically so the pipeline still runs end-to-end.

Install:
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost shap
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import statsmodels.api as sm

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT_DIR = "outputs_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# PHASE 1 — DATA LOADING, PREPROCESSING, EDA
# =============================================================================

DATA_PATH = "student-mat.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Loaded real dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
else:
    print("student-mat.csv not found — generating a synthetic placeholder dataset.")
    n = 395
    df = pd.DataFrame({
        "studytime": np.random.randint(1, 5, n),
        "failures": np.random.randint(0, 4, n),
        "absences": np.random.poisson(5, n),
        "Medu": np.random.randint(0, 5, n),
        "Fedu": np.random.randint(0, 5, n),
        "goout": np.random.randint(1, 6, n),
        "freetime": np.random.randint(1, 6, n),
        "internet": np.random.choice(["yes", "no"], n, p=[0.8, 0.2]),
        "schoolsup": np.random.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "famsup": np.random.choice(["yes", "no"], n, p=[0.6, 0.4]),
        "G1": np.random.randint(0, 21, n),
        "G2": np.random.randint(0, 21, n),
    })
    df["G3"] = (
        0.5 * df["G1"] + 0.5 * df["G2"]
        - 1.2 * df["failures"]
        - 0.05 * df["absences"] ** 1.5
        + 0.3 * df["studytime"]
        - 0.15 * (df["studytime"] * df["failures"])
        + np.random.normal(0, 1.5, n)
    ).clip(0, 20)

# Encode binary yes/no columns
binary_cols = [c for c in ["internet", "schoolsup", "famsup", "paid",
                            "activities", "higher", "romantic"] if c in df.columns]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

core_features = [c for c in ["studytime", "failures", "absences", "Medu", "Fedu",
                              "goout", "freetime"] if c in df.columns]
extra_features = binary_cols
FEATURES_WITH_GRADES = core_features + extra_features + [c for c in ["G1", "G2"] if c in df.columns]
FEATURES_NO_GRADES = core_features + extra_features
TARGET = "G3"

print("Features WITH prior grades:", FEATURES_WITH_GRADES)
print("Features WITHOUT prior grades:", FEATURES_NO_GRADES)

# EDA: correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES_WITH_GRADES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_correlation_matrix.png", dpi=150)
plt.close()

# EDA: target distribution
plt.figure(figsize=(6, 4))
sns.histplot(df[TARGET], kde=True, bins=15)
plt.title("Distribution of Final Grade (G3)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_target_distribution.png", dpi=150)
plt.close()


def make_split(feature_list):
    X = df[feature_list].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def evaluate(model, X_train, y_train, X_test, y_test, y_pred, n_params, label, results_list):
    r2_train = r2_score(y_train, model.predict(X_train)) if hasattr(model, "predict") else model.score(X_train, y_train)
    r2_test = r2_score(y_test, y_pred)
    n, p = X_train.shape[0], n_params - 1
    adj_r2_train = 1 - (1 - r2_train) * (n - 1) / max(n - p - 1, 1)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv)
        cv_rmse = -cv_scores.mean()
    except Exception:
        cv_rmse = np.nan

    print(f"{label:45s} | R2_test={r2_test:.3f} | RMSE_test={rmse_test:.3f} | CV_RMSE={cv_rmse:.3f}")
    results_list.append({
        "model": label, "train_r2": r2_train, "adj_train_r2": adj_r2_train,
        "test_r2": r2_test, "test_rmse": rmse_test, "test_mae": mae_test, "cv_rmse": cv_rmse,
    })


# =============================================================================
# PHASE 2 — STATISTICAL REGRESSION MODELS
# =============================================================================

def run_statistical_models(feature_list, scenario_label, results_list):
    X_train, X_test, y_train, y_test = make_split(feature_list)
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # --- Linear Regression ---
    lin = LinearRegression().fit(X_train, y_train)
    evaluate(lin, X_train, y_train, X_test, y_test, lin.predict(X_test),
              len(feature_list) + 1, f"Linear ({scenario_label})", results_list)

    # --- Ridge Regression ---
    ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
    evaluate(ridge, X_train_s, y_train, X_test_s, y_test, ridge.predict(X_test_s),
              len(feature_list) + 1, f"Ridge ({scenario_label})", results_list)

    # --- Lasso Regression ---
    lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)
    evaluate(lasso, X_train_s, y_train, X_test_s, y_test, lasso.predict(X_test_s),
              len(feature_list) + 1, f"Lasso ({scenario_label})", results_list)
    zeroed = [f for f, c in zip(feature_list, lasso.coef_) if abs(c) < 1e-6]
    print(f"   Lasso zeroed-out features: {zeroed}")

    # --- Polynomial Regression (degree 2 on studytime, absences) ---
    poly_terms = [f for f in ["studytime", "absences"] if f in feature_list]

    def add_poly(X):
        X = X.copy()
        for f in poly_terms:
            X[f"{f}^2"] = X[f] ** 2
        return X

    Xtr_poly, Xte_poly = add_poly(X_train), add_poly(X_test)
    poly_model = LinearRegression().fit(Xtr_poly, y_train)
    evaluate(poly_model, Xtr_poly, y_train, Xte_poly, y_test, poly_model.predict(Xte_poly),
              Xtr_poly.shape[1] + 1, f"Polynomial ({scenario_label})", results_list)

    # --- Interaction Regression ---
    pairs = [(a, b) for a, b in [("studytime", "failures"), ("Medu", "Fedu")]
             if a in feature_list and b in feature_list]

    def add_interactions(X):
        X = X.copy()
        for a, b in pairs:
            X[f"{a}_x_{b}"] = X[a] * X[b]
        return X

    Xtr_int, Xte_int = add_interactions(X_train), add_interactions(X_test)
    int_model = LinearRegression().fit(Xtr_int, y_train)
    evaluate(int_model, Xtr_int, y_train, Xte_int, y_test, int_model.predict(Xte_int),
              Xtr_int.shape[1] + 1, f"Interaction ({scenario_label})", results_list)

    # statsmodels OLS summary for the interaction model -> coefficients + p-values
    ols = sm.OLS(y_train, sm.add_constant(Xtr_int)).fit()
    print(f"\n--- OLS Summary: Interaction model ({scenario_label}) ---")
    print(ols.summary())
    print()

    return ols  # returned for later comparison with SHAP


# =============================================================================
# PHASE 3 — ADVANCED MACHINE LEARNING MODELS
# =============================================================================

def run_ml_models(feature_list, scenario_label, results_list):
    X_train, X_test, y_train, y_test = make_split(feature_list)
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
        "SVR (RBF)": SVR(kernel="rbf", C=5, epsilon=0.5),
    }

    fitted = {}
    for name, model in models.items():
        if name == "SVR (RBF)":
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            evaluate(model, X_train_s, y_train, X_test_s, y_test, pred,
                      len(feature_list) + 1, f"{name} ({scenario_label})", results_list)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            evaluate(model, X_train, y_train, X_test, y_test, pred,
                      len(feature_list) + 1, f"{name} ({scenario_label})", results_list)
        fitted[name] = model

    return fitted, X_train, X_test, y_train, y_test


# =============================================================================
# PHASE 4 — EXPLAINABLE AI (XAI)
# =============================================================================

def run_xai(fitted_models, X_train, X_test, y_test, scenario_label):
    # Use the best tree-based model available for SHAP
    best_name = "Random Forest"
    model = fitted_models[best_name]

    # --- Built-in feature importance ---
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"{best_name} Feature Importance ({scenario_label})")
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/03_feature_importance_{scenario_label.replace(' ', '_')}.png",
        dpi=150
    )
    plt.close()

    # --- Permutation importance ---
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE
    )

    perm_series = pd.Series(
        perm.importances_mean,
        index=X_test.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=perm_series.values, y=perm_series.index)
    plt.title(f"Permutation Importance ({scenario_label})")
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/04_permutation_importance_{scenario_label.replace(' ', '_')}.png",
        dpi=150
    )
    plt.close()

    # --- SHAP global + local explanations ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary ({scenario_label})")
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/05_shap_summary_{scenario_label.replace(' ', '_')}.png",
        dpi=150
    )
    plt.close()

    # --- Local explanation ---
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Local Explanation — Student 0 ({scenario_label})")
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/06_shap_local_student0_{scenario_label.replace(' ', '_')}.png",
        dpi=150
    )
    plt.close()

    # --- SHAP dependence plot ---
    if "absences" in X_test.columns:
        plt.figure()
        shap.dependence_plot(
            "absences",
            shap_values.values,
            X_test,
            show=False
        )
        plt.title(f"SHAP Dependence: absences ({scenario_label})")
        plt.tight_layout()
        plt.savefig(
            f"{OUT_DIR}/07_shap_dependence_absences_{scenario_label.replace(' ', '_')}.png",
            dpi=150
        )
        plt.close()

    # --- Partial Dependence / ICE plots ---
    pdp_features = [
        f for f in ["studytime", "absences", "failures"]
        if f in X_train.columns
    ]

    if pdp_features:
        # Convert integer columns to floating point for scikit-learn PDP
        X_train_pdp = X_train.astype(float)

        fig, ax = plt.subplots(figsize=(12, 4))

        PartialDependenceDisplay.from_estimator(
            model,
            X_train_pdp,
            pdp_features,
            kind="both",
            ax=ax
        )

        plt.suptitle(f"PDP / ICE Plots ({scenario_label})")
        plt.tight_layout()
        plt.savefig(
            f"{OUT_DIR}/08_pdp_ice_{scenario_label.replace(' ', '_')}.png",
            dpi=150
        )
        plt.close()

    # --- SHAP interaction values ---
    pair = ("studytime", "failures")

    if pair[0] in X_test.columns and pair[1] in X_test.columns:
        try:
            shap_interact = explainer.shap_interaction_values(X_test)

            i = list(X_test.columns).index(pair[0])
            j = list(X_test.columns).index(pair[1])

            interaction_strength = np.abs(
                shap_interact[:, i, j]
            ).mean()

            print(
                f"   Mean |SHAP interaction| for "
                f"{pair[0]} x {pair[1]} ({scenario_label}): "
                f"{interaction_strength:.4f}"
            )

        except Exception as e:
            print(
                f"   SHAP interaction values unavailable: {e}"
            )

    return importances, perm_series



# =============================================================================
# RUN FULL PIPELINE FOR BOTH SCENARIOS
# =============================================================================

all_results = []
ols_summaries = {}
xai_outputs = {}

for scenario_label, feature_set in [("Scenario A - with G1G2", FEATURES_WITH_GRADES),
                                     ("Scenario B - without G1G2", FEATURES_NO_GRADES)]:
    print(f"\n{'='*90}\n{scenario_label}\n{'='*90}")
    print("\n--- Phase 2: Statistical Models ---")
    ols_summaries[scenario_label] = run_statistical_models(feature_set, scenario_label, all_results)

    print("\n--- Phase 3: Advanced ML Models ---")
    fitted, X_train, X_test, y_train, y_test = run_ml_models(feature_set, scenario_label, all_results)

    print("\n--- Phase 4: XAI ---")
    xai_outputs[scenario_label] = run_xai(fitted, X_train, X_test, y_test, scenario_label)

# =============================================================================
# FINAL COMPARISON TABLE
# =============================================================================

results_df = pd.DataFrame(all_results)
print("\n\n==================== FULL MODEL COMPARISON ====================")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison_full.csv", index=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="model", y="test_rmse")
plt.xticks(rotation=75, ha="right")
plt.ylabel("Test RMSE (lower is better)")
plt.title("Full Model Comparison: Statistical vs. ML — Test RMSE")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_full_model_comparison_rmse.png", dpi=150)
plt.close()

print(f"\nAll figures and tables saved to ./{OUT_DIR}/")
print("Done.")