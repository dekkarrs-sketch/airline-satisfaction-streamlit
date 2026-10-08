"""
Predicting Airline Passenger Satisfaction - A Machine-Learning Application Using Streamlit

Workflow:
  load data -> clean -> train/test split -> preprocessing -> three models
  (Logistic Regression, Decision Tree, Random Forest) -> evaluation
  -> overfitting checks -> feature importance -> interactive prediction.

Dataset: Airline Passenger Satisfaction (Kaggle, teejmahal20), train.csv + test.csv in ./data
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
DATA_DIR = Path(__file__).parent / "data"
TARGET = "satisfaction"
POSITIVE = "Satisfied"
NEGATIVE = "Neutral or dissatisfied"

CATEGORICAL_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]
NUMERIC_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
] + RATING_COLS
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load train.csv and test.csv, combine them, and clean the result."""
    frames = []
    for name in ("train.csv", "test.csv"):
        path = DATA_DIR / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Drop index / id columns
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("unnamed: 0", "id")])
    # Median-impute Arrival Delay (delays are skewed, so the median is more robust)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Arrival Delay in Minutes"].median()
    )
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = np.where(df[TARGET].str.strip().str.lower() == "satisfied", POSITIVE, NEGATIVE)
    return df


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )


@st.cache_resource(show_spinner="Training the three models (first run only)...")
def train_models():
    df = load_data()
    X = df[FEATURES]
    y = (df[TARGET] == POSITIVE).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
        ),
    }

    models, rows, preds, probas, train_acc = {}, [], {}, {}, {}
    for name, est in estimators.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", est)])
        pipe.fit(X_train, y_train)
        p = pipe.predict(X_test)
        pr = pipe.predict_proba(X_test)[:, 1]
        models[name], preds[name], probas[name] = pipe, p, pr
        train_acc[name] = pipe.score(X_train, y_train)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, p),
                "Precision": precision_score(y_test, p),
                "Recall": recall_score(y_test, p),
                "F1-score": f1_score(y_test, p),
                "ROC-AUC": roc_auc_score(y_test, pr),
            }
        )
    results = pd.DataFrame(rows).set_index("Model")

    gap = pd.DataFrame(
        {
            "Train accuracy": pd.Series(train_acc),
            "Test accuracy": results["Accuracy"],
        }
    )
    gap["Gap (train - test)"] = gap["Train accuracy"] - gap["Test accuracy"]

    # Decision-tree depth sensitivity (overfitting check)
    prep = make_preprocessor().fit(X_train)
    Xtr, Xte = prep.transform(X_train), prep.transform(X_test)
    depth_rows = []
    for d in [3, 5, 8, None]:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, y_train)
        depth_rows.append(
            {"max_depth": str(d), "Train accuracy": t.score(Xtr, y_train), "Test accuracy": t.score(Xte, y_test)}
        )
    depth_df = pd.DataFrame(depth_rows).set_index("max_depth")

    # Random-forest feature importance
    rf_pipe = models["Random Forest"]
    names = rf_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    importances = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=names).sort_values(
        ascending=False
    )

    reports = {
        n: classification_report(y_test, preds[n], target_names=[NEGATIVE, POSITIVE]) for n in models
    }
    cms = {n: confusion_matrix(y_test, preds[n]) for n in models}
    roc = {n: roc_curve(y_test, probas[n]) for n in models}
    return models, results, gap, depth_df, importances, reports, cms, roc, len(X_train), len(X_test)


# ---------------------------------------------------------------------------
# 3. Interface
# ---------------------------------------------------------------------------
df = load_data()
st.title("✈️ Predicting Airline Passenger Satisfaction")
st.caption("A machine-learning application using Streamlit")

if df.empty:
    st.error(
        "No data found. Place `train.csv` and `test.csv` (Kaggle: Airline Passenger Satisfaction) "
        "inside a folder named `data` next to `app.py`."
    )
    st.stop()

st.info(
    "**Note:** This is a student learning project built on a public Kaggle survey dataset. "
    "Predictions describe patterns in that dataset only and are not guarantees about any real airline or passenger."
)

models, results, gap, depth_df, importances, reports, cms, roc, n_train, n_test = train_models()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore data", "Prediction demo", "Model results"])

# ---- Tab 1: Overview -------------------------------------------------------
with tab1:
    st.header("Project objective")
    st.markdown(
        f"""
**Research question:** Which passenger, travel, delay, and in-flight service variables best
predict whether a passenger is **{POSITIVE.lower()}** with their flight?

The app combines `train.csv` and `test.csv` (about 130,000 survey records), cleans them, trains
three classifiers, compares their performance, checks for overfitting, and lets you test a
passenger profile.

**How it works**
1. Load and combine `train.csv` and `test.csv`; drop the `id` column.
2. Fill missing *Arrival Delay* values with the median; remove duplicate rows.
3. Split into 80% training and 20% testing data (random seed {SEED}, stratified).
4. Standardize numeric variables and one-hot encode categorical variables.
5. Train **Logistic Regression**, **Decision Tree**, and **Random Forest**.
6. Evaluate on unseen test data and run overfitting checks.
7. Rank the features that drive satisfaction.

**Responsible use:** survey labels reflect opinions at one time, for one airline context. The
model cannot explain *why* a person felt a certain way and should not be used to judge individuals.
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records (after cleaning)", f"{len(df):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Train / test rows", f"{n_train:,} / {n_test:,}")
    c4.metric("Satisfied share", f"{(df[TARGET] == POSITIVE).mean():.1%}")

# ---- Tab 2: Explore data ---------------------------------------------------
with tab2:
    st.header("Explore the data")
    sample = df.sample(min(20000, len(df)), random_state=SEED)

    left, right = st.columns(2)
    with left:
        st.subheader("Overall satisfaction")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x=TARGET, order=[NEGATIVE, POSITIVE], ax=ax)
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Satisfaction rate by category")
        cat = st.selectbox("Choose a category", CATEGORICAL_COLS, index=3)
        rate = (df[TARGET] == POSITIVE).groupby(df[cat]).mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_ylabel("Share satisfied")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Distribution by satisfaction")
    col = st.selectbox("Choose a numeric variable", NUMERIC_COLS, index=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.histplot(
        data=sample, x=col, hue=TARGET, bins=30, element="step", stat="density", common_norm=False, ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation with satisfaction")
    corr_df = df[NUMERIC_COLS].copy()
    corr_df["Satisfied (1/0)"] = (df[TARGET] == POSITIVE).astype(int)
    corr = corr_df.corr()["Satisfied (1/0)"].drop("Satisfied (1/0)").sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=corr.values, y=corr.index, ax=ax)
    ax.set_xlabel("Correlation with satisfied (1) vs not (0)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample records")
    st.dataframe(df.head(50), width="stretch")

# ---- Tab 3: Prediction demo ------------------------------------------------
with tab3:
    st.header("Prediction demo")
    st.write("Describe one passenger and flight, then choose a model.")

    model_name = st.selectbox("Model", list(models.keys()), index=2)

    a, b, c = st.columns(3)
    with a:
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        ctype = st.selectbox("Customer type", sorted(df["Customer Type"].unique()))
        age = st.slider("Age", 7, 85, 35)
    with b:
        ttype = st.selectbox("Type of travel", sorted(df["Type of Travel"].unique()))
        cls = st.selectbox("Class", ["Eco", "Eco Plus", "Business"], index=2)
        dist = st.slider("Flight distance (miles)", 30, 5000, 1000, 10)
    with c:
        dep = st.slider("Departure delay (min)", 0, 300, 0)
        arr = st.slider("Arrival delay (min)", 0, 300, 0)

    st.markdown("**Service ratings** (0 = not applicable, 1 = very poor, 5 = excellent)")
    ratings = {}
    cols = st.columns(4)
    for i, name in enumerate(RATING_COLS):
        ratings[name] = cols[i % 4].slider(name, 0, 5, 3, key=f"r_{i}")

    if st.button("Predict satisfaction"):
        row = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Customer Type": ctype,
                    "Type of Travel": ttype,
                    "Class": cls,
                    "Age": age,
                    "Flight Distance": dist,
                    "Departure Delay in Minutes": dep,
                    "Arrival Delay in Minutes": arr,
                    **ratings,
                }
            ]
        )[FEATURES]
        p = models[model_name].predict_proba(row)[0, 1]
        if p >= 0.5:
            st.success(f"Predicted: **{POSITIVE}** (probability {p:.1%})")
        else:
            st.error(f"Predicted: **{NEGATIVE}** (probability of being satisfied {p:.1%})")
        st.progress(float(p))
        st.caption("Educational demonstration based on a public survey dataset.")

# ---- Tab 4: Model results --------------------------------------------------
with tab4:
    st.header("Model results")
    best = results["F1-score"].idxmax()
    st.success(f"Best F1-score on the test set: **{best}** ({results.loc[best, 'F1-score']:.3f})")

    st.subheader("Model comparison")
    st.dataframe(results.style.format("{:.3f}"), width="stretch")
    fig, ax = plt.subplots(figsize=(9, 4))
    results.drop(columns="ROC-AUC").plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Neutral/Dissat.", "Satisfied"],
            yticklabels=["Neutral/Dissat.", "Satisfied"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, _) in roc.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {results.loc[name, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification report")
    report_model = st.selectbox("Model", list(reports.keys()), index=2, key="rep")
    st.code(reports[report_model])

    st.subheader("Overfitting checks")
    st.markdown("**Check 1 - Train vs test accuracy** (a small gap means the model generalizes)")
    st.dataframe(gap.style.format("{:.3f}"), width="stretch")
    st.markdown("**Check 2 - Decision-tree depth sensitivity**")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    depth_df.plot(marker="o", ax=ax)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "A much deeper tree fits the training data almost perfectly but gains little on the test set, "
        "which is a sign of overfitting. Cross-validation is demonstrated in the included notebook."
    )

    st.subheader("What drives satisfaction? (Random Forest importance)")
    top = importances.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Student learning project. Data: Airline Passenger Satisfaction (Kaggle). "
    "Educational demonstration only."
)
