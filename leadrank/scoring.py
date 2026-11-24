
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

NUMERIC = [
    "budget",
    "website_visits",
    "email_opens",
    "demo_requested"
]

CATEGORICAL = [
    "industry",
    "company_size",
    "source"
]


def _synthetic_training_data(n=1400, seed=42):
    rng = np.random.default_rng(seed)

    industries = np.array([
        "Technology", "Healthcare", "Finance",
        "Retail", "Education", "Manufacturing"
    ])

    sizes = np.array([
        "Startup", "Small", "Medium", "Enterprise"
    ])

    sources = np.array([
        "Website", "Referral", "LinkedIn",
        "Event", "Email"
    ])

    df = pd.DataFrame({
        "budget": rng.lognormal(9.5, 0.8, n).clip(500, 250000),
        "website_visits": rng.poisson(7, n).clip(0, 40),
        "email_opens": rng.poisson(3, n).clip(0, 20),
        "demo_requested": rng.binomial(1, 0.28, n),
        "industry": rng.choice(industries, n),
        "company_size": rng.choice(
            sizes, n, p=[0.28, 0.32, 0.25, 0.15]
        ),
        "source": rng.choice(sources, n)
    })

    logit = (
        -3.0
        + df["website_visits"] * 0.12
        + df["email_opens"] * 0.17
        + df["demo_requested"] * 1.5
        + np.log1p(df["budget"]) * 0.23
    )

    logit += df["company_size"].map({
        "Startup": -0.2,
        "Small": 0.0,
        "Medium": 0.25,
        "Enterprise": 0.5
    }).to_numpy()

    logit += df["source"].map({
        "Website": 0.0,
        "Referral": 0.45,
        "LinkedIn": 0.15,
        "Event": 0.25,
        "Email": -0.1
    }).to_numpy()

    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)

    return df, y


def get_model():
    X, y = _synthetic_training_data()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC)
    ])

    model = Pipeline([
        ("prep", pre),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "test_cases": int(len(y_test)),
        "data_note": "Synthetic demonstration dataset"
    }

    return model, metrics


def score_lead(model, lead):
    row = {
        key: lead[key]
        for key in NUMERIC + CATEGORICAL
    }

    probability = float(
        model.predict_proba(pd.DataFrame([row]))[0, 1]
    )

    score = round(probability * 100, 1)

    priority = (
        "High" if score >= 70
        else "Medium" if score >= 40
        else "Low"
    )

    if lead["demo_requested"]:
        recommendation = (
            "Arrange a product demonstration "
            "and confirm the buying timeline."
        )
    elif (
        lead["website_visits"] >= 8
        or lead["email_opens"] >= 4
    ):
        recommendation = (
            "Send a relevant case study and "
            "offer a short discovery call."
        )
    else:
        recommendation = (
            "Add to a helpful nurture sequence "
            "and review engagement later."
        )

    return {
        "probability": probability,
        "score": score,
        "priority": priority,
        "recommendation": recommendation
    }
