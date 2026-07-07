"""
2_train_model.py
-----------------
Step 3 of the project: build the model, training & prediction pipeline.

Usage:
    python 2_train_model.py

Expects data/dataset.csv from:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature"
]

RANDOM_STATE = 42


def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + ["track_name", "artists"])
    df = df.drop_duplicates(subset=["track_name", "artists"])
    return df.reset_index(drop=True)


def build_user_taste_profile(df, n_favorites=50, seed=RANDOM_STATE):
    """
    Simulate a user's listening history: randomly sample n_favorites tracks
    weighted toward higher-popularity tracks (more 'listened to' songs).
    Returns the centroid (mean feature vector) of those tracks.
    """
    rng = np.random.default_rng(seed)
    weights = (df["popularity"] + 1) ** 2
    weights = weights / weights.sum()
    idx = rng.choice(df.index, size=n_favorites, replace=False, p=weights.values)
    favorites = df.loc[idx]
    centroid = favorites[FEATURES].mean()
    return centroid, favorites


def create_target(df, centroid, scaler, percentile=70):
    """
    fits_user_taste = 1 if track is among the `percentile`-closest tracks
    to the user's taste centroid (by Euclidean distance on scaled features).
    """
    X_scaled = scaler.transform(df[FEATURES])
    centroid_scaled = scaler.transform(pd.DataFrame([centroid], columns=FEATURES))
    distances = np.linalg.norm(X_scaled - centroid_scaled, axis=1)
    threshold = np.percentile(distances, 100 - percentile)  # smaller distance = closer
    # invert: we want the closest `percentile`% to be class 1
    cutoff = np.percentile(distances, percentile)
    df = df.copy()
    df["distance_to_taste"] = distances
    df["fits_user_taste"] = (distances <= cutoff).astype(int)
    return df


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} tracks")

    # Fit scaler on full feature set first (for distance calc)
    pre_scaler = StandardScaler().fit(df[FEATURES])

    print("Building simulated user taste profile (top 50 favorite tracks)...")
    centroid, favorites = build_user_taste_profile(df, n_favorites=50)

    print("Creating target variable 'fits_user_taste'...")
    df = create_target(df, centroid, pre_scaler, percentile=30)
    print(df["fits_user_taste"].value_counts(normalize=True))

    X = df[FEATURES]
    y = df["fits_user_taste"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:\n", feat_imp)

    # Save artifacts
    joblib.dump(pipeline, "model.pkl")
    joblib.dump(centroid, "user_taste_centroid.pkl")
    df.to_csv("data/dataset_with_target.csv", index=False)
    print("\nSaved model.pkl, user_taste_centroid.pkl, data/dataset_with_target.csv")


if __name__ == "__main__":
    main()
