# 🎵 Music Fit Predictor — "Will this song match someone's taste?"

## 1. Project Goal
Predict whether a song **fits a given listener's music taste** based on their
listening history, using Spotify audio features (tempo, energy, danceability,
valence, acousticness, etc.).

This matches the brief's example:
> "using music history listening, predict if a music can fit somebody music test"

## 2. Pipeline (matches the 4 required steps)

| Step | What we do | File |
|------|-----------|------|
| 1. Gather data | Download Kaggle dataset (Spotify Tracks Dataset, 114k tracks) | `data/dataset.csv` |
| 2. Analyse & define target | EDA notebook + create binary target `fits_user_taste` | `1_eda_and_target.ipynb` |
| 3. Build & train model | Preprocessing pipeline + Random Forest classifier, evaluate with accuracy/F1/ROC-AUC | `2_train_model.py` |
| 4. Interface (UX) | Streamlit app: pick "fake listening history" → recommend & predict fit for new tracks | `app.py` |

## 3. Dataset (Kaggle)
**Spotify Tracks Dataset** — 114,000 tracks, 125 genres, audio features + popularity (0-100)
🔗 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Download `dataset.csv`, place it in `data/dataset.csv`.

## 4. How the target is defined
We simulate a "user taste profile" from a sample of tracks the user "listened to
a lot" (e.g. their top 50 tracks). We compute the **centroid** of their audio
feature vector. For any candidate track:

- `fits_user_taste = 1` if the track's audio-feature distance to the user's
  taste centroid is below a threshold (top 30th percentile of similarity)
- `fits_user_taste = 0` otherwise

This is a **binary classification** problem.

## 5. Features used
`danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms, time_signature`

## 6. Model
- Preprocessing: `StandardScaler` on numeric features
- Model: `RandomForestClassifier` (scikit-learn)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Train/test split: 80/20, stratified

## 7. Interface (UX)
Streamlit app `app.py`:
1. User selects a set of songs they "love" (simulated listening history) from
   the dataset using search.
2. App computes their taste profile.
3. App lets user search any other track → predicts **Fit / Not Fit** with a
   confidence score, and shows the top recommended songs from the dataset.

## 8. How to run

```bash
pip install -r requirements.txt
python 2_train_model.py        # trains and saves model.pkl
streamlit run app.py
```

## 9. Presentation (10 min)
Suggested structure:
1. Problem & motivation (1 min)
2. Data: source, size, features (1.5 min)
3. Target definition & EDA highlights (2 min)
4. Model & pipeline + metrics (2 min)
5. Live demo of the Streamlit app (3 min)
6. Limitations & future work (0.5 min)
