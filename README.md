# SongHit Predictor — Will This Song Blow Up on Spotify?  

**Predicts if a track will become a "banger" (>60 popularity score) using only metadata & artist features**  
**Dockerized Flask API • XGBoost • 0.75 F1 • Fully reproducible**

Live local demo: `curl` → **99.2% hit probability** on a viral pop single  

![](screenshots/curl_demo.png)

### Project Highlights
- Trained XGBoost classifier on **~9k+ Spotify tracks**
- Achieves **0.75 F1** on imbalanced test set (baseline ~0.58)
- Only **23 interpretable engineered features** (logs + one-hot top genres + release metadata)
- Clean pipeline: `train.py` → `xgb_songhit.bin` + `DictVectorizer`
- Production-ready Flask API (`predict.py`) served via Gunicorn
- Fully Dockerized (one-command deployment)
- Detailed EDA revealing why **explicit + pop + fresh + big artist = viral**

---

### Problem Statement
Given a song's metadata (artist followers, genre, explicit flag, release date, etc.), predict whether it will achieve **high popularity (>60/100)** on Spotify.

**Who benefits?**  
A&R teams, indie artists, playlist curators, and labels trying to spot the next big hit before it blows up.

**Business impact**: Reduce promotion spend on flops, prioritize high-potential tracks.

**Evaluation metric**: **F1-score** (best for imbalanced classes — only ~28% of songs are true bangers)

---

### Dataset
- Source: [Spotify Million Song Dataset + additional features]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/alyahmedts13/spotify-songs-for-ml-and-analysis-over-8700-tracks)) (public, ~9k rows, <100MB)
- Target: `track_popularity > 60` → binary `is_banger`
- Features used: artist followers/popularity, genres, explicit, album type, release date, duration, etc.
- Missing values handled gracefully (median + mode imputation)

---

### Key EDA Insights (Top 5)
| Insight                                    | Impact |
|-------------------------------------------|--------|
| Explicit songs are **2.1× more likely** to be bangers | Huge signal |
| Pop, Hip-Hop, Rap dominate top genres     | Expected but confirmed |
| Songs released in **2023–2024** have 3× higher hit rate | Freshness matters |
| Artists with **>10M followers** dominate bangers | Network effect |
| Singles > album tracks for virality       | Release strategy matters |

![](screenshots/eda_explicit.png)
![](screenshots/eda_genres.png)

---

### Modeling
| Model               | CV F1 (mean ± std) | Test F1 |
|---------------------|--------------------|---------|
| Logistic Regression (baseline) | 0.68 ± 0.012 | 0.69 |
| Random Forest       | 0.73 ± 0.010       | 0.74    |
| **XGBoost (final)** | **0.742 ± 0.009**  | **0.750** |

→ XGBoost selected for best performance + speed

---

### Project Structure
.
├── train.py              # Trains model + saves xgb_songhit.bin

├── predict.py            # Loads model + Flask API

├── xgb_songhit.bin       # Trained model + DictVectorizer

├── spotify_data.csv      # Raw dataset

├── requirements.txt      # All dependencies

├── Dockerfile            # Build once, run anywhere

├── notebooks/
│   └── SongHit_EDA_finetuning.ipynb

└── screenshots/          # Proof it works


---

### How to Run Locally

```bash
# 1. Train the model
pipenv install
pipenv run python train.py --data spotify_data.csv

# 2. Start the API
pipenv run python predict.py

# 3. Test it
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{
"log_artist_followers": 15.0,
  "log_artist_popularity": 4.38,
  "log_album_total_tracks": 2.48,
  "log_track_duration_min": 3.0,
  "log_release_age": 0.5,
  "release_month": 6,
  "release_year": 2024,
  "track_number": 1,
  "album_type_single": 1,
  "explicit_True": 1,
  "genre_pop": 1,
  "genre_hip hop": 0,
  "genre_rap": 0
}"
```
### For Docker Execution

```bash
# 1. Run docker desktop in background first, then in your command prompt go to the repo destination.

# 2. type these two commands
docker build -t songhit-api .
docker run -p 9696:9696 songhit-api

# 3. Test it
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "log_artist_followers":16.0,"log_artist_popularity":5.0,"log_album_total_tracks":2.48,
    "log_track_duration_min":3.1,"log_release_age":0.1,"release_month":8,"release_year":2024,
    "track_number":1,"album_type_album":0,"album_type_compilation":0,"album_type_single":1,
    "explicit_False":0,"explicit_True":1,"genre_alternative pop":0,"genre_country":0,
    "genre_folk":0,"genre_hip hop":0,"genre_indie":0,"genre_pop":1,"genre_rap":0,
    "genre_rock":0,"genre_soundtrack":0,"genre_unknown":0
  }'
```
### Response
{"hit_prob":0.9922537207603455,"is_banger":true}

### Limitations & Next Steps
> No audio features (MFCCs, tempo) → future versions can use Librosa + CNN
> Model may overfit to recent trends (2023–2024 bias)
> Could add real-time Spotify API lookup for live predictions


