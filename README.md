# SongHit Predictor — Will This Song Blow Up on Spotify?  

**Predicts if a track will become a "banger" (>60 popularity score) using only metadata & artist features**  
**Dockerized Flask API • XGBoost • 0.75 F1 • Fully reproducible**

Live local demo: `curl` → **99.2% hit probability** on a viral pop single  

![](screenshots/curl_demo.png)

### Project Highlights
- Trained XGBoost classifier on **~20k+ Spotify tracks**
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
- Source: [Spotify Million Song Dataset + additional features](https://www.kaggle.com/datasets) (public, ~20k rows, <100MB)
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
