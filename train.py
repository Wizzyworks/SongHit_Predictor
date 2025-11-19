import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer  # For genres multi-hot
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import pickle
import ast
from collections import Counter
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FitFailedWarning)

# Prep
n_splits = 5
output_file = 'xgb_songhit.bin'

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Target column
    df['is_banger'] = (df['track_popularity'] > 60).astype(int)
    df.drop('track_popularity', axis=1, inplace=True, errors='ignore')
    
    
    drops = ['track_id', 'track_name', 'album_id', 'album_name', 'artist_name']
    df.drop(columns=[col for col in drops if col in df.columns], inplace=True, errors='ignore')
    
    if 'album_release_date' in df.columns:
        df['album_release_date'] = pd.to_datetime(df['album_release_date'], errors='coerce')
        df['release_year'] = df['album_release_date'].dt.year
        df['release_month'] = df['album_release_date'].dt.month
        df['release_age'] = (pd.Timestamp.now() - df['album_release_date']).dt.days / 365.25
        df.drop('album_release_date', axis=1, inplace=True)
    
   
    if 'artist_genres' in df.columns:
        def safe_parse_genres(x):
            if pd.isna(x) or str(x) == 'nan' or str(x) == '':
                return ['unknown']
            try:
                return ast.literal_eval(str(x))
            except (ValueError, SyntaxError):
                try:
                    cleaned = str(x).replace("'", "").replace('"', '').replace('[', '').replace(']', '')
                    return [g.strip() for g in cleaned.split(',') if g.strip()]
                except:
                    return ['unknown']
        
        df['genres_list'] = df['artist_genres'].apply(safe_parse_genres)
        
       
        all_genres = [g for sublist in df['genres_list'] for g in sublist]
        top_genres = [g for g, _ in Counter(all_genres).most_common(10)]
        
        mlb = MultiLabelBinarizer(classes=top_genres)
        genre_matrix = mlb.fit_transform(df['genres_list'])
        genre_df = pd.DataFrame(genre_matrix, columns=[f'genre_{g}' for g in top_genres], index=df.index)
        df = pd.concat([df, genre_df], axis=1)
        df.drop(['artist_genres', 'genres_list'], axis=1, inplace=True)

    cat_cols = ['album_type', 'explicit']
    existing_cats = [col for col in cat_cols if col in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, prefix=existing_cats)
    
    # NUMS: LOG REPLACE 
    num_cols = ['artist_followers', 'artist_popularity', 'album_total_tracks', 'track_duration_min', 'release_age']
    for col in num_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            df.drop(col, axis=1, inplace=True)
    
    # Winsorize logs if outliers linger
    log_cols = [col for col in df.columns if col.startswith('log_')]
    for col in log_cols:
        q99 = df[col].quantile(0.99)
        df[col] = np.clip(df[col], None, q99)
    
    # NaNs last—0 for binaries/dummies, median for nums
    num_cols_final = df.select_dtypes(include=np.number).columns.drop('is_banger')
    df[num_cols_final] = df[num_cols_final].fillna(df[num_cols_final].median())
    df = df.fillna(0)  
    
    
    numerical = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'is_banger']
    categorical = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col != 'is_banger']  
    
    return df, numerical, categorical  

def train(df_train, y_train, numerical, categorical):
    dicts = df_train[numerical + categorical].to_dict(orient='records')  
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = XGBClassifier(
        eta=0.3, max_depth=6, min_child_weight=1, n_estimators=200,
        random_state=1, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model, numerical, categorical):
    dicts = df[numerical + categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

def run_validation(df_full_train, numerical, categorical):
    print("Doing validation using 5-fold KFold")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    fold = 0
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        y_train = df_train.is_banger.values
        y_val = df_val.is_banger.values
        dv, model = train(df_train, y_train, numerical, categorical)
        y_pred = predict(df_val, dv, model, numerical, categorical)
        y_bin = (y_pred >= 0.5).astype(int)
        f1 = f1_score(y_val, y_bin)
        scores.append(f1)
        print(f"F1 on fold {fold} is {f1}")
        fold += 1
    print("Validation results:")
    print("F1 mean = %.3f ± %.3f" % (np.mean(scores), np.std(scores)))

def main(data_path):
    df, numerical, categorical = load_and_clean_data(data_path)
    print(f"Engineered feats: {len(numerical) + len(categorical)} total (nums: {len(numerical)}, cats: {len(categorical)})")
    
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df['is_banger']
    )
    run_validation(df_full_train, numerical, categorical)
    print("Training the final model on full train set...")
    dv, model = train(df_full_train, df_full_train.is_banger.values, numerical, categorical)
    print("Evaluating on test set...")
    y_pred = predict(df_test, dv, model, numerical, categorical)
    y_bin = (y_pred >= 0.5).astype(int)
    f1 = f1_score(df_test.is_banger.values, y_bin)
    print(f"Final test F1 = {f1}")
    

    print("Final feats:", dv.get_feature_names_out())
    
    print("Saving the model...")
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"The model is saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGB Song Hit Predictor")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV')
    args = parser.parse_args()
    main(args.data)