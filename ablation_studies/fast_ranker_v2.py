"""
FAST_RANKER_V2: Minimal changes, maximum impact
- Switch to XGBoost rank:map (GPU) 
- Keep existing feature eng, just add semantic similarity
- Fast iteration
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split


SEED = 42
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
SKILLS_PATH = DATA_DIR / "Skills.csv"
OCCUPATIONS_PATH = DATA_DIR / "Occupations.csv"

SKILL_COLS = [f"skill_{i}" for i in range(1, 6)]
OCC_COLS = [f"occ_{i}" for i in range(1, 6)]


def apk(actual: list[str], predicted: list[str], k: int = 5) -> float:
    score = 0.0
    hits = 0.0
    for i, occ in enumerate(predicted[:k], start=1):
        if occ in actual and occ not in predicted[: i - 1]:
            hits += 1.0
            score += hits / i
    return score / min(len(actual), k)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    skills = pd.read_csv(SKILLS_PATH)
    occupations = pd.read_csv(OCCUPATIONS_PATH)

    for df in (train, test, skills, occupations):
        df["ID"] = df["ID"].astype(str)

    for col in SKILL_COLS:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
    for col in OCC_COLS:
        train[col] = train[col].astype(str)

    return train, test, skills.set_index("ID"), occupations.set_index("ID")


def build_ranking_data(
    fit: pd.DataFrame, val: pd.DataFrame, skills_df: pd.DataFrame, occs_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray, list[int]]:
    """Build features and labels for XGBoost ranking"""
    print("Building ranking data...", flush=True)
    
    occ_freq = Counter(fit[OCC_COLS].values.ravel())
    occ_list = sorted(set(occs_df.index.astype(str)))
    occ_to_idx = {oid: i for i, oid in enumerate(occ_list)}
    
    # Precompute TF-IDF for all occupations
    occ_texts = [" ".join([str(occs_df.loc[oid, c]) if oid in occs_df.index else "" 
                           for c in ["OCCUPATION_NAME", "OCCUPATION_DESCRIPTION", "OCCUPATION_GROUP_NAME"]])
                 for oid in occ_list]
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    occ_tfidf = vec.fit_transform(occ_texts)
    
    fit_docs = fit[SKILL_COLS].agg(" ".join, axis=1).values
    fit_tfidf = vec.transform(fit_docs)
    fit_occ_sim = linear_kernel(fit_tfidf, occ_tfidf)
    
    val_docs = val[SKILL_COLS].agg(" ".join, axis=1).values
    val_tfidf = vec.transform(val_docs)
    val_occ_sim = linear_kernel(val_tfidf, occ_tfidf)
    
    # Build fit data
    fit_features = []
    fit_labels = []
    fit_groups = []
    
    for row_idx, (_, row) in enumerate(fit.iterrows()):
        actual_occs = set(str(row[c]) for c in OCC_COLS)
        candidates = set(actual_occs)
        
        # Add top 50 by similarity
        sim_scores = fit_occ_sim[row_idx]
        top_idx = np.argsort(-sim_scores)[:80]
        for idx in top_idx:
            candidates.add(occ_list[idx])
        
        fit_groups.append(len(candidates))
        
        for occ in list(candidates)[:200]:
            if occ not in occ_to_idx:
                continue
            occ_idx = occ_to_idx[occ]
            
            feat = [
                float(occ in actual_occs),  # target
                float(fit_occ_sim[row_idx, occ_idx]),  # tfidf similarity
                math.log1p(occ_freq.get(occ, 1)),  # popularity
                float(len(actual_occs)),  # query size
            ]
            fit_features.append(feat)
            fit_labels.append(1.0 if occ in actual_occs else 0.0)
    
    # Build val data
    val_features = []
    val_labels = []
    val_groups = []
    
    for row_idx, (_, row) in enumerate(val.iterrows()):
        actual_occs = set(str(row[c]) for c in OCC_COLS)
        candidates = set(actual_occs)
        
        sim_scores = val_occ_sim[row_idx]
        top_idx = np.argsort(-sim_scores)[:80]
        for idx in top_idx:
            candidates.add(occ_list[idx])
        
        val_groups.append(len(candidates))
        
        for occ in list(candidates)[:200]:
            if occ not in occ_to_idx:
                continue
            occ_idx = occ_to_idx[occ]
            
            feat = [
                0.0,  # no label for val
                float(val_occ_sim[row_idx, occ_idx]),
                math.log1p(occ_freq.get(occ, 1)),
                float(len(actual_occs)),
            ]
            val_features.append(feat)
            val_labels.append(1.0 if occ in actual_occs else 0.0)
    
    return (
        np.array(fit_features),
        np.array(fit_labels),
        fit_groups,
        np.array(val_features),
        np.array(val_labels),
        val_groups,
    )


def train_and_evaluate(train: pd.DataFrame, skills_df: pd.DataFrame, occs_df: pd.DataFrame) -> float:
    """Train on 80% and evaluate on 20% holdout"""
    print("\n" + "="*60, flush=True)
    print("VALIDATION: 80/20 split", flush=True)
    print("="*60, flush=True)
    
    fit, val = train_test_split(train, test_size=0.2, random_state=SEED)
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)
    
    fit_X, fit_y, fit_groups, val_X, val_y, val_groups = build_ranking_data(fit, val, skills_df, occs_df)
    
    print("Training XGBoost with rank:map objective...", flush=True)
    ranker = xgb.XGBRanker(
        objective="rank:map",
        eval_metric="map@5",
        learning_rate=0.1,
        max_depth=8,
        n_estimators=150,
        random_state=SEED,
        tree_method="gpu_hist",
        gpu_id=0,
        verbosity=0,
    )
    
    ranker.fit(
        fit_X, fit_y,
        group=fit_groups,
        eval_set=[(val_X, val_y)],
        eval_group=[val_groups],
        verbose=False,
    )
    
    # Evaluate
    scores = ranker.predict(val_X)
    
    results = []
    score_idx = 0
    for row_idx, (_, row) in enumerate(val.iterrows()):
        actual = [str(row[c]) for c in OCC_COLS]
        group_size = val_groups[row_idx]
        group_scores = scores[score_idx:score_idx+group_size]
        
        # Get order
        order = np.argsort(-group_scores)
        results.append(apk(actual, [str(i) for i in order[:5]], k=5))
        score_idx += group_size
    
    mean_ap = float(np.mean(results))
    print(f"Validation MAP@5: {mean_ap:.6f}", flush=True)
    print("="*60 + "\n", flush=True)
    return mean_ap


def generate_submission(train: pd.DataFrame, test: pd.DataFrame, skills_df: pd.DataFrame, occs_df: pd.DataFrame, name: str) -> None:
    """Generate test submission"""
    print(f"Generating submission: {name}...", flush=True)
    
    fit_X, fit_y, fit_groups, test_X, test_y, test_groups = build_ranking_data(train, test, skills_df, occs_df)
    
    print("Training on full train set...", flush=True)
    ranker = xgb.XGBRanker(
        objective="rank:map",
        eval_metric="map@5",
        learning_rate=0.1,
        max_depth=8,
        n_estimators=200,
        random_state=SEED,
        tree_method="gpu_hist",
        gpu_id=0,
        verbosity=0,
    )
    
    ranker.fit(fit_X, fit_y, group=fit_groups, verbose=False)
    
    scores = ranker.predict(test_X)
    
    rows = []
    score_idx = 0
    for row_idx, (_, row) in enumerate(test.iterrows()):
        group_size = test_groups[row_idx]
        group_scores = scores[score_idx:score_idx+group_size]
        
        order = np.argsort(-group_scores)
        top5 = [str(i) for i in order[:5]]
        
        rows.append({
            "ID": str(row["ID"]),
            "occ_1": top5[0] if len(top5) > 0 else "0",
            "occ_2": top5[1] if len(top5) > 1 else "0",
            "occ_3": top5[2] if len(top5) > 2 else "0",
            "occ_4": top5[3] if len(top5) > 3 else "0",
            "occ_5": top5[4] if len(top5) > 4 else "0",
        })
        
        score_idx += group_size
    
    submission = pd.DataFrame(rows)
    out_path = DATA_DIR / f"submission_{name}.csv"
    submission.to_csv(out_path, index=False)
    print(f"✓ Saved to {out_path}", flush=True)


def main():
    train, test, skills_df, occs_df = load_data()
    
    # Quick validation
    val_score = train_and_evaluate(train, skills_df, occs_df)
    
    # Generate submission
    generate_submission(train, test, skills_df, occs_df, "fast_xgb_rankmap")


if __name__ == "__main__":
    main()
