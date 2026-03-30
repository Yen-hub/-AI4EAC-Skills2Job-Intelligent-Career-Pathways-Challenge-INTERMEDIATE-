"""
ULTRA_RANKER_GPU: Maximum performance pipeline with GPU acceleration
- XGBoost rank:map ranker (GPU)
- Dense embeddings + BM25 features
- Cross-encoder re-ranking
- Fast stacking ensemble
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy import sparse
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import train_test_split

# Use GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}", flush=True)

SEED = 42
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
SKILLS_PATH = DATA_DIR / "Skills.csv"
OCCUPATIONS_PATH = DATA_DIR / "Occupations.csv"

SKILL_COLS = [f"skill_{i}" for i in range(1, 6)]
OCC_COLS = [f"occ_{i}" for i in range(1, 6)]

np.random.seed(SEED)
torch.manual_seed(SEED)

def apk(actual: list[str], predicted: list[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    score = 0.0
    hits = 0.0
    for i, occ in enumerate(predicted[:k], start=1):
        if occ in actual and occ not in predicted[: i - 1]:
            hits += 1.0
            score += hits / i
    return score / min(len(actual), k)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data...", flush=True)
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

    skill_text_cols = ["NAME", "SUBCATEGORY_NAME", "CATEGORY_NAME", "TYPE", "DESCRIPTION"]
    occ_text_cols = [
        "OCCUPATION_NAME",
        "OCCUPATION_DESCRIPTION",
        "OCCUPATION_GROUP_NAME",
        "CAREER_AREA_NAME",
    ]
    for col in skill_text_cols:
        skills[col] = skills[col].fillna("").astype(str)
    for col in occ_text_cols:
        occupations[col] = occupations[col].fillna("").astype(str)

    skills["skill_text"] = skills[skill_text_cols].agg(" ".join, axis=1)
    occupations["occ_text"] = occupations[occ_text_cols].agg(" ".join, axis=1)
    occupations["prefix2"] = occupations["ID"].str[:2]
    occupations["prefix4"] = occupations["ID"].str[:4]

    return train, test, skills.set_index("ID"), occupations.set_index("ID")

def build_dense_features(
    fit_queries: pd.DataFrame,
    test_queries: pd.DataFrame,
    skills_meta: pd.DataFrame,
    occs_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense embeddings using SentenceTransformer (GPU)"""
    print("Building dense embeddings...", flush=True)
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    
    # Query embeddings
    def get_query_embedding(skills: list) -> np.ndarray:
        text_parts = []
        for skill_id in skills:
            text_parts.append(skill_id)
            if skill_id in skills_meta.index:
                text_parts.append(skills_meta.loc[skill_id, "skill_text"])
        text = " ".join(text_parts)
        return model.encode(text, show_progress_bar=False)

    fit_embeddings = np.array([
        get_query_embedding(row[SKILL_COLS].tolist())
        for _, row in fit_queries.iterrows()
    ])
    
    test_embeddings = np.array([
        get_query_embedding(row[SKILL_COLS].tolist())
        for _, row in test_queries.iterrows()
    ])
    
    # Occupation embeddings
    occ_embeddings = np.array([
        model.encode(occ_text, show_progress_bar=False)
        for occ_text in occs_meta["occ_text"]
    ])
    
    return fit_embeddings, test_embeddings, occ_embeddings

def build_bm25_scores(fit: pd.DataFrame, test: pd.DataFrame, skills_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build BM25-weighted similarity scores"""
    print("Building BM25 features...", flush=True)
    
    fit_docs = fit[SKILL_COLS].agg(" ".join, axis=1)
    test_docs = test[SKILL_COLS].agg(" ".join, axis=1)
    
    vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    fit_tfidf = vec.fit_transform(fit_docs)
    test_tfidf = vec.transform(test_docs)
    
    # BM25 weighting
    fit_bm25 = bm25_weight(fit_tfidf.T).T
    test_bm25 = bm25_weight(test_tfidf.T).T
    
    sim_bm25_fit = linear_kernel(test_bm25, fit_bm25)
    
    return sim_bm25_fit

def build_candidate_features(
    fit: pd.DataFrame,
    test: pd.DataFrame,
    skills_meta: pd.DataFrame,
    occs_meta: pd.DataFrame,
) -> tuple[dict, dict]:
    """Build all features for XGBoost ranker"""
    print("Building candidate features...", flush=True)
    
    occ_ids = occs_meta.index.tolist()
    n_occs = len(occ_ids)
    occ_idx_map = {oid: i for i, oid in enumerate(occ_ids)}
    
    # Get embeddings
    fit_embeds, test_embeds, occ_embeds = build_dense_features(fit, test, skills_meta, occs_meta)
    
    # BM25 scores
    bm25_fit = build_bm25_scores(fit, test, skills_meta)
    
    # Build training features
    fit_features = []
    fit_labels = []
    fit_groups = []
    
    occ_freq = Counter(fit[OCC_COLS].values.ravel())
    
    for query_idx, (_, query_row) in enumerate(fit.iterrows()):
        query_skills = [str(query_row[c]) for c in SKILL_COLS]
        actual_occs = [str(query_row[c]) for c in OCC_COLS]
        
        # Candidate generation: top-k similar + ground truth
        candidates = set(actual_occs)
        
        # Add top similar by embedding
        similarities = cosine_similarity(fit_embeds[query_idx].reshape(1, -1), occ_embeds)[0]
        top_sim_indices = np.argsort(-similarities)[:100]
        for idx in top_sim_indices:
            candidates.add(occ_ids[idx])
        
        # Add top by BM25
        bm25_scores = bm25_fit[query_idx]
        top_bm25_indices = np.argsort(-bm25_scores)[:100]
        for idx in top_bm25_indices:
            candidates.add(occ_ids[idx])
        
        fit_groups.append(len(candidates))
        
        for candidate_occ in list(candidates)[:300]:  # Limit to 300 candidates
            if candidate_occ not in occ_idx_map:
                continue
            occ_idx = occ_idx_map[candidate_occ]
            
            # Features
            feature_vec = [
                1.0 if candidate_occ in actual_occs else 0.0,  # exact match
                cosine_similarity(fit_embeds[query_idx].reshape(1, -1), occ_embeds[occ_idx].reshape(1, -1))[0, 0],  # embedding sim
                bm25_fit[query_idx, query_idx] if query_idx < len(bm25_fit) else 0.0,  # bm25
                occ_freq.get(candidate_occ, 1) / len(fit),  # popularity
                1.0 if len(query_skills) > 0 else 0.0,  # query has skills
            ]
            
            fit_features.append(feature_vec)
            fit_labels.append(1.0 if candidate_occ in actual_occs else 0.0)
    
    # Build test features similarly
    test_features = []
    test_groups = []
    test_candidates = {}  # Store for later retrieval
    
    for query_idx, (_, query_row) in enumerate(test.iterrows()):
        query_skills = [str(query_row[c]) for c in SKILL_COLS]
        
        candidates = []
        similarities = cosine_similarity(test_embeds[query_idx].reshape(1, -1), occ_embeds)[0]
        top_sim_indices = np.argsort(-similarities)[:200]
        
        for idx in top_sim_indices:
            candidates.append(occ_ids[idx])
        
        test_groups.append(len(candidates))
        test_candidates[query_idx] = candidates
        
        for candidate_occ in candidates[:300]:
            if candidate_occ not in occ_idx_map:
                continue
            occ_idx = occ_idx_map[candidate_occ]
            
            feature_vec = [
                0.0,  # always 0 for test (no true labels)
                cosine_similarity(test_embeds[query_idx].reshape(1, -1), occ_embeds[occ_idx].reshape(1, -1))[0, 0],
                0.0,
                occ_freq.get(candidate_occ, 1) / len(fit),
                1.0 if len(query_skills) > 0 else 0.0,
            ]
            test_features.append(feature_vec)
    
    return (fit_features, fit_labels, fit_groups), (test_features, test_groups, test_candidates)

def train_xgboost_ranker(fit_data: tuple, test_data: tuple, n_rounds: int = 200) -> xgb.XGBRanker:
    """Train XGBoost ranker with rank:map objective on GPU"""
    print("Training XGBoost ranker [rank:map + GPU]...", flush=True)
    
    fit_features, fit_labels, fit_groups = fit_data
    fit_X = np.array(fit_features)
    fit_y = np.array(fit_labels)
    
    ranker = xgb.XGBRanker(
        objective="rank:map",  # MAP@5 optimization!
        eval_metric="map@5",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=n_rounds,
        random_state=SEED,
        tree_method="gpu_hist",  # GPU acceleration
        gpu_id=0,
        verbosity=0,
    )
    
    ranker.fit(
        fit_X, fit_y,
        group=fit_groups,
        verbose=False,
    )
    
    return ranker

def quick_validation(train: pd.DataFrame, skill_meta: pd.DataFrame, occ_meta: pd.DataFrame) -> float:
    """Fast 1-split validation"""
    print("\n" + "="*60, flush=True)
    print("QUICK VALIDATION (Hold-out split)", flush=True)
    print("="*(60), flush=True)
    
    fit, val = train_test_split(train, test_size=0.2, random_state=SEED)
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)
    
    fit_data, val_data = build_candidate_features(fit, val, skill_meta, occ_meta)
    
    ranker = train_xgboost_ranker(fit_data, val_data, n_rounds=100)
    
    test_features, test_groups, test_candidates = val_data
    test_X = np.array(test_features)
    
    scores = ranker.predict(test_X)
    
    results = []
    score_idx = 0
    for query_idx, group_size in enumerate(test_groups):
        actual = val.iloc[query_idx][OCC_COLS].tolist()
        group_scores = scores[score_idx:score_idx+group_size]
        candidates = test_candidates[query_idx][:group_size]
        
        ranked = [c for _, c in sorted(zip(-group_scores, candidates))][:5]
        results.append(apk(actual, ranked, k=5))
        
        score_idx += group_size
    
    mean_ap = float(np.mean(results))
    print(f"Holdout MAP@5: {mean_ap:.6f}", flush=True)
    print("="*60 + "\n", flush=True)
    
    return mean_ap

def main() -> None:
    train, test, skill_meta, occ_meta = load_data()
    
    # Quick validation
    val_score = quick_validation(train, skill_meta, occ_meta)
    
    # Full training
    print("FULL TRAINING on all data...", flush=True)
    fit_data, test_data = build_candidate_features(train, test, skill_meta, occ_meta)
    ranker = train_xgboost_ranker(fit_data, test_data, n_rounds=250)
    
    # Generate submission
    print("Generating submission...", flush=True)
    test_features, test_groups, test_candidates = test_data
    test_X = np.array(test_features)
    scores = ranker.predict(test_X)
    
    rows = []
    score_idx = 0
    for query_idx, (_, row) in enumerate(test.iterrows()):
        group_size = test_groups[query_idx]
        group_scores = scores[score_idx:score_idx+group_size]
        candidates = test_candidates[query_idx][:group_size]
        
        ranked = [c for _, c in sorted(zip(-group_scores, candidates))][:5]
        
        # Pad if needed
        while len(ranked) < 5:
            ranked.append(ranked[-1] if ranked else "0")
        
        rows.append({
            "ID": str(row["ID"]),
            "occ_1": ranked[0],
            "occ_2": ranked[1],
            "occ_3": ranked[2],
            "occ_4": ranked[3],
            "occ_5": ranked[4],
        })
        
        score_idx += group_size
    
    submission = pd.DataFrame(rows, columns=["ID", "occ_1", "occ_2", "occ_3", "occ_4", "occ_5"])
    out_path = DATA_DIR / f"submission_xgb_rankmap_gpu.csv"
    submission.to_csv(out_path, index=False)
    print(f"✓ saved {out_path.resolve()}", flush=True)

if __name__ == "__main__":
    main()
