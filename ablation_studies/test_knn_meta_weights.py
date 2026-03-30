"""Quick test of meta_weight variations with 1-fold CV"""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold

SEED = 42
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
SKILLS_PATH = DATA_DIR / "Skills.csv"

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

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    skills = pd.read_csv(SKILLS_PATH)

    for df in (train, test, skills):
        df["ID"] = df["ID"].astype(str)

    for col in SKILL_COLS:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    for col in OCC_COLS:
        train[col] = train[col].astype(str)

    skills["ID"] = skills["ID"].astype(str)
    return train, test, skills.set_index("ID")

def build_meta_doc(skill_ids: list[str], skill_meta: pd.DataFrame) -> str:
    parts: list[str] = []
    for skill_id in skill_ids:
        parts.append(skill_id)
        if skill_id not in skill_meta.index:
            continue
        row = skill_meta.loc[skill_id]
        parts.extend(
            [
                str(row.get("NAME", "")),
                str(row.get("SUBCATEGORY_NAME", "")),
                str(row.get("CATEGORY_NAME", "")),
                str(row.get("TYPE", "")),
                "software" if bool(row.get("IS_SOFTWARE", False)) else "",
                "language" if bool(row.get("IS_LANGUAGE", False)) else "",
            ]
        )
    return " ".join(part for part in parts if part)

def build_similarity(
    fit: pd.DataFrame, val: pd.DataFrame, skill_meta: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    print("  Building ID similarity...", flush=True)
    fit_docs_id = fit[SKILL_COLS].agg(" ".join, axis=1)
    val_docs_id = val[SKILL_COLS].agg(" ".join, axis=1)
    vec_id = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    x_fit_id = vec_id.fit_transform(fit_docs_id)
    x_val_id = vec_id.transform(val_docs_id)
    sim_id = linear_kernel(x_val_id, x_fit_id)

    print("  Building metadata similarity...", flush=True)
    fit_docs_meta = fit[SKILL_COLS].apply(lambda row: build_meta_doc(row.tolist(), skill_meta), axis=1)
    val_docs_meta = val[SKILL_COLS].apply(lambda row: build_meta_doc(row.tolist(), skill_meta), axis=1)
    vec_meta = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=60000,
    )
    x_fit_meta = vec_meta.fit_transform(fit_docs_meta)
    x_val_meta = vec_meta.transform(val_docs_meta)
    sim_meta = linear_kernel(x_val_meta, x_fit_meta)
    return sim_id, sim_meta

def rank_from_similarity(
    fit: pd.DataFrame,
    sim_row: np.ndarray,
    occ_freq: Counter[str],
    global_rank: list[str],
    topn_rows: int,
    damp: float,
) -> tuple[list[str], dict[str, float], list[dict[str, float]]]:
    if sim_row.shape[0] == 0:
        return global_rank[:5], {}, [defaultdict(float) for _ in range(5)]

    take = min(topn_rows, sim_row.shape[0])
    if take == sim_row.shape[0]:
        nn = np.argsort(-sim_row)
    else:
        nn = np.argpartition(-sim_row, take - 1)[:take]
        nn = nn[np.argsort(-sim_row[nn])]

    scores: dict[str, float] = defaultdict(float)
    slot_scores: list[dict[str, float]] = [defaultdict(float) for _ in range(5)]
    for row_idx in nn:
        weight = float(sim_row[row_idx])
        if weight <= 0:
            continue
        for slot, occ in enumerate(fit.loc[row_idx, OCC_COLS].tolist()):
            denom = occ_freq[occ] ** damp if damp else 1.0
            scores[occ] += weight / denom
            slot_scores[slot][occ] += weight

    ranked = [occ for occ, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    ranked.extend([occ for occ in global_rank if occ not in scores])
    return ranked, scores, slot_scores

def blend_predictions(
    pred_id: list[str],
    pred_meta: list[str],
    scores_id: dict[str, float],
    scores_meta: dict[str, float],
    global_rank: list[str],
    meta_weight: float = 0.45,
    pop_weight: float = 0.002,
) -> list[str]:
    candidates: list[str] = []
    for occ in pred_id + pred_meta + global_rank[:20]:
        if occ not in candidates:
            candidates.append(occ)

    pop_bonus = {occ: 1.0 / (idx + 1) for idx, occ in enumerate(global_rank[:200])}
    blend_scores = {
        occ: scores_id.get(occ, 0.0) + meta_weight * scores_meta.get(occ, 0.0) + pop_weight * pop_bonus.get(occ, 0.0)
        for occ in candidates
    }
    return [occ for occ, _ in sorted(blend_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]]

def quick_test_cv(train: pd.DataFrame, skill_meta: pd.DataFrame) -> None:
    """Test meta_weight configs with just 1 split (50% train, 50% val)"""
    print("\nTesting meta_weight configurations (1-split quick test)...\n", flush=True)
    
    configs = [
        ('meta_w_30', 0.30),
        ('meta_w_40', 0.40),
        ('meta_w_50', 0.50),
        ('meta_w_60', 0.60),
    ]
    
    # Split at midpoint for simplicity
    split_idx = len(train) // 2
    fit = train.iloc[:split_idx].reset_index(drop=True)
    val = train.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Fit size: {len(fit)}, Val size: {len(val)}", flush=True)
    
    sim_id, sim_meta = build_similarity(fit, val, skill_meta)
    occ_freq: Counter[str] = Counter(fit[OCC_COLS].values.ravel())
    global_rank = [occ for occ, _ in occ_freq.most_common()]
    
    results = {}
    for config_name, meta_weight in configs:
        print(f"\nTesting {config_name} (meta_weight={meta_weight})...", flush=True)
        scores_list = []
        
        for row_idx, (_, row) in enumerate(val.iterrows()):
            if row_idx % 100 == 0:
                print(f"  Row {row_idx}/{len(val)}", flush=True)
            
            actual = row[OCC_COLS].tolist()
            ranked_id, scores_id, _ = rank_from_similarity(
                fit=fit,
                sim_row=sim_id[row_idx],
                occ_freq=occ_freq,
                global_rank=global_rank,
                topn_rows=40,
                damp=0.15,
            )
            ranked_meta, scores_meta, _ = rank_from_similarity(
                fit=fit,
                sim_row=sim_meta[row_idx],
                occ_freq=occ_freq,
                global_rank=global_rank,
                topn_rows=40,
                damp=0.10,
            )
            pred_id = ranked_id[:5]
            pred_meta = ranked_meta[:5]
            pred = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=meta_weight)
            scores_list.append(apk(actual, pred))
        
        avg_score = float(np.mean(scores_list))
        results[config_name] = avg_score
        print(f"  {config_name}: {avg_score:.6f}", flush=True)
    
    print("\n" + "="*50, flush=True)
    print("Summary of Results:", flush=True)
    print("="*50, flush=True)
    for config_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config_name:15s}: {score:.6f}", flush=True)
    print("="*50, flush=True)

if __name__ == "__main__":
    print("Loading data...", flush=True)
    train, test, skill_meta = load_data()
    quick_test_cv(train, skill_meta)
    print("\nDone!", flush=True)
