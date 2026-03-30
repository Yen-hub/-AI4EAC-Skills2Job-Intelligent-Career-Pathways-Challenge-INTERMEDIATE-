from __future__ import annotations

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
    fit_docs_id = fit[SKILL_COLS].agg(" ".join, axis=1)
    val_docs_id = val[SKILL_COLS].agg(" ".join, axis=1)
    vec_id = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    x_fit_id = vec_id.fit_transform(fit_docs_id)
    x_val_id = vec_id.transform(val_docs_id)
    sim_id = linear_kernel(x_val_id, x_fit_id)

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


def reorder_blend_predictions(
    pred_id: list[str],
    pred_meta: list[str],
    scores_id: dict[str, float],
    scores_meta: dict[str, float],
    slot_scores_id: list[dict[str, float]],
    slot_scores_meta: list[dict[str, float]],
    global_rank: list[str],
    candidate_pool_size: int = 10,
) -> list[str]:
    candidates: list[str] = []
    for occ in pred_id + pred_meta + global_rank[:20]:
        if occ not in candidates:
            candidates.append(occ)

    ranked_candidates = sorted(
        candidates,
        key=lambda occ: scores_id.get(occ, 0.0) + 0.45 * scores_meta.get(occ, 0.0),
        reverse=True,
    )
    pool = ranked_candidates[:candidate_pool_size]

    ordered: list[str] = []
    for slot in range(5):
        best_occ = None
        best_score = float("-inf")
        for occ in pool:
            if occ in ordered:
                continue
            score = (
                scores_id.get(occ, 0.0)
                + 0.45 * scores_meta.get(occ, 0.0)
                + 0.35 * slot_scores_id[slot].get(occ, 0.0)
                + 0.20 * slot_scores_meta[slot].get(occ, 0.0)
            )
            if score > best_score:
                best_score = score
                best_occ = occ

        if best_occ is None:
            for occ in ranked_candidates:
                if occ not in ordered:
                    best_occ = occ
                    break

        ordered.append(best_occ or global_rank[len(ordered)])

    return ordered


def cross_validate(train: pd.DataFrame, skill_meta: pd.DataFrame, n_splits: int = 3) -> dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics: dict[str, list[float]] = {
        "id_knn": [],
        "meta_knn": [],
        "blend_knn": [],
        "blend_w025": [],
        "blend_w030": [],
        "blend_meta_w_30": [],
        "blend_meta_w_40": [],
        "blend_meta_w_50": [],
        "blend_meta_w_60": [],
        "blend_reordered": [],
    }

    for fold, (fit_idx, val_idx) in enumerate(kf.split(train), start=1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        val = train.iloc[val_idx].reset_index(drop=True)
        sim_id, sim_meta = build_similarity(fit, val, skill_meta)

        occ_freq: Counter[str] = Counter(fit[OCC_COLS].values.ravel())
        global_rank = [occ for occ, _ in occ_freq.most_common()]

        for row_idx, (_, row) in enumerate(val.iterrows()):
            actual = row[OCC_COLS].tolist()
            ranked_id, scores_id, slot_scores_id = rank_from_similarity(
                fit=fit,
                sim_row=sim_id[row_idx],
                occ_freq=occ_freq,
                global_rank=global_rank,
                topn_rows=40,
                damp=0.15,
            )
            ranked_meta, scores_meta, slot_scores_meta = rank_from_similarity(
                fit=fit,
                sim_row=sim_meta[row_idx],
                occ_freq=occ_freq,
                global_rank=global_rank,
                topn_rows=40,
                damp=0.10,
            )
            pred_id = ranked_id[:5]
            pred_meta = ranked_meta[:5]
            pred_blend = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank)
            pred_blend_w025 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.25
            )
            pred_blend_w030 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.30
            )
            pred_blend_meta_w_30 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.30
            )
            pred_blend_meta_w_40 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.40
            )
            pred_blend_meta_w_50 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.50
            )
            pred_blend_meta_w_60 = blend_predictions(
                pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.60
            )
            pred_reordered = reorder_blend_predictions(
                pred_id=pred_id,
                pred_meta=pred_meta,
                scores_id=scores_id,
                scores_meta=scores_meta,
                slot_scores_id=slot_scores_id,
                slot_scores_meta=slot_scores_meta,
                global_rank=global_rank,
            )

            metrics["id_knn"].append(apk(actual, pred_id))
            metrics["meta_knn"].append(apk(actual, pred_meta))
            metrics["blend_knn"].append(apk(actual, pred_blend))
            metrics["blend_w025"].append(apk(actual, pred_blend_w025))
            metrics["blend_w030"].append(apk(actual, pred_blend_w030))
            metrics["blend_meta_w_30"].append(apk(actual, pred_blend_meta_w_30))
            metrics["blend_meta_w_40"].append(apk(actual, pred_blend_meta_w_40))
            metrics["blend_meta_w_50"].append(apk(actual, pred_blend_meta_w_50))
            metrics["blend_meta_w_60"].append(apk(actual, pred_blend_meta_w_60))
            metrics["blend_reordered"].append(apk(actual, pred_reordered))

        print(f"finished fold {fold}/{n_splits}", flush=True)

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def build_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    skill_meta: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    sim_id, sim_meta = build_similarity(train, test, skill_meta)
    occ_freq: Counter[str] = Counter(train[OCC_COLS].values.ravel())
    global_rank = [occ for occ, _ in occ_freq.most_common()]

    rows: list[dict[str, str]] = []
    for row_idx, (_, row) in enumerate(test.iterrows()):
        ranked_id, scores_id, slot_scores_id = rank_from_similarity(
            fit=train,
            sim_row=sim_id[row_idx],
            occ_freq=occ_freq,
            global_rank=global_rank,
            topn_rows=40,
            damp=0.15,
        )
        ranked_meta, scores_meta, slot_scores_meta = rank_from_similarity(
            fit=train,
            sim_row=sim_meta[row_idx],
            occ_freq=occ_freq,
            global_rank=global_rank,
            topn_rows=40,
            damp=0.10,
        )
        pred_id = ranked_id[:5]
        pred_meta = ranked_meta[:5]

        if variant == "id_knn":
            top5 = pred_id
        elif variant == "meta_knn":
            top5 = pred_meta
        elif variant == "blend_knn":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank)
        elif variant == "blend_w025":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.25)
        elif variant == "blend_w030":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.30)
        elif variant == "blend_meta_w_30":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.30)
        elif variant == "blend_meta_w_40":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.40)
        elif variant == "blend_meta_w_50":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.50)
        elif variant == "blend_meta_w_60":
            top5 = blend_predictions(pred_id, pred_meta, scores_id, scores_meta, global_rank, meta_weight=0.60)
        elif variant == "blend_reordered":
            top5 = reorder_blend_predictions(
                pred_id,
                pred_meta,
                scores_id,
                scores_meta,
                slot_scores_id,
                slot_scores_meta,
                global_rank,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        rows.append(
            {
                "ID": str(row["ID"]),
                "occ_1": top5[0],
                "occ_2": top5[1],
                "occ_3": top5[2],
                "occ_4": top5[3],
                "occ_5": top5[4],
            }
        )

    return pd.DataFrame(rows, columns=["ID", "occ_1", "occ_2", "occ_3", "occ_4", "occ_5"])


def main() -> None:
    train, test, skill_meta = load_data()
    cv_scores = cross_validate(train, skill_meta)

    print("\n3-fold CV MAP@5", flush=True)
    for name, score in cv_scores.items():
        print(f"{name}: {score:.6f}", flush=True)

    for variant in ("id_knn", "meta_knn", "blend_knn", "blend_w025", "blend_w030", "blend_meta_w_30", "blend_meta_w_40", "blend_meta_w_50", "blend_meta_w_60", "blend_reordered"):
        submission = build_submission(train, test, skill_meta, variant)
        out_path = DATA_DIR / f"submission_{variant}.csv"
        submission.to_csv(out_path, index=False)
        print(f"saved {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
