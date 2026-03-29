from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


SEED = 42
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
SKILLS_PATH = DATA_DIR / "Skills.csv"
OCCUPATIONS_PATH = DATA_DIR / "Occupations.csv"
SKILL_COLS = [f"skill_{i}" for i in range(1, 6)]
OCC_COLS = [f"occ_{i}" for i in range(1, 6)]


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
        "OCCUPATION_GROUP_DESCRIPTION",
        "CAREER_AREA_NAME",
        "CAREER_AREA_DESCRIPTION",
        "REQUIREMENT_LEVEL_DESCRIPTION",
        "SPECIALIZED_TRAINING_DESCRIPTION",
    ]
    for col in skill_text_cols:
        skills[col] = skills[col].fillna("").astype(str)
    for col in occ_text_cols:
        occupations[col] = occupations[col].fillna("").astype(str)

    bool_cols = [
        "LICENSE_TYPICALLY_REQUIRED",
        "CERTIFICATION_TYPICALLY_REQUIRED",
        "REQUIRES_SPECIALIZED_TRAINING",
    ]
    for col in bool_cols:
        occupations[col] = occupations[col].fillna(False).astype(str)

    occupations["prefix2"] = occupations["ID"].str[:2]
    occupations["prefix4"] = occupations["ID"].str[:4]
    occupations["occ_text"] = occupations[occ_text_cols].agg(" ".join, axis=1)
    occupations["min_train_months"] = occupations["MINIMUM_TRAINING_LENGTH_MONTHS"].fillna(0).astype(float)
    skills["skill_text"] = skills[skill_text_cols].agg(" ".join, axis=1)
    return train, test, skills.set_index("ID"), occupations.set_index("ID")


def build_query_frame(df: pd.DataFrame, skill_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        skill_ids = [str(getattr(row, c)) for c in SKILL_COLS]
        meta_parts: list[str] = []
        cats: list[str] = []
        subcats: list[str] = []
        types: list[str] = []
        software_count = 0
        skill_cat_slots: list[str] = []
        skill_type_slots: list[str] = []

        for skill_id in skill_ids:
            if skill_id in skill_meta.index:
                srow = skill_meta.loc[skill_id]
                cat = str(srow.get("CATEGORY_NAME", "unknown")) or "unknown"
                subcat = str(srow.get("SUBCATEGORY_NAME", "unknown")) or "unknown"
                skill_type = str(srow.get("TYPE", "unknown")) or "unknown"
                if str(srow.get("IS_SOFTWARE", "False")).lower() == "true":
                    software_count += 1
                meta_parts.extend(
                    [
                        skill_id,
                        str(srow.get("NAME", "")),
                        subcat,
                        cat,
                        skill_type,
                        str(srow.get("DESCRIPTION", "")),
                    ]
                )
                cats.append(cat)
                subcats.append(subcat)
                types.append(skill_type)
                skill_cat_slots.append(cat)
                skill_type_slots.append(skill_type)
            else:
                meta_parts.append(skill_id)
                skill_cat_slots.append("unknown")
                skill_type_slots.append("unknown")

        rows.append(
            {
                "id_doc": " ".join(skill_ids),
                "meta_doc": " ".join(x for x in meta_parts if x),
                "query_cat_signature": "|".join(sorted(set(cats))) or "unknown",
                "query_type_signature": "|".join(sorted(set(types))) or "unknown",
                "query_subcat_signature": "|".join(sorted(set(subcats))) or "unknown",
                "query_unique_skill_categories": len(set(cats)),
                "query_unique_skill_subcategories": len(set(subcats)),
                "query_unique_skill_types": len(set(types)),
                "query_software_count": software_count,
                "skill_cat_1": skill_cat_slots[0],
                "skill_cat_2": skill_cat_slots[1],
                "skill_cat_3": skill_cat_slots[2],
                "skill_cat_4": skill_cat_slots[3],
                "skill_cat_5": skill_cat_slots[4],
                "skill_type_1": skill_type_slots[0],
                "skill_type_2": skill_type_slots[1],
                "skill_type_3": skill_type_slots[2],
                "skill_type_4": skill_type_slots[3],
                "skill_type_5": skill_type_slots[4],
            }
        )
    return pd.DataFrame(rows, index=df.index)


def build_cooccurrence_artifacts(
    fit: pd.DataFrame, skill_meta: pd.DataFrame
) -> dict[str, object]:
    skill_occ: dict[str, Counter[str]] = defaultdict(Counter)
    pair_occ: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    occ_counter: Counter[str] = Counter(fit[OCC_COLS].values.ravel())
    prefix2_counter: Counter[str] = Counter()
    prefix4_counter: Counter[str] = Counter()
    prefix2_to_occ: dict[str, Counter[str]] = defaultdict(Counter)
    prefix4_to_occ: dict[str, Counter[str]] = defaultdict(Counter)
    cat_to_prefix2: dict[str, Counter[str]] = defaultdict(Counter)
    cat_to_prefix4: dict[str, Counter[str]] = defaultdict(Counter)
    type_to_prefix2: dict[str, Counter[str]] = defaultdict(Counter)
    type_to_prefix4: dict[str, Counter[str]] = defaultdict(Counter)

    for row in fit.itertuples(index=False):
        skill_ids = [str(getattr(row, c)) for c in SKILL_COLS]
        qskills = sorted(set(skill_ids))
        labels = [str(getattr(row, c)) for c in OCC_COLS]
        qcats = set()
        qtypes = set()
        for skill_id in qskills:
            if skill_id in skill_meta.index:
                srow = skill_meta.loc[skill_id]
                qcats.add(str(srow.get("CATEGORY_NAME", "unknown")) or "unknown")
                qtypes.add(str(srow.get("TYPE", "unknown")) or "unknown")

        for occ in labels:
            p2 = occ[:2]
            p4 = occ[:4]
            prefix2_counter[p2] += 1
            prefix4_counter[p4] += 1
            prefix2_to_occ[p2][occ] += 1
            prefix4_to_occ[p4][occ] += 1
            for cat in qcats:
                cat_to_prefix2[cat][p2] += 1
                cat_to_prefix4[cat][p4] += 1
            for skill_type in qtypes:
                type_to_prefix2[skill_type][p2] += 1
                type_to_prefix4[skill_type][p4] += 1
            for skill_id in qskills:
                skill_occ[skill_id][occ] += 1
            for i, s1 in enumerate(qskills):
                for s2 in qskills[i + 1 :]:
                    pair_occ[(s1, s2)][occ] += 1

    total_skill_edges = max(sum(sum(v.values()) for v in skill_occ.values()), 1)
    skill_df = {skill: len(v) for skill, v in skill_occ.items()}
    skill_weight = {
        skill: math.log((1 + total_skill_edges) / (1 + df)) + 1.0
        for skill, df in skill_df.items()
    }
    return {
        "skill_occ": skill_occ,
        "pair_occ": pair_occ,
        "occ_counter": occ_counter,
        "prefix2_counter": prefix2_counter,
        "prefix4_counter": prefix4_counter,
        "prefix2_to_occ": prefix2_to_occ,
        "prefix4_to_occ": prefix4_to_occ,
        "cat_to_prefix2": cat_to_prefix2,
        "cat_to_prefix4": cat_to_prefix4,
        "type_to_prefix2": type_to_prefix2,
        "type_to_prefix4": type_to_prefix4,
        "skill_weight": skill_weight,
        "global_rank": [occ for occ, _ in occ_counter.most_common()],
    }


def fit_prefix_models(x_fit, fit_df: pd.DataFrame):
    y_prefix2 = fit_df[OCC_COLS].apply(lambda row: sorted({str(x)[:2] for x in row}), axis=1)
    y_prefix4 = fit_df[OCC_COLS].apply(lambda row: sorted({str(x)[:4] for x in row}), axis=1)

    mlb2 = MultiLabelBinarizer(sparse_output=True)
    mlb4 = MultiLabelBinarizer(sparse_output=True)
    y2 = mlb2.fit_transform(y_prefix2)
    y4 = mlb4.fit_transform(y_prefix4)

    base_model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        random_state=SEED,
    )
    model2 = OneVsRestClassifier(base_model, n_jobs=1)
    model4 = OneVsRestClassifier(base_model, n_jobs=1)
    model2.fit(x_fit, y2)
    model4.fit(x_fit, y4)
    return model2, mlb2, model4, mlb4


def rank_dict(scores: dict[str, float]) -> dict[str, int]:
    return {key: idx for idx, key in enumerate(sorted(scores, key=scores.get, reverse=True), start=1)}


def aggregate_neighbor_scores(
    fit: pd.DataFrame,
    sim_row: np.ndarray,
    occ_freq: Counter[str],
    topn_rows: int,
    damp: float,
) -> tuple[dict[str, float], dict[str, int]]:
    if sim_row.shape[0] == 0:
        return {}, {}
    take = min(topn_rows, sim_row.shape[0])
    idx = np.argpartition(-sim_row, take - 1)[:take]
    idx = idx[np.argsort(-sim_row[idx])]
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row_idx in idx:
        weight = float(sim_row[row_idx])
        if weight <= 0:
            continue
        for occ in fit.loc[row_idx, OCC_COLS].tolist():
            denom = occ_freq[occ] ** damp if damp else 1.0
            scores[occ] += weight / denom
            counts[occ] += 1
    return scores, counts


def compute_cooccurrence_scores(
    qskills: list[str], artifacts: dict[str, object]
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    qskills = sorted(set(qskills))
    scores: dict[str, float] = defaultdict(float)
    skill_hits: dict[str, int] = defaultdict(int)
    pair_hits: dict[str, int] = defaultdict(int)
    for i, s1 in enumerate(qskills):
        for s2 in qskills[i + 1 :]:
            for occ, cnt in artifacts["pair_occ"].get((s1, s2), {}).items():
                scores[occ] += 1.35 * math.log1p(cnt)
                pair_hits[occ] += 1
    for skill_id in qskills:
        occs = artifacts["skill_occ"].get(skill_id, {})
        if not occs:
            continue
        total = sum(occs.values())
        for occ, cnt in occs.items():
            scores[occ] += artifacts["skill_weight"].get(skill_id, 1.0) * math.log1p(cnt)
            scores[occ] += 0.25 * ((cnt + 1.0) / (total + 1.0))
            skill_hits[occ] += 1
    for occ, cnt in artifacts["occ_counter"].items():
        scores[occ] += 0.01 * math.log1p(cnt)
    return scores, skill_hits, pair_hits


def empirical_support(
    query_info: pd.Series,
    prefix: str,
    counters: dict[str, Counter[str]],
    cols: list[str],
) -> float:
    total = 0.0
    for col in cols:
        key = str(query_info[col])
        if key in counters:
            counter = counters[key]
            denom = max(sum(counter.values()), 1)
            total += counter.get(prefix, 0) / denom
    return total


def build_candidate_frame(
    fit: pd.DataFrame,
    pred_df: pd.DataFrame,
    query_pred: pd.DataFrame,
    occ_meta: pd.DataFrame,
    artifacts: dict[str, object],
    sim_id: np.ndarray,
    sim_meta: np.ndarray,
    occ_text_sim: np.ndarray,
    prefix2_scores: np.ndarray,
    prefix2_classes: np.ndarray,
    prefix4_scores: np.ndarray,
    prefix4_classes: np.ndarray,
    label_available: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    occ_freq = artifacts["occ_counter"]
    prefix2_freq = artifacts["prefix2_counter"]
    prefix4_freq = artifacts["prefix4_counter"]
    occ_ids = occ_meta.index.tolist()

    rows: list[dict[str, object]] = []
    group_sizes: list[int] = []
    for qid, (_, row) in enumerate(pred_df.iterrows()):
        qskills = row[SKILL_COLS].tolist()
        query_info = query_pred.loc[row.name]
        actual = set(row[OCC_COLS].tolist()) if label_available else set()

        id_scores, id_counts = aggregate_neighbor_scores(fit, sim_id[qid], occ_freq, topn_rows=45, damp=0.15)
        meta_scores, meta_counts = aggregate_neighbor_scores(fit, sim_meta[qid], occ_freq, topn_rows=70, damp=0.10)
        co_scores, co_skill_hits, co_pair_hits = compute_cooccurrence_scores(qskills, artifacts)

        text_take = min(30, occ_text_sim.shape[1])
        text_idx = np.argpartition(-occ_text_sim[qid], text_take - 1)[:text_take]
        text_idx = text_idx[np.argsort(-occ_text_sim[qid][text_idx])]
        text_scores = {occ_ids[i]: float(occ_text_sim[qid][i]) for i in text_idx}

        p2_scores = {cls: float(score) for cls, score in zip(prefix2_classes, prefix2_scores[qid])}
        p4_scores = {cls: float(score) for cls, score in zip(prefix4_classes, prefix4_scores[qid])}

        combined_scores: dict[str, float] = defaultdict(float)
        for occ, score in id_scores.items():
            combined_scores[occ] += score
        for occ, score in meta_scores.items():
            combined_scores[occ] += 0.30 * score
        for occ, score in co_scores.items():
            combined_scores[occ] += 0.12 * score
        for occ, score in text_scores.items():
            combined_scores[occ] += 0.20 * score

        candidates: list[str] = []
        sources = [
            sorted(id_scores, key=id_scores.get, reverse=True)[:30],
            sorted(meta_scores, key=meta_scores.get, reverse=True)[:20],
            sorted(co_scores, key=co_scores.get, reverse=True)[:20],
            sorted(text_scores, key=text_scores.get, reverse=True)[:20],
        ]
        for source in sources:
            for occ in source:
                if occ not in candidates:
                    candidates.append(occ)

        top_prefix4 = sorted(p4_scores, key=p4_scores.get, reverse=True)[:5]
        top_prefix2 = sorted(p2_scores, key=p2_scores.get, reverse=True)[:3]

        for prefix4 in top_prefix4:
            if prefix4 not in artifacts["prefix4_to_occ"]:
                continue
            ranked = sorted(
                artifacts["prefix4_to_occ"][prefix4],
                key=lambda occ: combined_scores.get(occ, 0.0)
                + 0.3 * text_scores.get(occ, 0.0)
                + 0.15 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:6]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        for prefix2 in top_prefix2:
            if prefix2 not in artifacts["prefix2_to_occ"]:
                continue
            ranked = sorted(
                artifacts["prefix2_to_occ"][prefix2],
                key=lambda occ: combined_scores.get(occ, 0.0)
                + 0.2 * text_scores.get(occ, 0.0)
                + 0.2 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:8]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        rare_candidates = sorted(
            combined_scores,
            key=lambda occ: (combined_scores.get(occ, 0.0) + 0.2 * text_scores.get(occ, 0.0))
            / math.sqrt(max(occ_freq.get(occ, 0), 1)),
            reverse=True,
        )[:10]
        for occ in rare_candidates:
            if occ not in candidates:
                candidates.append(occ)

        for occ in artifacts["global_rank"][:15]:
            if occ not in candidates:
                candidates.append(occ)

        id_rank = rank_dict(id_scores)
        meta_rank = rank_dict(meta_scores)
        co_rank = rank_dict(co_scores)
        text_rank = rank_dict(text_scores)
        combined_rank = rank_dict(combined_scores)

        for occ in candidates[:120]:
            occ_row = occ_meta.loc[occ] if occ in occ_meta.index else None
            prefix2 = occ[:2]
            prefix4 = occ[:4]
            rows.append(
                {
                    "qid": qid,
                    "label": int(occ in actual) if label_available else 0,
                    "candidate_occ": occ,
                    "prefix2": prefix2,
                    "prefix4": prefix4,
                    "career_area": str(occ_row["CAREER_AREA_NAME"]) if occ_row is not None else "unknown",
                    "occupation_group": str(occ_row["OCCUPATION_GROUP_NAME"]) if occ_row is not None else "unknown",
                    "requirement_level": str(occ_row["REQUIREMENT_LEVEL"]) if occ_row is not None else "unknown",
                    "license_required": str(occ_row["LICENSE_TYPICALLY_REQUIRED"]) if occ_row is not None else "unknown",
                    "cert_required": str(occ_row["CERTIFICATION_TYPICALLY_REQUIRED"]) if occ_row is not None else "unknown",
                    "requires_training": str(occ_row["REQUIRES_SPECIALIZED_TRAINING"]) if occ_row is not None else "unknown",
                    "query_cat_signature": str(query_info["query_cat_signature"]),
                    "query_type_signature": str(query_info["query_type_signature"]),
                    "query_subcat_signature": str(query_info["query_subcat_signature"]),
                    "skill_1": row["skill_1"],
                    "skill_2": row["skill_2"],
                    "skill_3": row["skill_3"],
                    "skill_4": row["skill_4"],
                    "skill_5": row["skill_5"],
                    "skill_cat_1": query_info["skill_cat_1"],
                    "skill_cat_2": query_info["skill_cat_2"],
                    "skill_cat_3": query_info["skill_cat_3"],
                    "skill_cat_4": query_info["skill_cat_4"],
                    "skill_cat_5": query_info["skill_cat_5"],
                    "skill_type_1": query_info["skill_type_1"],
                    "skill_type_2": query_info["skill_type_2"],
                    "skill_type_3": query_info["skill_type_3"],
                    "skill_type_4": query_info["skill_type_4"],
                    "skill_type_5": query_info["skill_type_5"],
                    "id_score": float(id_scores.get(occ, 0.0)),
                    "meta_score": float(meta_scores.get(occ, 0.0)),
                    "co_score": float(co_scores.get(occ, 0.0)),
                    "text_score": float(text_scores.get(occ, 0.0)),
                    "id_rank": float(id_rank.get(occ, 999.0)),
                    "meta_rank": float(meta_rank.get(occ, 999.0)),
                    "co_rank": float(co_rank.get(occ, 999.0)),
                    "text_rank": float(text_rank.get(occ, 999.0)),
                    "combined_rank": float(combined_rank.get(occ, 999.0)),
                    "id_support_count": float(id_counts.get(occ, 0)),
                    "meta_support_count": float(meta_counts.get(occ, 0)),
                    "co_skill_hits": float(co_skill_hits.get(occ, 0)),
                    "co_pair_hits": float(co_pair_hits.get(occ, 0)),
                    "prefix2_model_score": float(p2_scores.get(prefix2, -20.0)),
                    "prefix4_model_score": float(p4_scores.get(prefix4, -20.0)),
                    "cat_prefix2_support": float(
                        empirical_support(query_info, prefix2, artifacts["cat_to_prefix2"], [f"skill_cat_{i}" for i in range(1, 6)])
                    ),
                    "cat_prefix4_support": float(
                        empirical_support(query_info, prefix4, artifacts["cat_to_prefix4"], [f"skill_cat_{i}" for i in range(1, 6)])
                    ),
                    "type_prefix2_support": float(
                        empirical_support(query_info, prefix2, artifacts["type_to_prefix2"], [f"skill_type_{i}" for i in range(1, 6)])
                    ),
                    "type_prefix4_support": float(
                        empirical_support(query_info, prefix4, artifacts["type_to_prefix4"], [f"skill_type_{i}" for i in range(1, 6)])
                    ),
                    "query_unique_skill_categories": float(query_info["query_unique_skill_categories"]),
                    "query_unique_skill_subcategories": float(query_info["query_unique_skill_subcategories"]),
                    "query_unique_skill_types": float(query_info["query_unique_skill_types"]),
                    "query_software_count": float(query_info["query_software_count"]),
                    "occ_freq": float(occ_freq.get(occ, 1)),
                    "occ_log_freq": math.log1p(occ_freq.get(occ, 1)),
                    "prefix2_freq": float(prefix2_freq.get(prefix2, 1)),
                    "prefix4_freq": float(prefix4_freq.get(prefix4, 1)),
                    "occ_share_in_prefix4": occ_freq.get(occ, 1) / max(prefix4_freq.get(prefix4, 1), 1),
                    "occ_share_in_prefix2": occ_freq.get(occ, 1) / max(prefix2_freq.get(prefix2, 1), 1),
                    "rare_flag": float(occ_freq.get(occ, 1) <= 5),
                    "ultra_rare_flag": float(occ_freq.get(occ, 1) <= 3),
                    "min_training_months": float(occ_row["min_train_months"]) if occ_row is not None else 0.0,
                }
            )
        group_sizes.append(min(len(candidates), 120))

    return pd.DataFrame(rows), np.asarray(group_sizes, dtype=np.int32)


def align_categoricals(train_df: pd.DataFrame, pred_df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        combined = pd.Index(
            pd.concat(
                [
                    train_df[col].fillna("unknown").astype(str),
                    pred_df[col].fillna("unknown").astype(str),
                ],
                ignore_index=True,
            ).unique()
        )
        train_df[col] = pd.Categorical(train_df[col].fillna("unknown").astype(str), categories=combined)
        pred_df[col] = pd.Categorical(pred_df[col].fillna("unknown").astype(str), categories=combined)


def fit_ranker(
    train_rank_df: pd.DataFrame,
    pred_rank_df: pd.DataFrame,
    group_train: np.ndarray,
    cat_cols: list[str],
) -> tuple[LGBMRanker, list[str]]:
    align_categoricals(train_rank_df, pred_rank_df, cat_cols)
    feature_cols = [c for c in train_rank_df.columns if c not in ["qid", "label"]]
    sample_weight = np.where(
        train_rank_df["label"].to_numpy() == 1,
        np.minimum(5.0, 1.0 + 2.0 / np.sqrt(train_rank_df["occ_freq"].to_numpy().clip(min=1.0))),
        1.0,
    )

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        force_row_wise=True,
        verbosity=-1,
    )
    ranker.fit(
        train_rank_df[feature_cols],
        train_rank_df["label"].astype(int),
        group=group_train,
        sample_weight=sample_weight,
        categorical_feature=cat_cols,
    )
    return ranker, feature_cols


def top5_from_scores(ranked_df: pd.DataFrame, score_col: str, tail_bonus: float = 0.0) -> list[list[str]]:
    outputs: list[list[str]] = []
    for _, group_df in ranked_df.groupby("qid", sort=False):
        df = group_df.copy()
        if tail_bonus:
            df["final_score"] = df[score_col] + tail_bonus * df["ultra_rare_flag"] * df["prefix4_model_score"]
        else:
            df["final_score"] = df[score_col]
        outputs.append(df.sort_values("final_score", ascending=False)["candidate_occ"].head(5).tolist())
    return outputs


def train_and_predict(
    fit: pd.DataFrame,
    pred_df: pd.DataFrame,
    skill_meta: pd.DataFrame,
    occ_meta: pd.DataFrame,
    label_available: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    query_fit = build_query_frame(fit, skill_meta)
    query_pred = build_query_frame(pred_df, skill_meta)

    vec_id = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    x_fit_id = vec_id.fit_transform(query_fit["id_doc"])
    x_pred_id = vec_id.transform(query_pred["id_doc"])

    vec_meta = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=60000)
    x_fit_meta = vec_meta.fit_transform(query_fit["meta_doc"])
    x_pred_meta = vec_meta.transform(query_pred["meta_doc"])

    x_fit_combined = hstack([x_fit_id, x_fit_meta]).tocsr()
    x_pred_combined = hstack([x_pred_id, x_pred_meta]).tocsr()

    occ_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=80000)
    occ_docs = occ_meta["occ_text"].tolist()
    occ_vec.fit(occ_docs + query_fit["meta_doc"].tolist())
    x_occ = occ_vec.transform(occ_docs)
    x_pred_occ_query = occ_vec.transform(query_pred["meta_doc"])

    sim_id = linear_kernel(x_pred_id, x_fit_id)
    sim_meta = linear_kernel(x_pred_meta, x_fit_meta)
    occ_text_sim = linear_kernel(x_pred_occ_query, x_occ)

    model2, mlb2, model4, mlb4 = fit_prefix_models(x_fit_combined, fit)
    prefix2_scores = model2.decision_function(x_pred_combined)
    prefix4_scores = model4.decision_function(x_pred_combined)
    if prefix2_scores.ndim == 1:
        prefix2_scores = prefix2_scores[:, None]
    if prefix4_scores.ndim == 1:
        prefix4_scores = prefix4_scores[:, None]

    artifacts = build_cooccurrence_artifacts(fit, skill_meta)
    candidate_df, group_sizes = build_candidate_frame(
        fit=fit,
        pred_df=pred_df,
        query_pred=query_pred,
        occ_meta=occ_meta,
        artifacts=artifacts,
        sim_id=sim_id,
        sim_meta=sim_meta,
        occ_text_sim=occ_text_sim,
        prefix2_scores=prefix2_scores,
        prefix2_classes=mlb2.classes_,
        prefix4_scores=prefix4_scores,
        prefix4_classes=mlb4.classes_,
        label_available=label_available,
    )
    return candidate_df, group_sizes


def run_validation(train: pd.DataFrame, skill_meta: pd.DataFrame, occ_meta: pd.DataFrame) -> None:
    fit, val = train_test_split(train, test_size=0.2, random_state=SEED, shuffle=True)
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_rank_df, group_train = train_and_predict(fit, fit, skill_meta, occ_meta, True)
    val_rank_df, _ = train_and_predict(fit, val, skill_meta, occ_meta, True)

    cat_cols = [
        "candidate_occ",
        "prefix2",
        "prefix4",
        "career_area",
        "occupation_group",
        "requirement_level",
        "license_required",
        "cert_required",
        "requires_training",
        "query_cat_signature",
        "query_type_signature",
        "query_subcat_signature",
        "skill_1",
        "skill_2",
        "skill_3",
        "skill_4",
        "skill_5",
        "skill_cat_1",
        "skill_cat_2",
        "skill_cat_3",
        "skill_cat_4",
        "skill_cat_5",
        "skill_type_1",
        "skill_type_2",
        "skill_type_3",
        "skill_type_4",
        "skill_type_5",
    ]
    ranker, feature_cols = fit_ranker(train_rank_df, val_rank_df, group_train, cat_cols)
    val_rank_df = val_rank_df.copy()
    val_rank_df["pred_score"] = ranker.predict(val_rank_df[feature_cols])

    scores = []
    for _, group_df in val_rank_df.groupby("qid", sort=False):
        actual = group_df.loc[group_df["label"] == 1, "candidate_occ"].tolist()
        predicted = group_df.sort_values("pred_score", ascending=False)["candidate_occ"].head(5).tolist()
        scores.append(apk(actual, predicted, 5))
    print(f"holdout MAP@5: {float(np.mean(scores)):.6f}", flush=True)


def build_submission_df(ids: list[str], top5s: list[list[str]]) -> pd.DataFrame:
    rows = []
    for item_id, top5 in zip(ids, top5s):
        rows.append(
            {
                "ID": item_id,
                "occ_1": top5[0],
                "occ_2": top5[1],
                "occ_3": top5[2],
                "occ_4": top5[3],
                "occ_5": top5[4],
            }
        )
    return pd.DataFrame(rows, columns=["ID", "occ_1", "occ_2", "occ_3", "occ_4", "occ_5"])


def main() -> None:
    train, test, skill_meta, occ_meta = load_data()
    run_validation(train, skill_meta, occ_meta)

    train_rank_df, group_train = train_and_predict(train, train, skill_meta, occ_meta, True)
    test_rank_df, _ = train_and_predict(train, test, skill_meta, occ_meta, False)

    cat_cols = [
        "candidate_occ",
        "prefix2",
        "prefix4",
        "career_area",
        "occupation_group",
        "requirement_level",
        "license_required",
        "cert_required",
        "requires_training",
        "query_cat_signature",
        "query_type_signature",
        "query_subcat_signature",
        "skill_1",
        "skill_2",
        "skill_3",
        "skill_4",
        "skill_5",
        "skill_cat_1",
        "skill_cat_2",
        "skill_cat_3",
        "skill_cat_4",
        "skill_cat_5",
        "skill_type_1",
        "skill_type_2",
        "skill_type_3",
        "skill_type_4",
        "skill_type_5",
    ]
    ranker, feature_cols = fit_ranker(train_rank_df, test_rank_df, group_train, cat_cols)
    test_rank_df = test_rank_df.copy()
    test_rank_df["pred_score"] = ranker.predict(test_rank_df[feature_cols])

    top5_base = top5_from_scores(test_rank_df, "pred_score", tail_bonus=0.0)
    top5_tail = top5_from_scores(test_rank_df, "pred_score", tail_bonus=0.01)

    base_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_base)
    tail_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_tail)

    base_path = DATA_DIR / "submission_hier_lgbm.csv"
    tail_path = DATA_DIR / "submission_hier_lgbm_tail.csv"
    base_submission.to_csv(base_path, index=False)
    tail_submission.to_csv(tail_path, index=False)
    print(f"saved {base_path.resolve()}", flush=True)
    print(f"saved {tail_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
