from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostRanker
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from lightgbm import LGBMRanker
from scipy import sparse
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

from make_hierarchical_ranker import (
    OCC_COLS,
    SKILL_COLS,
    aggregate_neighbor_scores,
    apk,
    build_cooccurrence_artifacts,
    build_query_frame,
    compute_cooccurrence_scores,
    empirical_support,
    fit_prefix_models,
    load_data,
    rank_dict,
)


SEED = 42
DATA_DIR = Path(".")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
np.random.seed(SEED)
torch.manual_seed(SEED)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CANDIDATES = 90
CROSS_ENCODER_TOP_K = int(os.environ.get("CROSS_ENCODER_TOP_K", "0"))
CROSS_ENCODER_MAX_LENGTH = int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "256"))
FINAL_CROSS_ENCODER_TOP_K = int(os.environ.get("FINAL_CROSS_ENCODER_TOP_K", "0"))
FINAL_CROSS_ENCODER_WEIGHT = float(os.environ.get("FINAL_CROSS_ENCODER_WEIGHT", "0.08"))
OOF_FOLDS = int(os.environ.get("OOF_FOLDS", "3"))
XGB_BAG_SEEDS = [int(x) for x in os.environ.get("XGB_BAG_SEEDS", "42,73,121").split(",") if x.strip()]
CATBOOST_SEED = int(os.environ.get("CATBOOST_SEED", "42"))
USE_FAST_DENSE = os.environ.get("USE_FAST_DENSE", "1") == "1"
SECOND_RANKER = os.environ.get("SECOND_RANKER", "lgbm").lower()
RECALL_EXPANSION = os.environ.get("RECALL_EXPANSION", "1") == "1"
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "submission_general_recall")
_CROSS_ENCODER = None

CAT_COLS = [
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


def build_sparse_knn(query_fit: pd.DataFrame, query_pred: pd.DataFrame):
    vec_id = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    x_fit_id = vec_id.fit_transform(query_fit["id_doc"])
    x_pred_id = vec_id.transform(query_pred["id_doc"])

    vec_meta = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=70000)
    x_fit_meta = vec_meta.fit_transform(query_fit["meta_doc"])
    x_pred_meta = vec_meta.transform(query_pred["meta_doc"])

    sim_id = linear_kernel(x_pred_id, x_fit_id)
    sim_meta = linear_kernel(x_pred_meta, x_fit_meta)

    if not RECALL_EXPANSION:
        sim_set = np.zeros((x_pred_id.shape[0], x_fit_id.shape[0]), dtype=np.float32)
        sim_sig = np.zeros((x_pred_id.shape[0], x_fit_id.shape[0]), dtype=np.float32)
        return x_fit_id, x_pred_id, x_fit_meta, x_pred_meta, sim_id, sim_meta, sim_set, sim_sig

    fit_set_docs = query_fit["id_doc"].map(lambda text: " ".join(sorted(set(str(text).split()))))
    pred_set_docs = query_pred["id_doc"].map(lambda text: " ".join(sorted(set(str(text).split()))))
    vec_set = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    x_fit_set = vec_set.fit_transform(fit_set_docs)
    x_pred_set = vec_set.transform(pred_set_docs)

    fit_sig_docs = (
        query_fit["query_cat_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_fit["query_type_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_fit["query_subcat_signature"].astype(str).str.replace("|", " ", regex=False)
    )
    pred_sig_docs = (
        query_pred["query_cat_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_pred["query_type_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_pred["query_subcat_signature"].astype(str).str.replace("|", " ", regex=False)
    )
    vec_sig = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=45000)
    x_fit_sig = vec_sig.fit_transform(fit_sig_docs)
    x_pred_sig = vec_sig.transform(pred_sig_docs)

    sim_set = linear_kernel(x_pred_set, x_fit_set)
    sim_sig = linear_kernel(x_pred_sig, x_fit_sig)
    return x_fit_id, x_pred_id, x_fit_meta, x_pred_meta, sim_id, sim_meta, sim_set, sim_sig


def build_skill_occ_matrix(
    fit: pd.DataFrame, occ_meta: pd.DataFrame
) -> tuple[sparse.csr_matrix, dict[str, int], dict[str, int], list[str], list[str]]:
    skill_ids = sorted({str(x) for x in fit[SKILL_COLS].values.ravel()})
    occ_ids = sorted(occ_meta.index.astype(str).tolist())
    skill_to_idx = {sid: i for i, sid in enumerate(skill_ids)}
    occ_to_idx = {oid: i for i, oid in enumerate(occ_ids)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row in fit.itertuples(index=False):
        qskills = sorted({str(getattr(row, c)) for c in SKILL_COLS})
        occs = [str(getattr(row, c)) for c in OCC_COLS]
        for sid in qskills:
            if sid not in skill_to_idx:
                continue
            s_idx = skill_to_idx[sid]
            for occ in occs:
                if occ not in occ_to_idx:
                    continue
                rows.append(s_idx)
                cols.append(occ_to_idx[occ])
                vals.append(1.0)

    matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(len(skill_ids), len(occ_ids)), dtype=np.float32).tocsr()
    matrix.sort_indices()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix, skill_to_idx, occ_to_idx, skill_ids, occ_ids


def precompute_implicit_scores(
    fit: pd.DataFrame,
    occ_meta: pd.DataFrame,
    skill_weight: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[str], dict[str, int]]:
    with threadpool_limits(limits=1, user_api="blas"):
        matrix, skill_to_idx, occ_to_idx, skill_ids, occ_ids = build_skill_occ_matrix(fit, occ_meta)
        bm25_matrix = bm25_weight(matrix, K1=100, B=0.8).tocsr()
        bm25_matrix.sort_indices()

        als = AlternatingLeastSquares(
            factors=64,
            regularization=0.02,
            iterations=40,
            random_state=SEED,
            use_gpu=False,
        )
        als.fit(matrix, show_progress=True)

    idx_to_occ = {idx: occ for occ, idx in occ_to_idx.items()}
    bm25_scores: dict[str, dict[str, float]] = {}
    als_scores: dict[str, dict[str, float]] = {}
    for sid in skill_ids:
        s_idx = skill_to_idx[sid]
        als_idx, als_sc = als.recommend(s_idx, matrix, N=40, filter_already_liked_items=False)
        row = bm25_matrix.getrow(s_idx)
        if row.nnz:
            top_idx = np.argsort(-row.data)[:40]
            bm25_scores[sid] = {
                idx_to_occ[int(row.indices[i])]: float(row.data[i]) * skill_weight.get(sid, 1.0)
                for i in top_idx
            }
        else:
            bm25_scores[sid] = {}
        als_scores[sid] = {idx_to_occ[i]: float(score) * skill_weight.get(sid, 1.0) for i, score in zip(als_idx, als_sc)}
    return bm25_scores, als_scores, occ_ids, occ_to_idx


def aggregate_skill_based_scores(
    query_skills: list[str], per_skill_scores: dict[str, dict[str, float]]
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for sid in query_skills:
        if sid not in per_skill_scores:
            continue
        for occ, score in per_skill_scores[sid].items():
            scores[occ] += score
            counts[occ] += 1
    return scores, counts


def build_extra_retrieval_artifacts(
    fit: pd.DataFrame,
    skill_meta: pd.DataFrame,
    occ_meta: pd.DataFrame,
) -> dict[str, object]:
    cat_occ: dict[str, Counter[str]] = defaultdict(Counter)
    subcat_occ: dict[str, Counter[str]] = defaultdict(Counter)
    type_occ: dict[str, Counter[str]] = defaultdict(Counter)
    cat_group: dict[str, Counter[str]] = defaultdict(Counter)
    type_group: dict[str, Counter[str]] = defaultdict(Counter)
    cat_career: dict[str, Counter[str]] = defaultdict(Counter)
    type_career: dict[str, Counter[str]] = defaultdict(Counter)
    skill_graph: dict[str, Counter[str]] = defaultdict(Counter)
    group_to_occ: dict[str, Counter[str]] = defaultdict(Counter)
    career_to_occ: dict[str, Counter[str]] = defaultdict(Counter)
    occ_profile_id_terms: dict[str, list[str]] = defaultdict(list)
    occ_profile_meta_terms: dict[str, list[str]] = defaultdict(list)
    group_profile_terms: dict[str, list[str]] = defaultdict(list)
    career_profile_terms: dict[str, list[str]] = defaultdict(list)
    skill_freq: Counter[str] = Counter()
    group_freq: Counter[str] = Counter()
    career_freq: Counter[str] = Counter()

    occ_group_map = occ_meta["OCCUPATION_GROUP_NAME"].astype(str).to_dict()
    occ_career_map = occ_meta["CAREER_AREA_NAME"].astype(str).to_dict()

    for row in fit.itertuples(index=False):
        qskills = sorted({str(getattr(row, c)) for c in SKILL_COLS})
        labels = [str(getattr(row, c)) for c in OCC_COLS]
        cats: set[str] = set()
        subcats: set[str] = set()
        types: set[str] = set()

        for skill_id in qskills:
            skill_freq[skill_id] += 1
            if skill_id not in skill_meta.index:
                continue
            srow = skill_meta.loc[skill_id]
            cats.add(str(srow.get("CATEGORY_NAME", "unknown")) or "unknown")
            subcats.add(str(srow.get("SUBCATEGORY_NAME", "unknown")) or "unknown")
            types.add(str(srow.get("TYPE", "unknown")) or "unknown")

        groups = {occ_group_map.get(occ, "unknown") for occ in labels}
        careers = {occ_career_map.get(occ, "unknown") for occ in labels}
        for group in groups:
            group_freq[group] += 1
        for career in careers:
            career_freq[career] += 1
            career_profile_terms[career].extend(qskills)
            career_profile_terms[career].extend(sorted(cats))
            career_profile_terms[career].extend(sorted(subcats))
            career_profile_terms[career].extend(sorted(types))
        for occ in labels:
            group_to_occ[occ_group_map.get(occ, "unknown")][occ] += 1
            career_to_occ[occ_career_map.get(occ, "unknown")][occ] += 1
            occ_profile_id_terms[occ].extend(qskills)
            occ_profile_meta_terms[occ].extend(sorted(cats))
            occ_profile_meta_terms[occ].extend(sorted(subcats))
            occ_profile_meta_terms[occ].extend(sorted(types))
            for cat in cats:
                cat_occ[cat][occ] += 1
            for subcat in subcats:
                subcat_occ[subcat][occ] += 1
            for skill_type in types:
                type_occ[skill_type][occ] += 1

        for cat in cats:
            for group in groups:
                cat_group[cat][group] += 1
                group_profile_terms[group].append(cat)
            for career in careers:
                cat_career[cat][career] += 1
        for skill_type in types:
            for group in groups:
                type_group[skill_type][group] += 1
            for career in careers:
                type_career[skill_type][career] += 1
                career_profile_terms[career].append(skill_type)

        for i, s1 in enumerate(qskills):
            for s2 in qskills[i + 1 :]:
                skill_graph[s1][s2] += 1
                skill_graph[s2][s1] += 1

    max_skill_freq = max(skill_freq.values(), default=1)
    skill_graph_weight = {
        sid: math.log1p(max_skill_freq / max(freq, 1)) + 1.0
        for sid, freq in skill_freq.items()
    }
    occ_profile_id_doc = {}
    occ_profile_meta_doc = {}
    for occ in occ_meta.index.astype(str):
        occ_row = occ_meta.loc[occ]
        occ_profile_id_doc[occ] = " ".join(occ_profile_id_terms.get(occ, [])) or str(occ)
        occ_profile_meta_doc[occ] = " ".join(
            occ_profile_meta_terms.get(occ, [])
            + [
                str(occ_row.get("OCCUPATION_NAME", "")),
                str(occ_row.get("OCCUPATION_GROUP_NAME", "")),
                str(occ_row.get("CAREER_AREA_NAME", "")),
                str(occ_row.get("REQUIREMENT_LEVEL", "")),
                str(occ_row.get("OCCUPATION_DESCRIPTION", ""))[:450],
            ]
        )
    group_profile_doc = {
        group_name: " ".join(terms + [group_name]) for group_name, terms in group_profile_terms.items()
    }
    career_profile_doc = {
        career_name: " ".join(terms + [career_name]) for career_name, terms in career_profile_terms.items()
    }
    return {
        "cat_occ": cat_occ,
        "subcat_occ": subcat_occ,
        "type_occ": type_occ,
        "cat_group": cat_group,
        "type_group": type_group,
        "cat_career": cat_career,
        "type_career": type_career,
        "skill_graph": skill_graph,
        "group_to_occ": group_to_occ,
        "career_to_occ": career_to_occ,
        "skill_graph_weight": skill_graph_weight,
        "group_freq": group_freq,
        "career_freq": career_freq,
        "occ_profile_id_doc": occ_profile_id_doc,
        "occ_profile_meta_doc": occ_profile_meta_doc,
        "group_profile_doc": group_profile_doc,
        "career_profile_doc": career_profile_doc,
    }


def aggregate_counter_scores(
    keys: list[str],
    mapping: dict[str, Counter[str]],
    occ_freq: Counter[str],
    topn_per_key: int,
    rarity_power: float = 0.15,
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    seen = set()
    for key in keys:
        key = str(key)
        if not key or key == "unknown" or key in seen or key not in mapping:
            continue
        seen.add(key)
        for item, cnt in mapping[key].most_common(topn_per_key):
            denom = max(occ_freq.get(item, 1), 1) ** rarity_power if rarity_power else 1.0
            scores[item] += math.log1p(cnt) / denom
            counts[item] += 1
    return scores, counts


def compute_skill_graph_scores(
    qskills: list[str],
    skill_graph: dict[str, Counter[str]],
    skill_occ: dict[str, Counter[str]],
    occ_freq: Counter[str],
    skill_graph_weight: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    seen = set()
    for sid in qskills:
        sid = str(sid)
        if sid in seen or sid not in skill_graph:
            continue
        seen.add(sid)
        for nbr_skill, edge_cnt in skill_graph[sid].most_common(8):
            occ_counter = skill_occ.get(nbr_skill, {})
            if not occ_counter:
                continue
            edge_weight = math.log1p(edge_cnt) * skill_graph_weight.get(nbr_skill, 1.0)
            for occ, occ_cnt in occ_counter.most_common(10):
                denom = max(occ_freq.get(occ, 1), 1) ** 0.12
                scores[occ] += edge_weight * math.log1p(occ_cnt) / denom
                counts[occ] += 1
    return scores, counts


def compute_skill_walk_scores(
    qskills: list[str],
    skill_graph: dict[str, Counter[str]],
    skill_occ: dict[str, Counter[str]],
    occ_freq: Counter[str],
    skill_graph_weight: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    seed_skills = {str(sid) for sid in qskills}
    frontier: dict[str, float] = {}

    for sid in seed_skills:
        if sid not in skill_graph:
            continue
        for nbr_skill, edge_cnt in skill_graph[sid].most_common(14):
            if nbr_skill in seed_skills:
                continue
            base_weight = math.log1p(edge_cnt) * skill_graph_weight.get(nbr_skill, 1.0)
            frontier[nbr_skill] = max(frontier.get(nbr_skill, 0.0), base_weight)
            occ_counter = skill_occ.get(nbr_skill, {})
            for occ, occ_cnt in occ_counter.most_common(14):
                denom = max(occ_freq.get(occ, 1), 1) ** 0.08
                scores[occ] += base_weight * math.log1p(occ_cnt) / denom
                counts[occ] += 1

    for nbr_skill, frontier_weight in sorted(frontier.items(), key=lambda item: item[1], reverse=True)[:18]:
        for nbr2_skill, edge_cnt in skill_graph.get(nbr_skill, {}).most_common(6):
            if nbr2_skill in seed_skills or nbr2_skill in frontier:
                continue
            occ_counter = skill_occ.get(nbr2_skill, {})
            if not occ_counter:
                continue
            walk_weight = 0.45 * frontier_weight * math.log1p(edge_cnt) * skill_graph_weight.get(nbr2_skill, 1.0)
            for occ, occ_cnt in occ_counter.most_common(8):
                denom = max(occ_freq.get(occ, 1), 1) ** 0.05
                scores[occ] += walk_weight * math.log1p(occ_cnt) / denom
                counts[occ] += 1
    return scores, counts


def build_dense_semantic_scores(
    query_fit: pd.DataFrame,
    query_pred: pd.DataFrame,
    occ_meta: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str], list[str]]:
    occ_texts, occ_dense_texts, occ_ce_map = build_occ_text_payloads(occ_meta)
    query_dense_texts = [str(text)[:500] for text in query_pred["meta_doc"].tolist()]
    query_ce_texts = [str(text)[:600] for text in query_pred["meta_doc"].tolist()]
    if USE_FAST_DENSE:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=60000)
        occ_emb = vec.fit_transform(occ_dense_texts)
        query_emb = vec.transform(query_dense_texts)
        dense_scores = linear_kernel(query_emb, occ_emb)
    else:
        model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
        occ_emb = model.encode(occ_dense_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        query_emb = model.encode(query_dense_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        dense_scores = query_emb @ occ_emb.T
    return dense_scores, occ_meta.index.astype(str).tolist(), query_ce_texts, occ_ce_map, occ_texts


def topk_dense_score_dict(score_row: np.ndarray, ids: list[str], topn: int) -> dict[str, float]:
    if score_row.shape[0] == 0 or not ids or topn <= 0:
        return {}
    take = min(topn, score_row.shape[0])
    idx = np.argpartition(-score_row, take - 1)[:take]
    idx = idx[np.argsort(-score_row[idx])]
    return {str(ids[i]): float(score_row[i]) for i in idx if float(score_row[i]) > 0.0}


def build_profile_similarity_scores(
    query_pred: pd.DataFrame,
    extra_artifacts: dict[str, object],
    occ_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    occ_ids = occ_meta.index.astype(str).tolist()
    if not RECALL_EXPANSION:
        zero_occ = np.zeros((len(query_pred), len(occ_ids)), dtype=np.float32)
        zero_group = np.zeros((len(query_pred), 0), dtype=np.float32)
        zero_career = np.zeros((len(query_pred), 0), dtype=np.float32)
        return zero_occ, zero_occ.copy(), occ_ids, zero_group, [], zero_career, []

    query_id_docs = query_pred["id_doc"].map(lambda text: " ".join(sorted(set(str(text).split())))).tolist()
    query_meta_docs = (
        query_pred["meta_doc"].astype(str)
        + " "
        + query_pred["query_cat_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_pred["query_type_signature"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + query_pred["query_subcat_signature"].astype(str).str.replace("|", " ", regex=False)
    ).tolist()

    occ_profile_id_docs = [str(extra_artifacts["occ_profile_id_doc"].get(occ, occ)) for occ in occ_ids]
    occ_profile_meta_docs = [str(extra_artifacts["occ_profile_meta_doc"].get(occ, occ)) for occ in occ_ids]

    vec_occ_id = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, norm="l2", use_idf=True)
    occ_id_mat = vec_occ_id.fit_transform(occ_profile_id_docs)
    query_id_mat = vec_occ_id.transform(query_id_docs)
    occ_profile_id_scores = linear_kernel(query_id_mat, occ_id_mat)

    vec_occ_meta = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=50000)
    occ_meta_mat = vec_occ_meta.fit_transform(occ_profile_meta_docs)
    query_meta_mat = vec_occ_meta.transform(query_meta_docs)
    occ_profile_meta_scores = linear_kernel(query_meta_mat, occ_meta_mat)

    group_names = sorted(str(x) for x in extra_artifacts["group_profile_doc"].keys())
    if group_names:
        group_docs = [str(extra_artifacts["group_profile_doc"][name]) for name in group_names]
        vec_group = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=30000)
        group_mat = vec_group.fit_transform(group_docs)
        group_query_mat = vec_group.transform(query_meta_docs)
        group_profile_scores = linear_kernel(group_query_mat, group_mat)
    else:
        group_profile_scores = np.zeros((len(query_pred), 0), dtype=np.float32)

    career_names = sorted(str(x) for x in extra_artifacts["career_profile_doc"].keys())
    if career_names:
        career_docs = [str(extra_artifacts["career_profile_doc"][name]) for name in career_names]
        vec_career = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=25000)
        career_mat = vec_career.fit_transform(career_docs)
        career_query_mat = vec_career.transform(query_meta_docs)
        career_profile_scores = linear_kernel(career_query_mat, career_mat)
    else:
        career_profile_scores = np.zeros((len(query_pred), 0), dtype=np.float32)

    return (
        occ_profile_id_scores,
        occ_profile_meta_scores,
        occ_ids,
        group_profile_scores,
        group_names,
        career_profile_scores,
        career_names,
    )


def build_occ_text_payloads(occ_meta: pd.DataFrame) -> tuple[list[str], list[str], dict[str, str]]:
    occ_texts = occ_meta["occ_text"].tolist()
    occ_dense_texts = [
        " ".join(
            [
                str(occ_meta.iloc[i]["OCCUPATION_NAME"]),
                str(occ_meta.iloc[i]["OCCUPATION_GROUP_NAME"]),
                str(occ_meta.iloc[i]["CAREER_AREA_NAME"]),
                str(occ_meta.iloc[i]["OCCUPATION_DESCRIPTION"])[:700],
            ]
        )
        for i in range(len(occ_meta))
    ]
    occ_ce_texts = [text[:600] for text in occ_dense_texts]
    occ_ce_map = {occ_id: text for occ_id, text in zip(occ_meta.index.astype(str).tolist(), occ_ce_texts)}
    return occ_texts, occ_dense_texts, occ_ce_map


def get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE, max_length=CROSS_ENCODER_MAX_LENGTH)
    return _CROSS_ENCODER


def build_candidate_frame(
    fit: pd.DataFrame,
    pred_df: pd.DataFrame,
    query_pred: pd.DataFrame,
    occ_meta: pd.DataFrame,
    artifacts: dict[str, object],
    extra_artifacts: dict[str, object],
    sim_id: np.ndarray,
    sim_meta: np.ndarray,
    sim_set: np.ndarray,
    sim_sig: np.ndarray,
    bm25_skill_scores: dict[str, dict[str, float]],
    als_skill_scores: dict[str, dict[str, float]],
    dense_scores: np.ndarray,
    dense_occ_ids: list[str],
    occ_profile_id_scores: np.ndarray,
    occ_profile_meta_scores: np.ndarray,
    occ_profile_occ_ids: list[str],
    group_profile_scores: np.ndarray,
    group_profile_names: list[str],
    career_profile_scores: np.ndarray,
    career_profile_names: list[str],
    prefix2_scores: np.ndarray,
    prefix2_classes: np.ndarray,
    prefix4_scores: np.ndarray,
    prefix4_classes: np.ndarray,
    label_available: bool,
    same_source: bool = False,
    inject_actual_labels: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    occ_freq = artifacts["occ_counter"]
    prefix2_freq = artifacts["prefix2_counter"]
    prefix4_freq = artifacts["prefix4_counter"]
    group_freq = extra_artifacts["group_freq"]
    career_freq = extra_artifacts["career_freq"]
    group_to_occ = extra_artifacts["group_to_occ"]
    career_to_occ = extra_artifacts["career_to_occ"]

    rows: list[dict[str, object]] = []
    group_sizes: list[int] = []
    for qid, (_, row) in enumerate(pred_df.iterrows()):
        qskills = row[SKILL_COLS].tolist()
        query_info = query_pred.loc[row.name]
        actual = set(row[OCC_COLS].tolist()) if label_available else set()
        sim_id_row = sim_id[qid].copy() if same_source else sim_id[qid]
        sim_meta_row = sim_meta[qid].copy() if same_source else sim_meta[qid]
        sim_set_row = sim_set[qid].copy() if same_source else sim_set[qid]
        sim_sig_row = sim_sig[qid].copy() if same_source else sim_sig[qid]
        if same_source and qid < sim_id_row.shape[0]:
            sim_id_row[qid] = -np.inf
        if same_source and qid < sim_meta_row.shape[0]:
            sim_meta_row[qid] = -np.inf
        if same_source and qid < sim_set_row.shape[0]:
            sim_set_row[qid] = -np.inf
        if same_source and qid < sim_sig_row.shape[0]:
            sim_sig_row[qid] = -np.inf

        id_scores, id_counts = aggregate_neighbor_scores(fit, sim_id_row, occ_freq, topn_rows=45, damp=0.15)
        meta_scores, meta_counts = aggregate_neighbor_scores(fit, sim_meta_row, occ_freq, topn_rows=70, damp=0.10)
        set_scores, set_counts = aggregate_neighbor_scores(fit, sim_set_row, occ_freq, topn_rows=55, damp=0.12)
        sig_scores, sig_counts = aggregate_neighbor_scores(fit, sim_sig_row, occ_freq, topn_rows=60, damp=0.08)
        bm25_scores, bm25_counts = aggregate_skill_based_scores(qskills, bm25_skill_scores)
        als_scores, als_counts = aggregate_skill_based_scores(qskills, als_skill_scores)
        co_scores, co_skill_hits, co_pair_hits = compute_cooccurrence_scores(qskills, artifacts)
        query_cats = [str(query_info[f"skill_cat_{i}"]) for i in range(1, 6)]
        query_types = [str(query_info[f"skill_type_{i}"]) for i in range(1, 6)]
        query_subcats = [x for x in str(query_info["query_subcat_signature"]).split("|") if x and x != "unknown"]

        cat_occ_scores, cat_occ_counts = aggregate_counter_scores(
            query_cats, extra_artifacts["cat_occ"], occ_freq, topn_per_key=15, rarity_power=0.16
        )
        subcat_occ_scores, subcat_occ_counts = aggregate_counter_scores(
            query_subcats, extra_artifacts["subcat_occ"], occ_freq, topn_per_key=12, rarity_power=0.12
        )
        type_occ_scores, type_occ_counts = aggregate_counter_scores(
            query_types, extra_artifacts["type_occ"], occ_freq, topn_per_key=15, rarity_power=0.10
        )
        graph_scores, graph_counts = compute_skill_graph_scores(
            qskills,
            extra_artifacts["skill_graph"],
            artifacts["skill_occ"],
            occ_freq,
            extra_artifacts["skill_graph_weight"],
        )
        if RECALL_EXPANSION:
            walk_scores, walk_counts = compute_skill_walk_scores(
                qskills,
                extra_artifacts["skill_graph"],
                artifacts["skill_occ"],
                occ_freq,
                extra_artifacts["skill_graph_weight"],
            )
        else:
            walk_scores, walk_counts = {}, {}
        profile_id_occ_scores = topk_dense_score_dict(occ_profile_id_scores[qid], occ_profile_occ_ids, topn=30)
        profile_meta_occ_scores = topk_dense_score_dict(occ_profile_meta_scores[qid], occ_profile_occ_ids, topn=30)

        group_scores = defaultdict(float)
        for key_list, mapping in [
            (query_cats, extra_artifacts["cat_group"]),
            (query_types, extra_artifacts["type_group"]),
        ]:
            partial_scores, _ = aggregate_counter_scores(
                key_list,
                mapping,
                group_freq,
                topn_per_key=8,
                rarity_power=0.0,
            )
            for group_name, score in partial_scores.items():
                group_scores[group_name] += score
        group_profile_dict = topk_dense_score_dict(group_profile_scores[qid], group_profile_names, topn=8)
        for group_name, score in group_profile_dict.items():
            group_scores[group_name] += 0.9 * score

        career_scores = defaultdict(float)
        for key_list, mapping in [
            (query_cats, extra_artifacts["cat_career"]),
            (query_types, extra_artifacts["type_career"]),
        ]:
            partial_scores, _ = aggregate_counter_scores(
                key_list,
                mapping,
                career_freq,
                topn_per_key=6,
                rarity_power=0.0,
            )
            for career_name, score in partial_scores.items():
                career_scores[career_name] += score
        career_profile_dict = topk_dense_score_dict(career_profile_scores[qid], career_profile_names, topn=5)
        for career_name, score in career_profile_dict.items():
            career_scores[career_name] += 0.9 * score

        dense_take = min(30, dense_scores.shape[1])
        dense_idx = np.argpartition(-dense_scores[qid], dense_take - 1)[:dense_take]
        dense_idx = dense_idx[np.argsort(-dense_scores[qid][dense_idx])]
        semantic_scores = {dense_occ_ids[i]: float(dense_scores[qid][i]) for i in dense_idx}

        p2_scores = {cls: float(score) for cls, score in zip(prefix2_classes, prefix2_scores[qid])}
        p4_scores = {cls: float(score) for cls, score in zip(prefix4_classes, prefix4_scores[qid])}

        fused_scores: dict[str, float] = defaultdict(float)
        for score_dict, weight in [
            (id_scores, 1.0),
            (meta_scores, 0.35),
            (set_scores, 0.55),
            (sig_scores, 0.25),
            (bm25_scores, 0.40),
            (als_scores, 0.35),
            (co_scores, 0.15),
            (cat_occ_scores, 0.25),
            (subcat_occ_scores, 0.20),
            (type_occ_scores, 0.15),
            (graph_scores, 0.22),
            (walk_scores, 0.26),
            (profile_id_occ_scores, 0.34),
            (profile_meta_occ_scores, 0.28),
            (semantic_scores, 0.45),
        ]:
            for occ, score in score_dict.items():
                fused_scores[occ] += weight * score

        candidates: list[str] = []
        sources = [
            sorted(id_scores, key=id_scores.get, reverse=True)[:25],
            sorted(meta_scores, key=meta_scores.get, reverse=True)[:20],
            sorted(set_scores, key=set_scores.get, reverse=True)[:20],
            sorted(sig_scores, key=sig_scores.get, reverse=True)[:15],
            sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:20],
            sorted(als_scores, key=als_scores.get, reverse=True)[:20],
            sorted(co_scores, key=co_scores.get, reverse=True)[:20],
            sorted(cat_occ_scores, key=cat_occ_scores.get, reverse=True)[:15],
            sorted(subcat_occ_scores, key=subcat_occ_scores.get, reverse=True)[:12],
            sorted(type_occ_scores, key=type_occ_scores.get, reverse=True)[:12],
            sorted(graph_scores, key=graph_scores.get, reverse=True)[:15],
            sorted(walk_scores, key=walk_scores.get, reverse=True)[:15],
            sorted(profile_id_occ_scores, key=profile_id_occ_scores.get, reverse=True)[:20],
            sorted(profile_meta_occ_scores, key=profile_meta_occ_scores.get, reverse=True)[:20],
            sorted(semantic_scores, key=semantic_scores.get, reverse=True)[:20],
        ]
        for source in sources:
            for occ in source:
                if occ not in candidates:
                    candidates.append(occ)

        if label_available and inject_actual_labels:
            actual_ordered = [str(getattr(row, col)) for col in OCC_COLS]
            actual_prefix = []
            seen_actual = set()
            for occ in actual_ordered:
                if occ not in seen_actual:
                    actual_prefix.append(occ)
                    seen_actual.add(occ)
            candidates = actual_prefix + [occ for occ in candidates if occ not in seen_actual]

        for prefix4 in sorted(p4_scores, key=p4_scores.get, reverse=True)[:5]:
            if prefix4 not in artifacts["prefix4_to_occ"]:
                continue
            ranked = sorted(
                artifacts["prefix4_to_occ"][prefix4],
                key=lambda occ: fused_scores.get(occ, 0.0) + 0.4 * semantic_scores.get(occ, 0.0) + 0.2 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:6]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        for prefix2 in sorted(p2_scores, key=p2_scores.get, reverse=True)[:3]:
            if prefix2 not in artifacts["prefix2_to_occ"]:
                continue
            ranked = sorted(
                artifacts["prefix2_to_occ"][prefix2],
                key=lambda occ: fused_scores.get(occ, 0.0) + 0.2 * semantic_scores.get(occ, 0.0) + 0.25 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:8]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        for group_name in sorted(group_scores, key=group_scores.get, reverse=True)[:4]:
            if group_name not in group_to_occ:
                continue
            ranked = sorted(
                group_to_occ[group_name],
                key=lambda occ: fused_scores.get(occ, 0.0) + 0.12 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:6]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        for career_name in sorted(career_scores, key=career_scores.get, reverse=True)[:2]:
            if career_name not in career_to_occ:
                continue
            ranked = sorted(
                career_to_occ[career_name],
                key=lambda occ: fused_scores.get(occ, 0.0) + 0.08 / math.sqrt(max(occ_freq.get(occ, 0), 1)),
                reverse=True,
            )[:6]
            for occ in ranked:
                if occ not in candidates:
                    candidates.append(occ)

        rare_candidates = sorted(
            fused_scores,
            key=lambda occ: (fused_scores.get(occ, 0.0) + 0.2 * semantic_scores.get(occ, 0.0)) / math.sqrt(max(occ_freq.get(occ, 0), 1)),
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
        set_rank = rank_dict(set_scores)
        sig_rank = rank_dict(sig_scores)
        bm25_rank = rank_dict(bm25_scores)
        als_rank = rank_dict(als_scores)
        co_rank = rank_dict(co_scores)
        cat_occ_rank = rank_dict(cat_occ_scores)
        subcat_occ_rank = rank_dict(subcat_occ_scores)
        type_occ_rank = rank_dict(type_occ_scores)
        graph_rank = rank_dict(graph_scores)
        walk_rank = rank_dict(walk_scores)
        profile_id_rank = rank_dict(profile_id_occ_scores)
        profile_meta_rank = rank_dict(profile_meta_occ_scores)
        semantic_rank = rank_dict(semantic_scores)
        fused_rank = rank_dict(fused_scores)

        for occ in candidates[:MAX_CANDIDATES]:
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
                    "set_score": float(set_scores.get(occ, 0.0)),
                    "sig_score": float(sig_scores.get(occ, 0.0)),
                    "bm25_score": float(bm25_scores.get(occ, 0.0)),
                    "als_score": float(als_scores.get(occ, 0.0)),
                    "co_score": float(co_scores.get(occ, 0.0)),
                    "cat_occ_score": float(cat_occ_scores.get(occ, 0.0)),
                    "subcat_occ_score": float(subcat_occ_scores.get(occ, 0.0)),
                    "type_occ_score": float(type_occ_scores.get(occ, 0.0)),
                    "graph_score": float(graph_scores.get(occ, 0.0)),
                    "walk_score": float(walk_scores.get(occ, 0.0)),
                    "profile_id_score": float(profile_id_occ_scores.get(occ, 0.0)),
                    "profile_meta_score": float(profile_meta_occ_scores.get(occ, 0.0)),
                    "semantic_score": float(semantic_scores.get(occ, 0.0)),
                    "id_rank": float(id_rank.get(occ, 999.0)),
                    "meta_rank": float(meta_rank.get(occ, 999.0)),
                    "set_rank": float(set_rank.get(occ, 999.0)),
                    "sig_rank": float(sig_rank.get(occ, 999.0)),
                    "bm25_rank": float(bm25_rank.get(occ, 999.0)),
                    "als_rank": float(als_rank.get(occ, 999.0)),
                    "co_rank": float(co_rank.get(occ, 999.0)),
                    "cat_occ_rank": float(cat_occ_rank.get(occ, 999.0)),
                    "subcat_occ_rank": float(subcat_occ_rank.get(occ, 999.0)),
                    "type_occ_rank": float(type_occ_rank.get(occ, 999.0)),
                    "graph_rank": float(graph_rank.get(occ, 999.0)),
                    "walk_rank": float(walk_rank.get(occ, 999.0)),
                    "profile_id_rank": float(profile_id_rank.get(occ, 999.0)),
                    "profile_meta_rank": float(profile_meta_rank.get(occ, 999.0)),
                    "semantic_rank": float(semantic_rank.get(occ, 999.0)),
                    "fused_rank": float(fused_rank.get(occ, 999.0)),
                    "id_support_count": float(id_counts.get(occ, 0)),
                    "meta_support_count": float(meta_counts.get(occ, 0)),
                    "set_support_count": float(set_counts.get(occ, 0)),
                    "sig_support_count": float(sig_counts.get(occ, 0)),
                    "bm25_support_count": float(bm25_counts.get(occ, 0)),
                    "als_support_count": float(als_counts.get(occ, 0)),
                    "co_skill_hits": float(co_skill_hits.get(occ, 0)),
                    "co_pair_hits": float(co_pair_hits.get(occ, 0)),
                    "cat_occ_support_count": float(cat_occ_counts.get(occ, 0)),
                    "subcat_occ_support_count": float(subcat_occ_counts.get(occ, 0)),
                    "type_occ_support_count": float(type_occ_counts.get(occ, 0)),
                    "graph_support_count": float(graph_counts.get(occ, 0)),
                    "walk_support_count": float(walk_counts.get(occ, 0)),
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
                    "group_retrieval_score": float(group_scores.get(str(occ_row["OCCUPATION_GROUP_NAME"]) if occ_row is not None else "unknown", 0.0)),
                    "career_retrieval_score": float(career_scores.get(str(occ_row["CAREER_AREA_NAME"]) if occ_row is not None else "unknown", 0.0)),
                    "group_profile_score": float(group_profile_dict.get(str(occ_row["OCCUPATION_GROUP_NAME"]) if occ_row is not None else "unknown", 0.0)),
                    "career_profile_score": float(career_profile_dict.get(str(occ_row["CAREER_AREA_NAME"]) if occ_row is not None else "unknown", 0.0)),
                    "retrieval_vote_count": float(
                        sum(
                            int(score_dict.get(occ, 0.0) > 0.0)
                            for score_dict in [
                                id_scores,
                                meta_scores,
                                set_scores,
                                sig_scores,
                                bm25_scores,
                                als_scores,
                                co_scores,
                                cat_occ_scores,
                                subcat_occ_scores,
                                type_occ_scores,
                                graph_scores,
                                walk_scores,
                                profile_id_occ_scores,
                                profile_meta_occ_scores,
                                semantic_scores,
                            ]
                        )
                    ),
                    "occ_freq": float(occ_freq.get(occ, 0)),
                    "occ_log_freq": math.log1p(max(occ_freq.get(occ, 0), 0)),
                    "prefix2_freq": float(prefix2_freq.get(prefix2, 0)),
                    "prefix4_freq": float(prefix4_freq.get(prefix4, 0)),
                    "occ_share_in_prefix4": occ_freq.get(occ, 0) / max(prefix4_freq.get(prefix4, 1), 1),
                    "occ_share_in_prefix2": occ_freq.get(occ, 0) / max(prefix2_freq.get(prefix2, 1), 1),
                    "rare_flag": float(occ_freq.get(occ, 0) <= 5),
                    "ultra_rare_flag": float(occ_freq.get(occ, 0) <= 3),
                    "min_training_months": float(occ_row["min_train_months"]) if occ_row is not None else 0.0,
                }
            )
        group_sizes.append(min(len(candidates), MAX_CANDIDATES))

    return pd.DataFrame(rows), np.asarray(group_sizes, dtype=np.int32)


def add_cross_encoder_scores(
    df: pd.DataFrame,
    query_texts: list[str],
    occ_text_map: dict[str, str],
) -> pd.DataFrame:
    df = df.copy()
    df["cross_score"] = np.float32(0.0)
    if CROSS_ENCODER_TOP_K <= 0 or df.empty:
        print("Skipping cross-encoder scoring.", flush=True)
        return df

    score_index = (
        df.sort_values(
            ["qid", "fused_rank", "semantic_rank", "co_rank", "id_rank", "meta_rank"],
            ascending=[True, True, True, True, True, True],
        )
        .groupby("qid", sort=False)
        .head(CROSS_ENCODER_TOP_K)
        .index
    )
    if len(score_index) == 0:
        return df

    pairs = [
        (query_texts[int(qid)], occ_text_map[str(occ)])
        for qid, occ in zip(df.loc[score_index, "qid"].tolist(), df.loc[score_index, "candidate_occ"].tolist())
    ]
    print(
        f"Cross-encoder scoring {len(pairs):,} of {len(df):,} candidate pairs "
        f"(top {CROSS_ENCODER_TOP_K} per query).",
        flush=True,
    )
    scores = get_cross_encoder().predict(pairs, batch_size=256, show_progress_bar=True)
    df.loc[score_index, "cross_score"] = scores.astype(np.float32)
    return df


def rerank_topk_with_cross_encoder(
    df: pd.DataFrame,
    query_texts: list[str],
    occ_text_map: dict[str, str],
    base_score_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df["ce_final_raw"] = np.float32(0.0)
    df["ce_final_norm"] = np.float32(0.0)
    df["ce_blend_score"] = df[base_score_col].astype(np.float32)
    if FINAL_CROSS_ENCODER_TOP_K <= 0 or df.empty:
        print("Skipping final cross-encoder reranking.", flush=True)
        return df

    score_index = (
        df.sort_values(["qid", base_score_col], ascending=[True, False])
        .groupby("qid", sort=False)
        .head(FINAL_CROSS_ENCODER_TOP_K)
        .index
    )
    if len(score_index) == 0:
        return df

    pairs = [
        (query_texts[int(qid)], occ_text_map[str(occ)])
        for qid, occ in zip(df.loc[score_index, "qid"].tolist(), df.loc[score_index, "candidate_occ"].tolist())
    ]
    print(
        f"Final cross-encoder reranking {len(pairs):,} pairs "
        f"(top {FINAL_CROSS_ENCODER_TOP_K} per query).",
        flush=True,
    )
    scores = get_cross_encoder().predict(pairs, batch_size=256, show_progress_bar=True).astype(np.float32)
    df.loc[score_index, "ce_final_raw"] = scores

    subset = df.loc[score_index, ["qid", "ce_final_raw"]].copy()
    group_min = subset.groupby("qid", sort=False)["ce_final_raw"].transform("min")
    group_max = subset.groupby("qid", sort=False)["ce_final_raw"].transform("max")
    denom = (group_max - group_min).replace(0, np.nan)
    subset["ce_final_norm"] = ((subset["ce_final_raw"] - group_min) / denom).fillna(0.5).astype(np.float32)
    df.loc[score_index, "ce_final_norm"] = subset["ce_final_norm"].to_numpy()
    df["ce_blend_score"] = df[base_score_col] + FINAL_CROSS_ENCODER_WEIGHT * df["ce_final_norm"]
    return df


def encode_features(train_df: pd.DataFrame, pred_df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    pred_df = pred_df.copy()
    for col in cat_cols:
        combined = pd.concat(
            [
                train_df[col].fillna("unknown").astype(str),
                pred_df[col].fillna("unknown").astype(str),
            ],
            ignore_index=True,
        )
        cats = pd.Index(combined.unique())
        mapping = {v: i for i, v in enumerate(cats)}
        train_df[col] = train_df[col].fillna("unknown").astype(str).map(mapping).astype(np.int32)
        pred_df[col] = pred_df[col].fillna("unknown").astype(str).map(mapping).astype(np.int32)
    return train_df, pred_df


def prepare_ranker_data(
    train_rank_df: pd.DataFrame,
    pred_rank_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    train_enc, pred_enc = encode_features(train_rank_df, pred_rank_df, cat_cols)
    feature_cols = [c for c in train_enc.columns if c not in ["qid", "label"]]
    cat_feature_idx = [feature_cols.index(col) for col in cat_cols if col in feature_cols]
    return train_enc, pred_enc, feature_cols, cat_feature_idx


def fit_xgb_bagged_ranker(
    train_enc: pd.DataFrame,
    pred_enc: pd.DataFrame,
    group_train: np.ndarray,
    feature_cols: list[str],
    seeds: list[int],
) -> np.ndarray:
    preds = []
    for seed in seeds:
        print(f"Training XGBoost ranker seed {seed}...", flush=True)
        ranker = xgb.XGBRanker(
            objective="rank:map",
            eval_metric="map@5",
            tree_method="hist",
            device="cuda" if DEVICE == "cuda" else "cpu",
            n_estimators=500,
            learning_rate=0.04,
            max_depth=8,
            min_child_weight=1.0,
            subsample=0.82,
            colsample_bytree=0.78,
            reg_alpha=0.0,
            reg_lambda=2.0,
            gamma=0.0,
            random_state=seed,
        )
        ranker.fit(train_enc[feature_cols], train_enc["label"], group=group_train, verbose=False)
        preds.append(ranker.predict(pred_enc[feature_cols]).astype(np.float32))
    return np.mean(np.vstack(preds), axis=0)


def fit_catboost_ranker(
    train_enc: pd.DataFrame,
    pred_enc: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_idx: list[int],
    seed: int,
) -> np.ndarray:
    print(f"Training CatBoost ranker seed {seed}...", flush=True)
    ranker = CatBoostRanker(
        loss_function="YetiRankPairwise:mode=MAP",
        iterations=600,
        learning_rate=0.05,
        depth=8,
        random_seed=seed,
        l2_leaf_reg=6.0,
        min_data_in_leaf=20,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        verbose=False,
    )
    ranker.fit(
        train_enc[feature_cols],
        train_enc["label"],
        group_id=train_enc["qid"],
        cat_features=cat_feature_idx,
        verbose=False,
    )
    return ranker.predict(pred_enc[feature_cols]).astype(np.float32)


def fit_lgbm_ranker(
    train_enc: pd.DataFrame,
    pred_enc: pd.DataFrame,
    group_train: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    print("Training LightGBM ranker...", flush=True)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="map",
        eval_at=[5],
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
    )
    ranker.fit(train_enc[feature_cols], train_enc["label"], group=group_train)
    return ranker.predict(pred_enc[feature_cols]).astype(np.float32)


def fit_second_ranker(
    train_enc: pd.DataFrame,
    pred_enc: pd.DataFrame,
    group_train: np.ndarray,
    feature_cols: list[str],
    cat_feature_idx: list[int],
) -> tuple[np.ndarray, str]:
    if SECOND_RANKER == "catboost":
        return fit_catboost_ranker(train_enc, pred_enc, feature_cols, cat_feature_idx, CATBOOST_SEED), "catboost"
    return fit_lgbm_ranker(train_enc, pred_enc, group_train, feature_cols), "lgbm"


def add_groupwise_normalized_score(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    group_min = df.groupby("qid", sort=False)[score_col].transform("min")
    group_max = df.groupby("qid", sort=False)[score_col].transform("max")
    denom = (group_max - group_min).replace(0, np.nan)
    df[out_col] = ((df[score_col] - group_min) / denom).fillna(0.5).astype(np.float32)
    return df



def train_and_build_features(
    fit: pd.DataFrame,
    pred_df: pd.DataFrame,
    skill_meta: pd.DataFrame,
    occ_meta: pd.DataFrame,
    label_available: bool,
    same_source: bool = False,
    inject_actual_labels: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    query_fit = build_query_frame(fit, skill_meta)
    query_pred = build_query_frame(pred_df, skill_meta)

    x_fit_id, x_pred_id, x_fit_meta, x_pred_meta, sim_id, sim_meta, sim_set, sim_sig = build_sparse_knn(query_fit, query_pred)
    x_fit_combined = sparse.hstack([x_fit_id, x_fit_meta]).tocsr()
    x_pred_combined = sparse.hstack([x_pred_id, x_pred_meta]).tocsr()

    artifacts = build_cooccurrence_artifacts(fit, skill_meta)
    extra_artifacts = build_extra_retrieval_artifacts(fit, skill_meta, occ_meta)
    bm25_skill_scores, als_skill_scores, _, _ = precompute_implicit_scores(fit, occ_meta, artifacts["skill_weight"])

    dense_scores, dense_occ_ids, query_ce_texts, occ_ce_map, _ = build_dense_semantic_scores(query_fit, query_pred, occ_meta)
    (
        occ_profile_id_scores,
        occ_profile_meta_scores,
        occ_profile_occ_ids,
        group_profile_scores,
        group_profile_names,
        career_profile_scores,
        career_profile_names,
    ) = build_profile_similarity_scores(query_pred, extra_artifacts, occ_meta)

    model2, mlb2, model4, mlb4 = fit_prefix_models(x_fit_combined, fit)
    prefix2_scores = model2.decision_function(x_pred_combined)
    prefix4_scores = model4.decision_function(x_pred_combined)
    if prefix2_scores.ndim == 1:
        prefix2_scores = prefix2_scores[:, None]
    if prefix4_scores.ndim == 1:
        prefix4_scores = prefix4_scores[:, None]

    candidate_df, group_sizes = build_candidate_frame(
        fit=fit,
        pred_df=pred_df,
        query_pred=query_pred,
        occ_meta=occ_meta,
        artifacts=artifacts,
        extra_artifacts=extra_artifacts,
        sim_id=sim_id,
        sim_meta=sim_meta,
        sim_set=sim_set,
        sim_sig=sim_sig,
        bm25_skill_scores=bm25_skill_scores,
        als_skill_scores=als_skill_scores,
        dense_scores=dense_scores,
        dense_occ_ids=dense_occ_ids,
        occ_profile_id_scores=occ_profile_id_scores,
        occ_profile_meta_scores=occ_profile_meta_scores,
        occ_profile_occ_ids=occ_profile_occ_ids,
        group_profile_scores=group_profile_scores,
        group_profile_names=group_profile_names,
        career_profile_scores=career_profile_scores,
        career_profile_names=career_profile_names,
        prefix2_scores=prefix2_scores,
        prefix2_classes=mlb2.classes_,
        prefix4_scores=prefix4_scores,
        prefix4_classes=mlb4.classes_,
        label_available=label_available,
        same_source=same_source,
        inject_actual_labels=inject_actual_labels,
    )
    candidate_df = add_cross_encoder_scores(candidate_df, query_ce_texts, occ_ce_map)
    return candidate_df, group_sizes, query_ce_texts


def build_oof_train_features(
    train: pd.DataFrame,
    skill_meta: pd.DataFrame,
    occ_meta: pd.DataFrame,
    n_splits: int = OOF_FOLDS,
    inject_actual_labels: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    if n_splits < 2:
        train_rank_df, group_train, _ = train_and_build_features(
            train,
            train,
            skill_meta,
            occ_meta,
            True,
            same_source=True,
            inject_actual_labels=inject_actual_labels,
        )
        return train_rank_df, group_train

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_frames: list[pd.DataFrame] = []
    fold_groups: list[np.ndarray] = []
    qid_offset = 0
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(train), start=1):
        print(f"Building OOF fold {fold_idx}/{n_splits}...", flush=True)
        fit_fold = train.iloc[fit_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        fold_df, fold_group, _ = train_and_build_features(
            fit_fold,
            val_fold,
            skill_meta,
            occ_meta,
            True,
            same_source=False,
            inject_actual_labels=inject_actual_labels,
        )
        fold_df = fold_df.copy()
        fold_df["qid"] = fold_df["qid"].astype(np.int32) + qid_offset
        qid_offset = int(fold_df["qid"].max()) + 1
        fold_frames.append(fold_df)
        fold_groups.append(fold_group)

    train_rank_df = pd.concat(fold_frames, ignore_index=True)
    group_train = np.concatenate(fold_groups).astype(np.int32)
    return train_rank_df, group_train


def score_predictions(ranked_df: pd.DataFrame, score_col: str, tail_bonus: float = 0.0) -> list[list[str]]:
    outputs: list[list[str]] = []
    for _, group_df in ranked_df.groupby("qid", sort=False):
        df = group_df.copy()
        if tail_bonus:
            df["final_score"] = df[score_col] + tail_bonus * df["ultra_rare_flag"] * df["semantic_score"]
        else:
            df["final_score"] = df[score_col]
        outputs.append(df.sort_values("final_score", ascending=False)["candidate_occ"].head(5).tolist())
    return outputs


def evaluate_map5(ranked_df: pd.DataFrame, score_col: str) -> float:
    scores = []
    for _, group_df in ranked_df.groupby("qid", sort=False):
        actual = group_df.loc[group_df["label"] == 1, "candidate_occ"].tolist()
        predicted = group_df.sort_values(score_col, ascending=False)["candidate_occ"].head(5).tolist()
        scores.append(apk(actual, predicted, 5))
    return float(np.mean(scores))


def build_submission_df(ids: list[str], top5s: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": item_id,
                "occ_1": top5[0],
                "occ_2": top5[1],
                "occ_3": top5[2],
                "occ_4": top5[3],
                "occ_5": top5[4],
            }
            for item_id, top5 in zip(ids, top5s)
        ],
        columns=["ID", "occ_1", "occ_2", "occ_3", "occ_4", "occ_5"],
    )


def run_holdout(train: pd.DataFrame, skill_meta: pd.DataFrame, occ_meta: pd.DataFrame) -> None:
    fit, val = train_test_split(train, test_size=0.18, random_state=SEED, shuffle=True)
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_rank_df, group_train = build_oof_train_features(fit, skill_meta, occ_meta, n_splits=min(OOF_FOLDS, 3))
    val_rank_df, _, _ = train_and_build_features(fit, val, skill_meta, occ_meta, True, same_source=False)
    train_enc, val_enc, feature_cols, cat_feature_idx = prepare_ranker_data(train_rank_df, val_rank_df, CAT_COLS)

    val_rank_df = val_rank_df.copy()
    val_rank_df["xgb_bag_score"] = fit_xgb_bagged_ranker(train_enc, val_enc, group_train, feature_cols, XGB_BAG_SEEDS)
    val_rank_df["alt_score"], alt_name = fit_second_ranker(train_enc, val_enc, group_train, feature_cols, cat_feature_idx)
    val_rank_df = add_groupwise_normalized_score(val_rank_df, "xgb_bag_score", "xgb_bag_norm")
    val_rank_df = add_groupwise_normalized_score(val_rank_df, "alt_score", "alt_norm")
    val_rank_df["blend_score"] = 0.78 * val_rank_df["xgb_bag_norm"] + 0.22 * val_rank_df["alt_norm"]

    print(f"holdout MAP@5 xgb_bag: {evaluate_map5(val_rank_df, 'xgb_bag_score'):.6f}", flush=True)
    print(f"holdout MAP@5 {alt_name}: {evaluate_map5(val_rank_df, 'alt_score'):.6f}", flush=True)
    print(f"holdout MAP@5 blend: {evaluate_map5(val_rank_df, 'blend_score'):.6f}", flush=True)


def main() -> None:
    train, test, skill_meta, occ_meta = load_data()
    if os.environ.get("RUN_HOLDOUT", "0") == "1":
        run_holdout(train, skill_meta, occ_meta)

    train_rank_df, group_train = build_oof_train_features(train, skill_meta, occ_meta, n_splits=OOF_FOLDS)
    test_rank_df, _, test_query_ce_texts = train_and_build_features(train, test, skill_meta, occ_meta, False, same_source=False)

    train_enc, test_enc, feature_cols, cat_feature_idx = prepare_ranker_data(train_rank_df, test_rank_df, CAT_COLS)
    test_rank_df = test_rank_df.copy()
    test_rank_df["xgb_bag_score"] = fit_xgb_bagged_ranker(train_enc, test_enc, group_train, feature_cols, XGB_BAG_SEEDS)
    test_rank_df["alt_score"], alt_name = fit_second_ranker(train_enc, test_enc, group_train, feature_cols, cat_feature_idx)
    test_rank_df = add_groupwise_normalized_score(test_rank_df, "xgb_bag_score", "xgb_bag_norm")
    test_rank_df = add_groupwise_normalized_score(test_rank_df, "alt_score", "alt_norm")
    test_rank_df["blend_score"] = 0.78 * test_rank_df["xgb_bag_norm"] + 0.22 * test_rank_df["alt_norm"]

    top5_xgb = score_predictions(test_rank_df, "xgb_bag_score", tail_bonus=0.0)
    top5_alt = score_predictions(test_rank_df, "alt_score", tail_bonus=0.0)
    top5_blend = score_predictions(test_rank_df, "blend_score", tail_bonus=0.0)

    xgb_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_xgb)
    alt_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_alt)
    blend_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_blend)

    xgb_path = DATA_DIR / f"{OUTPUT_PREFIX}_xgb_bag.csv"
    cat_path = DATA_DIR / f"{OUTPUT_PREFIX}_{alt_name}.csv"
    blend_path = DATA_DIR / f"{OUTPUT_PREFIX}_blend.csv"
    xgb_submission.to_csv(xgb_path, index=False)
    alt_submission.to_csv(cat_path, index=False)
    blend_submission.to_csv(blend_path, index=False)
    print(f"saved {xgb_path.resolve()}", flush=True)
    print(f"saved {cat_path.resolve()}", flush=True)
    print(f"saved {blend_path.resolve()}", flush=True)

    if FINAL_CROSS_ENCODER_TOP_K > 0:
        _, _, test_occ_ce_map = build_occ_text_payloads(occ_meta)
        test_reranked_df = rerank_topk_with_cross_encoder(test_rank_df, test_query_ce_texts, test_occ_ce_map, "blend_score")
        top5_ce = score_predictions(test_reranked_df, "ce_blend_score", tail_bonus=0.0)
        ce_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_ce)
        ce_path = DATA_DIR / f"{OUTPUT_PREFIX}_blend_ce.csv"
        ce_submission.to_csv(ce_path, index=False)
        print(f"saved {ce_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
