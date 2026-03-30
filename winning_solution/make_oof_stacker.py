from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("USE_FAST_DENSE", "1")
os.environ.setdefault("SECOND_RANKER", "lgbm")
os.environ.setdefault("FINAL_CROSS_ENCODER_TOP_K", "0")
os.environ.setdefault("OUTPUT_PREFIX", "submission_general_recall")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split

from make_map_ranker_stack import (
    CAT_COLS,
    DATA_DIR,
    OOF_FOLDS,
    SEED,
    XGB_BAG_SEEDS,
    DEVICE,
    add_groupwise_normalized_score,
    build_oof_train_features,
    build_submission_df,
    evaluate_map5,
    fit_second_ranker,
    fit_xgb_bagged_ranker,
    load_data,
    prepare_ranker_data,
    score_predictions,
    train_and_build_features,
)


np.random.seed(SEED)
torch.manual_seed(SEED)

STACKER_FOLDS = int(os.environ.get("STACKER_FOLDS", "3"))
STACKER_OUTPUT_PREFIX = os.environ.get("STACKER_OUTPUT_PREFIX", "submission_general_recall_stacker")


def build_group_array(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("qid", sort=False).size().to_numpy(dtype=np.int32)


def augment_stacker_features(df: pd.DataFrame, xgb_col: str, alt_col: str) -> pd.DataFrame:
    df = df.copy()
    df = add_groupwise_normalized_score(df, xgb_col, "xgb_base_norm")
    df = add_groupwise_normalized_score(df, alt_col, "alt_base_norm")
    df["base_mean_norm"] = 0.5 * (df["xgb_base_norm"] + df["alt_base_norm"])
    df["base_diff_norm"] = df["xgb_base_norm"] - df["alt_base_norm"]
    df["base_abs_diff"] = df["base_diff_norm"].abs()
    df["base_prod_norm"] = df["xgb_base_norm"] * df["alt_base_norm"]
    df["base_max_norm"] = df[["xgb_base_norm", "alt_base_norm"]].max(axis=1)
    df["base_min_norm"] = df[["xgb_base_norm", "alt_base_norm"]].min(axis=1)
    df["base_support_sum"] = (
        df["id_support_count"]
        + df["meta_support_count"]
        + df["set_support_count"]
        + df["sig_support_count"]
        + df["bm25_support_count"]
        + df["als_support_count"]
        + df["graph_support_count"]
        + df["walk_support_count"]
    ).astype(np.float32)
    df["retrieval_vote_ratio"] = (df["retrieval_vote_count"] / 15.0).astype(np.float32)

    for rank_col in [
        "fused_rank",
        "semantic_rank",
        "id_rank",
        "meta_rank",
        "set_rank",
        "sig_rank",
        "bm25_rank",
        "als_rank",
        "graph_rank",
        "walk_rank",
        "profile_id_rank",
        "profile_meta_rank",
    ]:
        if rank_col in df.columns:
            df[f"{rank_col}_inv"] = (1.0 / (1.0 + df[rank_col])).astype(np.float32)

    df["tail_semantic_bonus"] = (df["ultra_rare_flag"] * df["semantic_score"]).astype(np.float32)
    df["tail_profile_bonus"] = (df["rare_flag"] * (df["profile_id_score"] + df["profile_meta_score"])).astype(np.float32)
    return df


def fit_meta_ranker(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    group_train: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    ranker = xgb.XGBRanker(
        objective="rank:map",
        eval_metric="map@5",
        tree_method="hist",
        device="cuda" if DEVICE == "cuda" else "cpu",
        n_estimators=320,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=4.0,
        subsample=0.84,
        colsample_bytree=0.78,
        reg_alpha=0.25,
        reg_lambda=5.0,
        gamma=0.0,
        random_state=seed,
    )
    ranker.fit(train_df[feature_cols], train_df["label"], group=group_train, verbose=False)
    return ranker.predict(pred_df[feature_cols]).astype(np.float32)


def build_meta_feature_cols(df: pd.DataFrame) -> list[str]:
    preferred = [
        "xgb_bag_score",
        "alt_score",
        "xgb_base_norm",
        "alt_base_norm",
        "base_mean_norm",
        "base_diff_norm",
        "base_abs_diff",
        "base_prod_norm",
        "base_max_norm",
        "base_min_norm",
        "fused_rank",
        "fused_rank_inv",
        "semantic_score",
        "semantic_rank",
        "semantic_rank_inv",
        "id_score",
        "id_rank",
        "id_rank_inv",
        "meta_score",
        "meta_rank",
        "meta_rank_inv",
        "set_score",
        "set_rank",
        "set_rank_inv",
        "sig_score",
        "sig_rank",
        "sig_rank_inv",
        "bm25_score",
        "bm25_rank",
        "bm25_rank_inv",
        "als_score",
        "als_rank",
        "als_rank_inv",
        "co_score",
        "graph_score",
        "graph_rank",
        "graph_rank_inv",
        "walk_score",
        "walk_rank",
        "walk_rank_inv",
        "profile_id_score",
        "profile_id_rank",
        "profile_id_rank_inv",
        "profile_meta_score",
        "profile_meta_rank",
        "profile_meta_rank_inv",
        "prefix2_model_score",
        "prefix4_model_score",
        "group_retrieval_score",
        "career_retrieval_score",
        "group_profile_score",
        "career_profile_score",
        "retrieval_vote_count",
        "retrieval_vote_ratio",
        "base_support_sum",
        "co_skill_hits",
        "co_pair_hits",
        "query_unique_skill_categories",
        "query_unique_skill_subcategories",
        "query_unique_skill_types",
        "query_software_count",
        "occ_freq",
        "occ_log_freq",
        "occ_share_in_prefix4",
        "occ_share_in_prefix2",
        "rare_flag",
        "ultra_rare_flag",
        "tail_semantic_bonus",
        "tail_profile_bonus",
        "min_training_months",
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
    return [col for col in preferred if col in df.columns]


def generate_base_oof_predictions(
    train_enc: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_idx: list[int],
    n_splits: int = STACKER_FOLDS,
) -> tuple[pd.DataFrame, str]:
    train_enc = train_enc.copy()
    train_enc["xgb_bag_score"] = np.float32(0.0)
    train_enc["alt_score"] = np.float32(0.0)

    unique_qids = train_enc["qid"].drop_duplicates().to_numpy()
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + 17)
    alt_name = "alt"

    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(unique_qids), start=1):
        fit_qids = set(unique_qids[fit_idx].tolist())
        val_qids = set(unique_qids[val_idx].tolist())
        fit_mask = train_enc["qid"].isin(fit_qids).to_numpy()
        val_mask = train_enc["qid"].isin(val_qids).to_numpy()
        fit_df = train_enc.loc[fit_mask].copy()
        val_df = train_enc.loc[val_mask].copy()
        fold_group = build_group_array(fit_df)

        print(f"Generating base OOF predictions fold {fold_idx}/{n_splits}...", flush=True)
        xgb_pred = fit_xgb_bagged_ranker(fit_df, val_df, fold_group, feature_cols, XGB_BAG_SEEDS)
        alt_pred, alt_name = fit_second_ranker(fit_df, val_df, fold_group, feature_cols, cat_feature_idx)
        train_enc.loc[val_mask, "xgb_bag_score"] = xgb_pred
        train_enc.loc[val_mask, "alt_score"] = alt_pred

    return train_enc, alt_name


def fit_full_base_predictions(
    train_enc: pd.DataFrame,
    pred_enc: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_idx: list[int],
    group_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    xgb_pred = fit_xgb_bagged_ranker(train_enc, pred_enc, group_train, feature_cols, XGB_BAG_SEEDS)
    alt_pred, alt_name = fit_second_ranker(train_enc, pred_enc, group_train, feature_cols, cat_feature_idx)
    return xgb_pred, alt_pred, alt_name


def run_stacker_holdout(train: pd.DataFrame, skill_meta: pd.DataFrame, occ_meta: pd.DataFrame) -> dict[str, float]:
    fit, val = train_test_split(train, test_size=0.18, random_state=SEED, shuffle=True)
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_rank_df, group_train = build_oof_train_features(fit, skill_meta, occ_meta, n_splits=min(OOF_FOLDS, STACKER_FOLDS))
    val_rank_df, _, _ = train_and_build_features(fit, val, skill_meta, occ_meta, True, same_source=False)
    train_enc, val_enc, feature_cols, cat_feature_idx = prepare_ranker_data(train_rank_df, val_rank_df, CAT_COLS)

    train_oof, alt_name = generate_base_oof_predictions(train_enc, feature_cols, cat_feature_idx, n_splits=min(STACKER_FOLDS, 3))
    val_xgb, val_alt, alt_name = fit_full_base_predictions(train_enc, val_enc, feature_cols, cat_feature_idx, group_train)

    train_stack = augment_stacker_features(train_oof, "xgb_bag_score", "alt_score")
    val_stack_features = val_enc.copy()
    val_stack_features["xgb_bag_score"] = val_xgb
    val_stack_features["alt_score"] = val_alt
    val_stack = augment_stacker_features(val_stack_features, "xgb_bag_score", "alt_score")
    stack_feature_cols = build_meta_feature_cols(train_stack)

    print("Training OOF stacker on holdout split...", flush=True)
    val_stack["stack_score"] = fit_meta_ranker(train_stack, val_stack, stack_feature_cols, build_group_array(train_stack))
    val_rank_df = val_rank_df.copy()
    val_rank_df["xgb_bag_score"] = val_xgb
    val_rank_df["alt_score"] = val_alt
    val_rank_df["stack_score"] = val_stack["stack_score"].to_numpy()
    val_rank_df = add_groupwise_normalized_score(val_rank_df, "xgb_bag_score", "xgb_base_norm")
    val_rank_df = add_groupwise_normalized_score(val_rank_df, "alt_score", "alt_base_norm")
    val_rank_df = add_groupwise_normalized_score(val_rank_df, "stack_score", "stack_norm")
    val_rank_df["stack_blend_score"] = (
        0.72 * val_rank_df["stack_norm"]
        + 0.20 * val_rank_df["xgb_base_norm"]
        + 0.08 * val_rank_df["alt_base_norm"]
    )

    metrics = {
        "xgb_bag": evaluate_map5(val_rank_df, "xgb_bag_score"),
        alt_name: evaluate_map5(val_rank_df, "alt_score"),
        "stacker": evaluate_map5(val_rank_df, "stack_score"),
        "stacker_blend": evaluate_map5(val_rank_df, "stack_blend_score"),
    }
    return metrics


def main() -> None:
    train, test, skill_meta, occ_meta = load_data()

    print("Building recall-expanded candidate tables...", flush=True)
    train_rank_df, group_train = build_oof_train_features(train, skill_meta, occ_meta, n_splits=OOF_FOLDS)
    test_rank_df, _, _ = train_and_build_features(train, test, skill_meta, occ_meta, False, same_source=False)
    train_enc, test_enc, feature_cols, cat_feature_idx = prepare_ranker_data(train_rank_df, test_rank_df, CAT_COLS)

    print("Generating base OOF predictions for stacker training...", flush=True)
    train_oof, alt_name = generate_base_oof_predictions(train_enc, feature_cols, cat_feature_idx, n_splits=STACKER_FOLDS)

    print("Scoring test candidates with full base models...", flush=True)
    test_xgb, test_alt, alt_name = fit_full_base_predictions(train_enc, test_enc, feature_cols, cat_feature_idx, group_train)
    train_stack = augment_stacker_features(train_oof, "xgb_bag_score", "alt_score")
    test_stack_features = test_enc.copy()
    test_stack_features["xgb_bag_score"] = test_xgb
    test_stack_features["alt_score"] = test_alt
    test_stack = augment_stacker_features(test_stack_features, "xgb_bag_score", "alt_score")
    stack_feature_cols = build_meta_feature_cols(train_stack)

    print("Training final OOF stacker...", flush=True)
    test_stack["stack_score"] = fit_meta_ranker(train_stack, test_stack, stack_feature_cols, build_group_array(train_stack))
    test_rank_df = test_rank_df.copy()
    test_rank_df["xgb_bag_score"] = test_xgb
    test_rank_df["alt_score"] = test_alt
    test_rank_df["stack_score"] = test_stack["stack_score"].to_numpy()
    test_rank_df = add_groupwise_normalized_score(test_rank_df, "xgb_bag_score", "xgb_base_norm")
    test_rank_df = add_groupwise_normalized_score(test_rank_df, "alt_score", "alt_base_norm")
    test_rank_df = add_groupwise_normalized_score(test_rank_df, "stack_score", "stack_norm")
    test_rank_df["stack_blend_score"] = (
        0.72 * test_rank_df["stack_norm"]
        + 0.20 * test_rank_df["xgb_base_norm"]
        + 0.08 * test_rank_df["alt_base_norm"]
    )

    top5_stacker = score_predictions(test_rank_df, "stack_score", tail_bonus=0.0)
    top5_stacker_blend = score_predictions(test_rank_df, "stack_blend_score", tail_bonus=0.0)

    stacker_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_stacker)
    stacker_blend_submission = build_submission_df(test["ID"].astype(str).tolist(), top5_stacker_blend)

    stacker_path = DATA_DIR / f"{STACKER_OUTPUT_PREFIX}.csv"
    stacker_blend_path = DATA_DIR / f"{STACKER_OUTPUT_PREFIX}_blend.csv"
    stacker_submission.to_csv(stacker_path, index=False)
    stacker_blend_submission.to_csv(stacker_blend_path, index=False)
    print(f"saved {stacker_path.resolve()}", flush=True)
    print(f"saved {stacker_blend_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
