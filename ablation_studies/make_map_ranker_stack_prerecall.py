from __future__ import annotations

import os


# Legacy runner for the older top-performing "general" family before recall expansion.
os.environ.setdefault("RECALL_EXPANSION", "0")
os.environ.setdefault("OUTPUT_PREFIX", "submission_general")
os.environ.setdefault("USE_FAST_DENSE", "1")
os.environ.setdefault("SECOND_RANKER", "lgbm")
os.environ.setdefault("FINAL_CROSS_ENCODER_TOP_K", "0")
os.environ.setdefault("OOF_FOLDS", "3")
os.environ.setdefault("XGB_BAG_SEEDS", "42,73,121")

from make_map_ranker_stack import main


if __name__ == "__main__":
    main()
