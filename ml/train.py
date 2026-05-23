"""
ml/train.py — Train XGBoost p_model for BTC 15m prediction markets.

Reads prices zip directly (no extraction needed).
Features are computed from the first 5 minutes of each market only —
no look-ahead. The output model maps real-time order book state → P(Up).

Usage (Windows):
    pip install pandas pyarrow xgboost scikit-learn
    python ml/train.py --prices "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip"

Output:
    ml/btc_15m_model.pkl  — model artifact (load with pickle)
    ml/feature_cols.txt   — ordered list of feature names (for the bot)
"""

from __future__ import annotations

import argparse
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ── Constants ─────────────────────────────────────────────────────────────────

ENTRY_WINDOW  = 300   # seconds of data to use for features (first 5 min of 15-min market)
SNAPSHOTS     = [0, 30, 60, 120, 180, 300]   # timestamps to snapshot features at
MIN_ROWS      = 310   # skip files shorter than this (incomplete markets)
SETTLED_THRESH = 0.90  # up_bid >= this at end → Up settled; <= 1-this → Down settled


# ── Feature engineering ───────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, slug: str, date_str: str) -> dict | None:
    """
    Given a 900-row prices DataFrame for one market, return a feature dict.
    Returns None if the market is too short or outcome is ambiguous.
    """
    if len(df) < MIN_ROWS:
        return None

    # ── Derive label from end of market ──────────────────────────────────────
    # Up settled → up_bid near 1.0; Down settled → up_bid near 0.0
    final_up_bid = df["up_bid"].dropna().iloc[-1] if df["up_bid"].dropna().shape[0] else None
    if final_up_bid is None:
        return None
    if final_up_bid >= SETTLED_THRESH:
        label = 1
    elif final_up_bid <= (1.0 - SETTLED_THRESH):
        label = 0
    else:
        return None   # market didn't settle cleanly — skip

    window = df.iloc[:ENTRY_WINDOW]

    feat: dict = {"slug": slug, "date": date_str, "label": label}

    # ── Snapshot features at key timestamps ──────────────────────────────────
    for t in SNAPSHOTS:
        if t >= len(window):
            continue
        row = window.iloc[t]
        p = f"t{t}"
        feat[f"{p}_up_micro"]   = row.get("up_microprice")
        feat[f"{p}_up_obi"]     = row.get("up_ob_imbalance")
        feat[f"{p}_up_bid"]     = row.get("up_bid")
        feat[f"{p}_up_ask"]     = row.get("up_ask")
        feat[f"{p}_depth_ratio"] = (
            row.get("up_total_depth", 0) / row.get("down_total_depth", 1)
            if row.get("down_total_depth", 0) > 0 else 1.0
        )

    # ── Momentum features ─────────────────────────────────────────────────────
    micro = window["up_microprice"].ffill()
    obi   = window["up_ob_imbalance"].ffill()

    if len(micro) >= 60:
        feat["mom_60"]  = float(micro.iloc[59]  - micro.iloc[0])
        feat["obi_60"]  = float(obi.iloc[:60].mean())
    if len(micro) >= 120:
        feat["mom_120"] = float(micro.iloc[119] - micro.iloc[0])
        feat["obi_120"] = float(obi.iloc[:120].mean())
    if len(micro) >= 300:
        feat["mom_300"] = float(micro.iloc[299] - micro.iloc[0])
        feat["obi_300"] = float(obi.iloc[:300].mean())

    # ── Rolling statistics over window ────────────────────────────────────────
    feat["micro_mean"]    = float(micro.mean())
    feat["micro_std"]     = float(micro.std())
    feat["micro_max"]     = float(micro.max())
    feat["micro_min"]     = float(micro.min())
    feat["micro_range"]   = float(micro.max() - micro.min())
    feat["obi_mean"]      = float(obi.mean())
    feat["obi_std"]       = float(obi.std())
    feat["obi_pos_frac"]  = float((obi > 0).mean())   # fraction of seconds with buy pressure

    # ── Depth ratio over window ───────────────────────────────────────────────
    up_depth   = window["up_total_depth"].ffill()
    down_depth = window["down_total_depth"].ffill()
    safe_down  = down_depth.replace(0, np.nan)
    feat["depth_ratio_mean"] = float((up_depth / safe_down).mean())

    # ── Opening spread ────────────────────────────────────────────────────────
    opening = df.iloc[0]
    bid0 = opening.get("up_bid")
    ask0 = opening.get("up_ask")
    if bid0 is not None and ask0 is not None and not (np.isnan(bid0) or np.isnan(ask0)):
        feat["opening_spread"] = float(ask0 - bid0)
    else:
        feat["opening_spread"] = 0.02   # default 1-cent spread

    return feat


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_zip(zip_path: str) -> pd.DataFrame:
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        parquet_names = sorted(n for n in zf.namelist() if n.endswith(".parquet"))
        print(f"  {len(parquet_names)} parquet files found in zip")

        for name in parquet_names:
            date_str = None
            for part in name.replace("\\", "/").split("/"):
                if part.startswith("dt="):
                    date_str = part[3:]
                    break

            try:
                with zf.open(name) as f:
                    df = pd.read_parquet(io.BytesIO(f.read()))
            except Exception as e:
                print(f"  SKIP {name}: {e}")
                continue

            slug = df["slug"].iloc[0] if "slug" in df.columns and len(df) > 0 else name
            feat = extract_features(df, slug, date_str)
            if feat is not None:
                records.append(feat)

    result = pd.DataFrame(records)
    print(f"  {len(result)} usable markets loaded")
    return result


# ── Training ──────────────────────────────────────────────────────────────────

def train(prices_zip: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading prices from: {prices_zip}")
    df = load_from_zip(prices_zip)

    label_dist = df["label"].value_counts().to_dict()
    print(f"Label distribution: Up={label_dist.get(1,0)}  Down={label_dist.get(0,0)}")

    if df["label"].nunique() < 2:
        raise RuntimeError("Only one class in data — cannot train classifier")

    # ── Feature columns ───────────────────────────────────────────────────────
    meta_cols    = {"slug", "date", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df[feature_cols] = df[feature_cols].fillna(0.0)

    # ── Time-based split (never random — prevents look-ahead) ─────────────────
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].dropna().unique())

    if len(dates) < 3:
        raise RuntimeError(f"Need at least 3 distinct dates, got {len(dates)}")

    train_dates = dates[:-2]
    val_date    = dates[-2]
    test_date   = dates[-1]

    train_df = df[df["date"].isin(train_dates)]
    val_df   = df[df["date"] == val_date]
    test_df  = df[df["date"] == test_date]

    print(f"\nTrain: {len(train_df)} markets  ({train_dates[0]} → {train_dates[-1]})")
    print(f"Val:   {len(val_df)} markets  ({val_date})")
    print(f"Test:  {len(test_df)} markets  ({test_date})")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        eval_metric="logloss",
        early_stopping_rounds=25,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print()
    for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        if len(y) == 0:
            continue
        proba = model.predict_proba(X)[:, 1]
        acc   = accuracy_score(y, (proba >= 0.5).astype(int))
        ll    = log_loss(y, proba)
        auc   = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
        print(f"{split_name:5s}: accuracy={acc:.3f}  log_loss={ll:.4f}  AUC={auc:.3f}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 10 features:")
    for feat_name, imp in importance[:10]:
        print(f"  {feat_name:35s}  {imp:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_path   = out / "btc_15m_model.pkl"
    feature_path = out / "feature_cols.txt"

    artifact = {
        "model":        model,
        "feature_cols": feature_cols,
        "entry_window": ENTRY_WINDOW,
        "snapshots":    SNAPSHOTS,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    feature_path.write_text("\n".join(feature_cols))

    print(f"\nSaved: {model_path}")
    print(f"Saved: {feature_path}")
    print("\nDone. Run inference with: from ml.predictor import MLPredictor")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BTC 15m p_model")
    parser.add_argument(
        "--prices",
        required=True,
        help='Path to prices zip, e.g. "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip"',
    )
    parser.add_argument(
        "--output",
        default="ml",
        help="Directory to write model artifacts (default: ml/)",
    )
    args = parser.parse_args()
    train(args.prices, args.output)
