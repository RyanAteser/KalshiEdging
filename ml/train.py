"""
ml/train.py — Train XGBoost p_model for BTC 15m prediction markets.

Reads both prices + orderbook zips directly (no extraction needed).
Processes one zip pair at a time to keep peak RAM low (~1 zip in memory).
Features are computed from the first 5 minutes of each market only —
no look-ahead. The output model maps real-time order book state → P(Up).

Usage (Windows):
    pip install pandas pyarrow xgboost scikit-learn

    # Single pair
    python ml/train.py ^
        --prices    "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
        --orderbook "E:\\orderbook_btc_15m_2026-04-20_2026-04-27.zip"

    # All four pairs (RAM-safe — processed one at a time)
    python ml/train.py ^
        --prices    "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                    "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
                    "E:\\prices_btc_15m_2026-05-06_2026-05-12.zip" ^
                    "E:\\prices_btc_15m_2026-05-13_2026-05-18.zip" ^
        --orderbook "E:\\orderbook_btc_15m_2026-04-20_2026-04-27.zip" ^
                    "E:\\orderbook_btc_15m_2026-04-28_2026-05-05.zip" ^
                    "E:\\orderbook_btc_15m_2026-05-06_2026-05-12.zip" ^
                    "E:\\orderbook_btc_15m_2026-05-13_2026-05-18.zip"

Output:
    ml/btc_15m_model.pkl  — model artifact (load with pickle)
    ml/feature_cols.txt   — ordered list of feature names (for the bot)
"""

from __future__ import annotations

import argparse
import gc
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ── Constants ─────────────────────────────────────────────────────────────────

ENTRY_WINDOW   = 300    # seconds of features to use (first 5 min of 15-min market)
SNAPSHOTS      = [0, 30, 60, 120, 180, 300]
MIN_ROWS       = 310    # skip files shorter than this
SETTLED_THRESH = 0.90   # up_bid >= this → Up settled; <= 1-this → Down settled


# ── Loading helpers ───────────────────────────────────────────────────────────

def _index_zip_by_slug(zip_path: str, kind: str) -> dict[str, str]:
    """
    Build a lightweight slug → filename index from a zip.
    Only reads the 'slug' column from each parquet — never keeps full DataFrames
    in memory. Peak RAM = one slug-only DataFrame at a time (~1 KB each).
    """
    index: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        parquet_names = sorted(n for n in zf.namelist() if n.endswith(".parquet"))
        print(f"  [{kind}] {len(parquet_names)} files in zip")

        for name in parquet_names:
            try:
                with zf.open(name) as f:
                    # Read only the slug column — avoids loading all numeric data
                    tiny = pd.read_parquet(io.BytesIO(f.read()), columns=["slug"])
            except Exception:
                continue

            if len(tiny) > 0:
                index[tiny["slug"].iloc[0]] = name

    print(f"  [{kind}] {len(index)} slugs indexed")
    return index


# ── Feature engineering ───────────────────────────────────────────────────────

def _window(df: pd.DataFrame) -> pd.DataFrame:
    """Return the first ENTRY_WINDOW rows, sorted by time."""
    df = df.sort_values("time").reset_index(drop=True)
    return df.iloc[:ENTRY_WINDOW]


def _derive_label(prices_df: pd.DataFrame) -> int | None:
    """Derive Up/Down label from end of prices market. Returns 1/0/None."""
    final = prices_df["up_bid"].dropna()
    if len(final) == 0:
        return None
    v = final.iloc[-1]
    if v >= SETTLED_THRESH:
        return 1
    if v <= (1.0 - SETTLED_THRESH):
        return 0
    return None


def _prices_features(w: pd.DataFrame) -> dict:
    """Features from the prices (UP/DOWN aggregated) file."""
    feat: dict = {}

    micro = w["up_microprice"].ffill()
    obi   = w["up_ob_imbalance"].ffill()

    # Snapshots
    for t in SNAPSHOTS:
        idx = min(t, len(w) - 1)
        p   = f"t{t}"
        row = w.iloc[idx]
        feat[f"{p}_up_micro"]    = float(row.get("up_microprice") or 0)
        feat[f"{p}_up_obi"]      = float(row.get("up_ob_imbalance") or 0)
        feat[f"{p}_up_bid"]      = float(row.get("up_bid") or 0)
        feat[f"{p}_up_ask"]      = float(row.get("up_ask") or 0)
        up_d   = float(row.get("up_total_depth")   or 0)
        down_d = float(row.get("down_total_depth")  or 1)
        feat[f"{p}_depth_ratio"] = up_d / (down_d + 1e-9)

    # Momentum
    n = len(micro)
    for steps in [60, 120, 300]:
        if n >= steps:
            feat[f"mom_{steps}"] = float(micro.iloc[steps - 1] - micro.iloc[0])
            feat[f"obi_{steps}"] = float(obi.iloc[:steps].mean())

    # Rolling stats
    feat.update({
        "micro_mean":   float(micro.mean()),
        "micro_std":    float(micro.std()),
        "micro_max":    float(micro.max()),
        "micro_min":    float(micro.min()),
        "micro_range":  float(micro.max() - micro.min()),
        "obi_mean":     float(obi.mean()),
        "obi_std":      float(obi.std()),
        "obi_pos_frac": float((obi > 0).mean()),
    })

    # Depth ratio over window
    up_d   = w["up_total_depth"].ffill()
    down_d = w["down_total_depth"].ffill()
    feat["depth_ratio_mean"] = float((up_d / (down_d + 1e-9)).mean())

    # Opening spread
    bid0 = w["up_bid"].iloc[0]
    ask0 = w["up_ask"].iloc[0]
    feat["opening_spread"] = float(ask0 - bid0) if pd.notna(bid0) and pd.notna(ask0) else 0.02

    return feat


def _orderbook_features(w: pd.DataFrame) -> dict:
    """
    Features from the full orderbook file.

    Adds signals NOT in the prices file:
      - top_bid_size / top_ask_size   (immediate execution pressure)
      - sum_bid_size / sum_ask_size   (total depth imbalance)
      - n_bids, n_asks                (book width)
      - spread                        (explicit bid-ask spread)
    """
    feat: dict = {}

    top_ratio  = w["top_bid_size"]  / (w["top_ask_size"]  + 1e-9)
    sum_ratio  = w["sum_bid_size"]  / (w["sum_ask_size"]  + 1e-9)
    spread_s   = w["spread"].ffill()
    n_bid_s    = w["n_bids"].astype(float)
    n_ask_s    = w["n_asks"].astype(float)

    # Snapshots
    for t in SNAPSHOTS:
        idx = min(t, len(w) - 1)
        p   = f"t{t}"
        feat[f"{p}_top_ratio"]  = float(top_ratio.iloc[idx])
        feat[f"{p}_sum_ratio"]  = float(sum_ratio.iloc[idx])
        feat[f"{p}_spread"]     = float(spread_s.iloc[idx] if pd.notna(spread_s.iloc[idx]) else 0.02)
        feat[f"{p}_n_bids"]     = float(n_bid_s.iloc[idx])
        feat[f"{p}_n_asks"]     = float(n_ask_s.iloc[idx])

    # Rolling stats on top-of-book ratio
    feat["top_ratio_mean"]   = float(top_ratio.mean())
    feat["top_ratio_std"]    = float(top_ratio.std())
    feat["top_ratio_max"]    = float(top_ratio.max())
    feat["sum_ratio_mean"]   = float(sum_ratio.mean())
    feat["sum_ratio_std"]    = float(sum_ratio.std())
    feat["spread_mean"]      = float(spread_s.mean())
    feat["spread_std"]       = float(spread_s.std())

    # Momentum of top-of-book ratio (is buy pressure growing?)
    n = len(top_ratio)
    for steps in [60, 120, 300]:
        if n >= steps:
            feat[f"top_ratio_mom_{steps}"] = float(
                top_ratio.iloc[steps - 1] - top_ratio.iloc[0]
            )

    return feat


def extract_features(
    slug: str,
    date_str: str,
    prices_df: pd.DataFrame,
    ob_df: pd.DataFrame | None,
) -> dict | None:
    """Combine prices + orderbook features for one market."""
    if len(prices_df) < MIN_ROWS:
        return None

    label = _derive_label(prices_df)
    if label is None:
        return None

    feat: dict = {"slug": slug, "date": date_str, "label": label}

    # ── Prices features ───────────────────────────────────────────────────────
    pw = _window(prices_df)
    feat.update(_prices_features(pw))

    # ── Orderbook features (if available) ─────────────────────────────────────
    if ob_df is not None and len(ob_df) >= MIN_ROWS:
        if "time" in ob_df.columns and "time" in prices_df.columns:
            market_start = prices_df["time"].min()
            market_end   = prices_df["time"].max()
            mask    = (ob_df["time"] >= market_start) & (ob_df["time"] <= market_end)
            aligned = ob_df[mask].sort_values("time").reset_index(drop=True)
        else:
            aligned = ob_df.sort_values("time").reset_index(drop=True) if "time" in ob_df.columns else ob_df

        required = {"top_bid_size", "top_ask_size", "sum_bid_size", "sum_ask_size", "n_bids", "n_asks"}
        if required.issubset(aligned.columns) and len(aligned) >= MIN_ROWS:
            obw = aligned.iloc[:ENTRY_WINDOW]
            feat.update(_orderbook_features(obw))

    return feat


# ── Streaming zip-pair processor ──────────────────────────────────────────────

def _process_zip_pair(prices_zip_path: str, ob_zip_path: str | None) -> list[dict]:
    """
    Process one prices+orderbook zip pair with minimal peak RAM.

    Step 1: Index the orderbook zip (slug → filename, reads only 'slug' column).
            Peak RAM during indexing = one tiny DataFrame at a time.
    Step 2: Stream through the prices zip one market at a time, reading the
            matching orderbook file on-demand and freeing it immediately.
            Peak RAM during processing = 1 prices DataFrame + 1 orderbook DataFrame.
    """
    # Build lightweight index: slug → filename (no DataFrames kept)
    ob_index: dict[str, str] = {}
    if ob_zip_path:
        print(f"  Indexing orderbook: {ob_zip_path}")
        ob_index = _index_zip_by_slug(ob_zip_path, "orderbook")

    records: list[dict] = []
    print(f"  Streaming prices:   {prices_zip_path}")

    # Keep both zips open simultaneously for on-demand orderbook reads
    ob_zf = zipfile.ZipFile(ob_zip_path, "r") if ob_zip_path else None
    try:
        with zipfile.ZipFile(prices_zip_path, "r") as prices_zf:
            parquet_names = sorted(n for n in prices_zf.namelist() if n.endswith(".parquet"))
            print(f"  [prices] {len(parquet_names)} files in zip")

            for name in parquet_names:
                try:
                    with prices_zf.open(name) as fh:
                        prices_df = pd.read_parquet(io.BytesIO(fh.read()))
                except Exception as e:
                    print(f"  SKIP {name}: {e}")
                    continue

                if "slug" not in prices_df.columns or len(prices_df) == 0:
                    continue

                slug     = prices_df["slug"].iloc[0]
                date_str = None
                if "time" in prices_df.columns:
                    ts       = pd.to_datetime(prices_df["time"].iloc[0], utc=True)
                    date_str = ts.strftime("%Y-%m-%d")

                # Read matching orderbook file on-demand, free immediately after
                ob_df = None
                if ob_zf and slug in ob_index:
                    try:
                        with ob_zf.open(ob_index[slug]) as fh:
                            ob_df = pd.read_parquet(io.BytesIO(fh.read()))
                    except Exception as e:
                        print(f"  SKIP ob {slug}: {e}")
                        ob_df = None

                feat = extract_features(slug, date_str, prices_df, ob_df)
                if feat is not None:
                    records.append(feat)

                del prices_df
                if ob_df is not None:
                    del ob_df
    finally:
        if ob_zf:
            ob_zf.close()

    gc.collect()
    print(f"  [pair] {len(records)} usable markets extracted")
    return records


# ── Training ──────────────────────────────────────────────────────────────────

def train(prices_zips: list[str], orderbook_zips: list[str], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_prices = len(prices_zips)
    n_ob     = len(orderbook_zips)
    print(f"\nProcessing {n_prices} zip pair(s) — one at a time to keep RAM low...")

    all_records: list[dict] = []
    for i, prices_path in enumerate(prices_zips):
        ob_path = orderbook_zips[i] if i < n_ob else None
        print(f"\n── Pair {i + 1}/{n_prices} ──")
        all_records.extend(_process_zip_pair(prices_path, ob_path))

    df = pd.DataFrame(all_records)
    del all_records
    gc.collect()

    ob_count = int(df["top_ratio_mean"].notna().sum()) if "top_ratio_mean" in df.columns else 0
    print(f"\n{len(df)} usable markets  ({ob_count} with orderbook features)")
    print(f"Label distribution: Up={int((df['label']==1).sum())}  Down={int((df['label']==0).sum())}")

    if df["label"].nunique() < 2:
        raise RuntimeError("Only one class in data — cannot train classifier")

    # ── Feature columns ───────────────────────────────────────────────────────
    meta_cols    = {"slug", "date", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # ── Time-based split (NEVER random — would leak future data) ──────────────
    df = df.sort_values("date").reset_index(drop=True)
    dates = sorted(df["date"].dropna().unique())

    if len(dates) < 5:
        raise RuntimeError(f"Need at least 5 distinct dates, got {len(dates)}")

    # Last 3 days = test, 3 days before that = val, rest = train
    test_dates  = dates[-3:]
    val_dates   = dates[-6:-3]
    train_dates = dates[:-6]

    train_df = df[df["date"].isin(train_dates)]
    val_df   = df[df["date"].isin(val_dates)]
    test_df  = df[df["date"].isin(test_dates)]

    print(f"\nTrain: {len(train_df)} markets  ({train_dates[0]} → {train_dates[-1]})")
    print(f"Val:   {len(val_df)} markets  ({val_dates[0]} → {val_dates[-1]})")
    print(f"Test:  {len(test_df)} markets  ({test_dates[0]} → {test_dates[-1]})")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

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
    print("\nTop 15 features:")
    for name, imp in importance[:15]:
        bar = "█" * int(imp * 200)
        print(f"  {name:40s}  {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path   = out / "btc_15m_model.pkl"
    feature_path = out / "feature_cols.txt"

    artifact = {
        "model":        model,
        "feature_cols": feature_cols,
        "entry_window": ENTRY_WINDOW,
        "snapshots":    SNAPSHOTS,
        "has_orderbook_features": ob_count > 0,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    feature_path.write_text("\n".join(feature_cols))

    print(f"\nSaved: {model_path}")
    print(f"Saved: {feature_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BTC 15m p_model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single zip pair
  python ml/train.py ^
      --prices    "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
      --orderbook "E:\\orderbook_btc_15m_2026-04-20_2026-04-27.zip"

  # All four pairs (RAM-safe — processed one at a time)
  python ml/train.py ^
      --prices    "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                  "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
                  "E:\\prices_btc_15m_2026-05-06_2026-05-12.zip" ^
                  "E:\\prices_btc_15m_2026-05-13_2026-05-18.zip" ^
      --orderbook "E:\\orderbook_btc_15m_2026-04-20_2026-04-27.zip" ^
                  "E:\\orderbook_btc_15m_2026-04-28_2026-05-05.zip" ^
                  "E:\\orderbook_btc_15m_2026-05-06_2026-05-12.zip" ^
                  "E:\\orderbook_btc_15m_2026-05-13_2026-05-18.zip"
""",
    )
    parser.add_argument(
        "--prices", required=True, nargs="+",
        help="One or more paths to prices zip files",
    )
    parser.add_argument(
        "--orderbook", default=None, nargs="+",
        help="One or more paths to orderbook zip files (optional but recommended)",
    )
    parser.add_argument("--output", default="ml", help="Output directory (default: ml/)")
    args = parser.parse_args()
    train(args.prices, args.orderbook or [], args.output)
