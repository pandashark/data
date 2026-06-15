"""Continuous Futures from Databento — download, cost-check, and intraday insights.

Demonstrates the futures acquisition path (ContinuousDownloader) that the other
examples don't cover, plus two things worth knowing before you pull large slices:

1. A REAL pre-flight cost quote via Databento's metadata API
   (the library's built-in estimate_cost() is a flat per-year heuristic that
   ignores schema, so it is unreliable for minute/tick data).
2. Session-aware intraday structure in the resulting bars (the CME maintenance
   break, the Sunday Globex reopen, and the intraday volume/volatility U-shape).

Continuous symbology used: {PRODUCT}.{roll}.{tenor}, e.g. ES.v.0 = volume-rolled
front month. Output is Hive-partitioned Parquet at:
    {storage}/{SCHEMA}/product={PRODUCT}/year={YEAR}/data.parquet
This example builds the per-schema base dir explicitly (main() passes
{storage}/{SCHEMA} as the downloader's storage_path) so different schemas (daily
vs minute) never share a partition — otherwise download_product's skip_existing
would treat an already-downloaded year as complete and skip the new schema.

Requirements:
    - ml4t-data installed (or run from the repo with src/ on path, as below)
    - databento extra:  pip install "ml4t-data[databento]"
    - DATABENTO_API_KEY environment variable set
    - NOTE: get_cost()/get_billable_size() are FREE quotes. download_product()
      makes a BILLED call (OHLCV is cheap; raw trades/mbp/mbo are not).

Usage:
    # Free cost quote only (no download):
    python examples/continuous_futures_databento.py --cost-only --schema ohlcv-1m \
        --start 2024-01-01 --end 2024-02-01

    # Download one week of ES daily bars, then show insights:
    python examples/continuous_futures_databento.py --schema ohlcv-1d \
        --start 2024-01-02 --end 2024-01-09

    # Download one month of ES minute bars:
    python examples/continuous_futures_databento.py --schema ohlcv-1m \
        --start 2024-01-01 --end 2024-02-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Allow running from the repo without installing (prefers local src/).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from ml4t.data.futures import ContinuousDownloadConfig, ContinuousDownloader


def real_cost_quote(product: str, schema: str, start: str, end: str, roll: str, tenor: int) -> None:
    """Free pre-flight quote via Databento's metadata API (no data billed)."""
    from databento import Historical

    client = Historical(os.environ["DATABENTO_API_KEY"])
    symbol = f"{product}.{roll}.{tenor}"
    size = client.metadata.get_billable_size(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        stype_in="continuous",
        schema=schema,
        start=start,
        end=end,
    )
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        stype_in="continuous",
        schema=schema,
        start=start,
        end=end,
    )
    print(f"  symbol={symbol}  schema={schema}  {start}..{end}")
    print(f"  billable_size = {size:,} bytes")
    print(f"  cost          = ${cost:.4f}  (default historical-streaming mode)")


def download(
    product: str, schema: str, start: str, end: str, roll: str, tenor: int, storage: str
) -> pl.DataFrame:
    """Download a continuous-contract slice and return the stored bars."""
    config = ContinuousDownloadConfig(
        products=[product],
        start=start,
        end=end,
        tenors=[tenor],
        roll_type=roll,
        schema=schema,
        storage_path=storage,
        # api_key defaults to DATABENTO_API_KEY
    )
    dl = ContinuousDownloader(config)
    # Library heuristic, shown for contrast — note it ignores schema:
    print(f"  estimate_cost() (heuristic, unreliable): {dl.estimate_cost()}")
    years_written, failed = dl.download_product(product)
    print(f"  years_written={years_written}  failed={failed}")

    frames = [
        pl.read_parquet(p) for p in Path(storage).glob(f"product={product}/year=*/data.parquet")
    ]
    if not frames:
        return pl.DataFrame()
    # download_product defaults to skip_existing=True, so a shared storage dir keeps
    # every previously-downloaded year on disk. Clip to the requested [start, end)
    # window so insights reflect this run's slice, not the union of all prior runs.
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    return (
        pl.concat(frames)
        .filter(pl.col("ts_event").dt.date().is_between(lo, hi, closed="left"))
        .sort("ts_event")
    )


def insights(df: pl.DataFrame, schema: str) -> None:
    """Intraday volume/volatility profile and the largest single-bar moves.

    The intraday hour-profile only makes sense for sub-daily schemas; daily bars
    are stamped at the UTC session boundary, so we show a daily summary instead.
    """
    if df.height == 0:
        print("  (no rows)")
        return

    if schema in ("ohlcv-1d",):
        daily = df.with_columns(
            ((pl.col("close") / pl.col("close").shift(1) - 1) * 100).round(2).alias("chg_%")
        ).select(["ts_event", "open", "high", "low", "close", "volume", "chg_%"])
        print("  Daily bars (intraday profile skipped — not meaningful at daily resolution):")
        with pl.Config(tbl_rows=40):
            print(daily)
        return

    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("America/New_York").alias("et")
    ).with_columns(
        [
            pl.col("et").dt.hour().alias("hr"),
            (pl.col("close").log() - pl.col("close").log().shift(1)).alias("ret"),
        ]
    )

    days = df["et"].dt.date().n_unique()
    print(f"  rows={df.height}  ET trading days={days}")
    print(f"  span {df['et'].min()} -> {df['et'].max()}")

    print("\n  Intraday profile by ET hour (note missing 17:00 = CME maintenance break):")
    prof = (
        df.group_by("hr")
        .agg(
            [
                pl.col("volume").mean().round(0).alias("avg_vol_min"),
                (pl.col("ret").std() * 1e4).round(2).alias("ret_std_bps"),
            ]
        )
        .sort("hr")
    )
    with pl.Config(tbl_rows=24):
        print(prof)

    print("\n  Largest 1-minute moves (scheduled macro at 08:30/10:00/14:00 ET, or thin reopens):")
    big = (
        df.drop_nulls("ret")
        .with_columns((pl.col("ret").abs() * 1e4).round(1).alias("bps"))
        .sort("bps", descending=True)
        .head(8)
        .select(["et", "close", "bps", "volume"])
    )
    with pl.Config(tbl_rows=10):
        print(big)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", default="ES")
    p.add_argument("--schema", default="ohlcv-1d", help="ohlcv-1d | ohlcv-1h | ohlcv-1m")
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--end", default="2024-01-09")
    p.add_argument("--roll", default="v", help="v=volume, c=calendar")
    p.add_argument("--tenor", type=int, default=0, help="0=front month")
    p.add_argument(
        "--storage",
        default="/tmp/ml4t-futures-demo/continuous",
        help="Base dir; the schema is appended as a subdir so schemas don't collide.",
    )
    p.add_argument("--cost-only", action="store_true", help="Free quote, no download")
    args = p.parse_args()

    if "DATABENTO_API_KEY" not in os.environ:
        raise SystemExit("Set DATABENTO_API_KEY in your environment first.")

    print("=" * 72)
    print("  Pre-flight cost quote (Databento metadata — FREE)")
    print("=" * 72)
    real_cost_quote(args.product, args.schema, args.start, args.end, args.roll, args.tenor)

    if args.cost_only:
        return

    print("\n" + "=" * 72)
    print("  Download (BILLED call)")
    print("=" * 72)
    # Per-schema storage dir so skip_existing doesn't skip a new schema for an already-pulled year.
    storage = str(Path(args.storage) / args.schema)
    df = download(args.product, args.schema, args.start, args.end, args.roll, args.tenor, storage)

    print("\n" + "=" * 72)
    print("  Insights")
    print("=" * 72)
    insights(df, args.schema)


if __name__ == "__main__":
    main()
