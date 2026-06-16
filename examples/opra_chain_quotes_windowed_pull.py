"""Windowed whole-chain OPRA quotes via DataBentoProvider.fetch_option_chain_quotes.

Pulls an ENTIRE SPX/SPXW option chain's consolidated bid/ask (``cbbo-1m``) for a short
intraday window in a SINGLE ``stype_in="parent"`` request, then saves it to parquet. This is
the sanctioned, cost-guarded counterpart to the per-contract ``fetch_option_quotes``.

WHY A WINDOW: Databento bills on bytes returned, and ``cbbo-1m`` emits one row per contract per
minute. A whole-chain *full day* of SPX+SPXW runs ~$1,900 across 2018-2024; the same chain for a
~20-min window around the cash close is ~$94 — a ~20x lever. Pass ISO *datetimes* (with a time
component) in ``--start``/``--end`` to bill only the window; date-only bounds pull the whole day.

WHAT THIS DEMONSTRATES (the responsible pull recipe):
    1. get_dataset_condition  -> skip/avoid days Databento flags as ``degraded`` (lower quality:
       more crossed/dead quotes). Checked first so you don't analyze a bad day unknowingly.
    2. get_cost_quote         -> FREE pre-flight sizing (stype_in="parent") before any billed call.
    3. fetch_option_chain_quotes -> the single windowed, availability- and cost-guarded pull. Each
       row carries instrument_id AND the OSI raw_symbol (resolved via a same-day definition join).

RESILIENCE: the OPRA gateway can return transient 504s; the metadata and billed calls here are
wrapped in a small retry so a flaky window does not lose the (already cost-verified) pull.

SPX has TWO roots: ``SPX`` (AM-settled monthlies) and ``SPXW`` (PM-settled weeklys / EOM) — pull
each separately for full coverage.

Examples:
    # cost-check + condition only (no billed pull):
    python examples/opra_chain_quotes_windowed_pull.py --root SPX \
        --start 2024-05-15T19:50:00 --end 2024-05-15T20:10:00 --dry-run

    # windowed pull, guarded at $0.50, saved to parquet:
    python examples/opra_chain_quotes_windowed_pull.py --root SPX \
        --start 2024-05-15T19:50:00 --end 2024-05-15T20:10:00 \
        --max-cost-usd 0.50 --out spx_chain_quotes_20240515.parquet

Requires DATABENTO_API_KEY in the environment (or a .env you source first).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ml4t.data.core.exceptions import (
    AuthenticationError,
    CostLimitError,
    DataNotAvailableError,
    NetworkError,
)
from ml4t.data.providers.databento import OPRA_DATASET, DataBentoProvider

# Deterministic, non-transient failures from fetch_option_chain_quotes: a cost ceiling breach,
# a schema-availability rejection, or a bad API key. Retrying these only burns the retry budget
# and masks the real error behind a generic exhaustion message — and for the cost ceiling it
# defeats the rail entirely. _retry re-raises them immediately.
_FATAL = (CostLimitError, DataNotAvailableError, AuthenticationError)


def _retry(fn, *, attempts: int, sleep: float, label: str):
    """Call ``fn`` until it returns a non-None value, riding out transient OPRA 504s.

    Returning ``None`` signals "try again" (a transient failure the caller swallowed); any
    deterministic guard error in ``_FATAL`` re-raises at once rather than being retried.
    """
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            val = fn()
            if val is not None:
                return val
        except _FATAL:
            raise
        except Exception as e:  # noqa: BLE001 - intentional: tolerate transient gateway errors
            last_exc = e
            print(f"  [{label}] attempt {i}/{attempts} failed: {e}")
        time.sleep(sleep)
    if last_exc is not None:
        raise RuntimeError(f"{label}: exhausted {attempts} attempts (last error: {last_exc})")
    raise RuntimeError(f"{label}: exhausted {attempts} attempts (the call kept returning no value)")


def check_condition(provider: DataBentoProvider, start: str, end: str) -> None:
    """Warn loudly if any day in the window is flagged below ``available`` (e.g. degraded)."""
    day_start, day_end = start[:10], end[:10]
    rows = provider.client.metadata.get_dataset_condition(OPRA_DATASET, day_start, day_end)
    bad = [r for r in rows if r.get("condition") != "available"]
    if bad:
        flagged = ", ".join(f"{r['date']}={r['condition']}" for r in bad)
        print(f"  WARNING: non-available day(s) in range -> {flagged}")
        print("           degraded days carry more crossed/dead quotes; prefer an 'available' day.")
    else:
        print(f"  condition: all days {day_start}..{day_end} are 'available' (good)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="SPX", help="underlying root (SPX or SPXW); no .OPT suffix")
    p.add_argument(
        "--start", required=True, help="ISO start, e.g. 2024-05-15T19:50:00 (intraday window)"
    )
    p.add_argument("--end", required=True, help="ISO end, e.g. 2024-05-15T20:10:00")
    p.add_argument(
        "--schema", default="cbbo-1m", help="consolidated-quote schema (default cbbo-1m)"
    )
    p.add_argument(
        "--max-cost-usd", type=float, default=None, help="opt-in hard cost ceiling (USD)"
    )
    p.add_argument("--out", default=None, help="output parquet path (omit to skip saving)")
    p.add_argument(
        "--dry-run", action="store_true", help="condition + cost check only; no billed pull"
    )
    p.add_argument(
        "--attempts", type=int, default=6, help="retry attempts per call (504 resilience)"
    )
    args = p.parse_args()

    provider = DataBentoProvider()
    parent = f"{args.root}.OPT"

    print(f"== dataset condition ({OPRA_DATASET}) ==")
    _retry(
        lambda: (check_condition(provider, args.start, args.end), True)[1],
        attempts=args.attempts,
        sleep=5.0,
        label="condition",
    )

    print("== free cost quote (stype_in='parent') ==")
    est = _retry(
        lambda: (lambda c: c if c and c > 0 else None)(
            provider.get_cost_quote(
                symbols=parent,
                schema=args.schema,
                start=args.start,
                end=args.end,
                stype_in="parent",
                dataset=OPRA_DATASET,
            )
        ),
        attempts=args.attempts,
        sleep=5.0,
        label="cost",
    )
    print(f"  estimated cost: ${est:.4f} for {parent} {args.schema} {args.start} .. {args.end}")

    if args.dry_run:
        print("  --dry-run: stopping before the billed pull.")
        return 0

    print("== windowed whole-chain pull ==")

    def _pull():
        try:
            # Return the frame directly — an EMPTY frame is a legitimate terminal result (no
            # quotes in the window), not a retry signal. Only a transient 504 returns None so
            # _retry tries again; the deterministic guard errors (_FATAL) propagate and fail fast.
            return provider.fetch_option_chain_quotes(
                args.root,
                args.start,
                args.end,
                schema=args.schema,
                max_cost_usd=args.max_cost_usd,
            )
        except NetworkError as e:  # transient 504 on the billed fetch -> retry
            print(f"  [pull] NetworkError (likely 504): {e}")
            return None

    df = _retry(_pull, attempts=args.attempts, sleep=5.0, label="pull")
    if df.is_empty():
        # A degraded/holiday session or an empty window yields a well-formed empty frame — report
        # it cleanly and stop before saving, rather than treating it as a failure.
        print("  no quotes in window (empty frame); nothing to save.")
        return 0
    resolved = df["raw_symbol"].is_not_null().sum()
    print(
        f"  pulled {df.height:,} rows x {df.width} cols | "
        f"contracts={df['instrument_id'].n_unique():,} | "
        f"raw_symbol resolved={resolved:,}/{df.height:,}"
    )

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        print(f"  saved -> {out}")
    else:
        print("  (no --out given; not saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
