"""Databento OPRA options (SPX) via DataBentoProvider — cost-check FIRST, then explore/download.

Focuses on SPX index options: cash-settled, European-style, ~10x SPY notional. This
example drives the shipped ``DataBentoProvider`` option API end to end:

    provider.get_cost_quote / get_billable_size   -> FREE pre-flight sizing
    provider.fetch_option_chain(...)              -> definition/reference layer (+ filters)
    provider.fetch_option_quotes(...)             -> consolidated bid/ask (publisher 30)
    provider.fetch_option_ohlcv(...)              -> per-contract OHLCV (venues consolidated)

WHY COST-CHECK FIRST: OPRA is the high-volume schema family. Unlike CME OHLCV (where a
year of ES minute bars quotes at ~$0), an OPRA *chain* trades query can be large fast — a
single day of all SPX option trades is millions of rows. The provider's free
``get_cost_quote`` / ``get_billable_size`` let you size any query before a billed call.
Every billed path here prints its own free quote before fetching: ``--quotes`` and
``--download`` each show a per-schema quote first, and the standalone single-contract-vs-
whole-chain contrast (``cost_check``) runs as the pre-flight before ``--download`` or on
its own when you pass neither ``--definition`` nor ``--quotes``.

SPX has TWO roots: `SPX` (AM-settled monthlies) and `SPXW` (PM-settled weeklys / EOM) —
separate chains, query both for full coverage.

Symbology (Databento OPRA):
    - Single contract (OSI 21-char):  "SPX   250321C05800000"
        root padded to 6 + YYMMDD + C/P + strike*1000 zero-padded to 8
      Discover valid contracts with --definition; feed a raw_symbol into --quotes/--download.
    - Whole chain (parent):           "SPX.OPT"   (built internally by fetch_option_chain)

Schema layers:
    - definition  -> reference: put/call (instrument_class), strike, expiration
    - ohlcv-*     -> per-contract prices (per-venue; fetch_option_ohlcv consolidates them)
    - cbbo-1m/cbbo-1s/tcbbo/cmbp-1 -> consolidated bid/ask (publisher 30). Availability is
      PER-SCHEMA: cbbo-1m back to 2013, cmbp-1/tcbbo to 2023, cbbo-1s only 2025-02-20+.
      fetch_option_quotes gates on this at runtime and raises DataNotAvailableError when
      --start is too early, so you never hardcode it.

Requirements:
    - pip install "ml4t-data[databento]"
    - DATABENTO_API_KEY environment variable set
    - OPRA.PILLAR entitlement on the key for any DOWNLOAD (cost/size quotes are free)
    - NOTE: get_cost_quote()/get_billable_size() are FREE. --download/--quotes are BILLED.

Usage:
    # Chain reference — every SPX contract's put/call, strike, expiration (cheap):
    python examples/opra_options_databento.py --definition --start 2025-03-03 --end 2025-03-04

    # Narrow the chain to near-the-money calls for one expiry:
    python examples/opra_options_databento.py --definition --expiry 2025-03-21 \
        --spot 5800 --moneyness 0.05 --right C --start 2025-03-03 --end 2025-03-04

    # Free cost quote — single contract vs. full chain, across schemas:
    python examples/opra_options_databento.py \
        --contract "SPX   250321C05800000" --start 2025-03-03 --end 2025-03-21

    # Consolidated bid/ask (cbbo-1m) for one contract (needs 2025-02-20+):
    python examples/opra_options_databento.py --quotes \
        --contract "SPX   250321C05800000" --start 2025-03-03 --end 2025-03-04

    # Download the single contract's daily bars (small, after you see the quote):
    python examples/opra_options_databento.py --download --frequency daily \
        --contract "SPX   250321C05800000" --start 2025-03-03 --end 2025-03-21
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from ml4t.data.core.exceptions import AuthenticationError, DataNotAvailableError
from ml4t.data.providers.databento import DataBentoProvider


def cost_check(
    provider: DataBentoProvider,
    contract: str | None,
    underlying: str,
    start: str,
    end: str,
) -> None:
    """Print FREE quotes: single contract (cheap) vs whole chain (expensive).

    Uses provider.get_billable_size / get_cost_quote — metadata only, nothing billed.
    """
    print(f"{'target':<22} {'schema':<10} {'billable_bytes':>15} {'cost_usd':>10}")
    print("-" * 60)

    # Single contract across a couple schemas — should be small.
    if contract:
        for schema in ("ohlcv-1d", "ohlcv-1m"):
            size = provider.get_billable_size(
                symbols=[contract], schema=schema, start=start, end=end
            )
            cost = provider.get_cost_quote(symbols=[contract], schema=schema, start=start, end=end)
            print(f"{'1 contract':<22} {schema:<10} {size:>15,} {cost:>10.4f}")

    # Whole chain (parent) — the part that gets expensive. trades is the big one.
    parent = f"{underlying}.OPT"
    for schema in ("ohlcv-1d", "trades"):
        size = provider.get_billable_size(
            symbols=[parent], schema=schema, start=start, end=end, stype_in="parent"
        )
        cost = provider.get_cost_quote(
            symbols=[parent], schema=schema, start=start, end=end, stype_in="parent"
        )
        print(f"{'chain ' + parent:<22} {schema:<10} {size:>15,} {cost:>10.4f}")

    print(
        "\n  (cost quotes use the SDK's default feed mode; batch mode is cheaper for large "
        "pulls and depends on your entitlement tier)"
    )


def definition_mode(
    provider: DataBentoProvider,
    underlying: str,
    start: str,
    end: str,
    *,
    expiry: str | None = None,
    spot: float | None = None,
    moneyness: float | None = None,
    right: str = "both",
) -> pl.DataFrame:
    """Discover the chain via fetch_option_chain: overview first, then the filtered pull-list.

    fetch_option_chain issues schema="definition" ONLY (tiny/free), returns the non-OHLCV
    OptionChainSchema [raw_symbol, instrument_class, strike_price, expiration, instrument_id],
    and applies the expiry/spot/moneyness/right filters internally (AND-composed, sorted).
    """
    full = provider.fetch_option_chain(underlying, start, end)
    if full.is_empty():
        print(f"  no contracts for {underlying} in {start}..{end}")
        return full

    print(
        f"  full chain: {full.height} contracts, "
        f"{full['expiration'].n_unique()} expirations "
        f"({str(full['expiration'].min())[:10]} .. {str(full['expiration'].max())[:10]}), "
        f"strikes {full['strike_price'].min():g}..{full['strike_price'].max():g}"
    )

    chain = full
    if expiry is not None or moneyness is not None or right.upper() in ("C", "P"):
        chain = provider.fetch_option_chain(
            underlying, start, end, expiry=expiry, spot=spot, moneyness=moneyness, right=right
        )
        bits = []
        if expiry:
            bits.append(f"expiry {expiry}")
        if moneyness is not None and spot is not None:
            bits.append(f"strikes within {moneyness:.0%} of spot {spot:g}")
        if right.upper() in ("C", "P"):
            bits.append(f"right={right.upper()}")
        print(f"  filter: {', '.join(bits)}")

    print(f"\n  -> {chain.height} contracts ready to pull:")
    with pl.Config(fmt_str_lengths=40, tbl_rows=20):
        print(chain.select(["raw_symbol", "instrument_class", "strike_price", "expiration"]))

    if 0 < chain.height <= 50:
        print("\n  feed any of these to --contract / --quotes / --download, e.g.:")
        print(f'    --contract "{chain["raw_symbol"][0]}"')
    elif chain.height > 50:
        print("\n  (narrow further with --expiry / --moneyness before pulling prices or quotes)")
    return chain


def quotes_mode(
    provider: DataBentoProvider, contract: str, quote_schema: str, start: str, end: str
) -> pl.DataFrame:
    """Pull consolidated bid/ask via fetch_option_quotes (cbbo-1m by default).

    fetch_option_quotes gates on per-schema availability BEFORE any billed fetch and raises
    DataNotAvailableError when --start predates the schema; it returns the non-OHLCV
    OptionQuoteSchema [timestamp, bid_px_00, ask_px_00, spread, bid_sz_00, ask_sz_00], sampled
    on the ts_recv clock (renamed to timestamp).
    """
    size = provider.get_billable_size(symbols=[contract], schema=quote_schema, start=start, end=end)
    cost = provider.get_cost_quote(symbols=[contract], schema=quote_schema, start=start, end=end)
    print(f"  {quote_schema} quote: {size:,} bytes  ${cost:.4f}  (BILLED on fetch)")

    try:
        q = provider.fetch_option_quotes(contract, start, end, schema=quote_schema)
    except DataNotAvailableError as e:
        raise SystemExit(f"  {e}")

    print(f"  rows={q.height}  (consolidated single book, publisher 30; clock=ts_recv->timestamp)")
    show = [
        c
        for c in ["timestamp", "bid_px_00", "ask_px_00", "spread", "bid_sz_00", "ask_sz_00"]
        if c in q.columns
    ]
    with pl.Config(tbl_width_chars=120, tbl_rows=8):
        print(q.drop_nulls("bid_px_00").select(show).head(8))
    return q


def download_contract(
    provider: DataBentoProvider,
    contract: str,
    frequency: str,
    start: str,
    end: str,
    storage: str,
    *,
    consolidate: bool = True,
) -> pl.DataFrame:
    """Download a SINGLE contract's OHLCV via fetch_option_ohlcv (billed) and store as Parquet.

    fetch_option_ohlcv collapses OPRA's per-venue bars into one bar per timestamp
    (consolidate=True, default) and adds an n_venues column; pass consolidate=False to keep the
    plain canonical OHLCV with no venue metadata.
    """
    df = provider.fetch_option_ohlcv(contract, start, end, frequency, consolidate=consolidate)
    if df.is_empty():
        print(f"  no {frequency} bars for {contract!r} in {start}..{end}")
        return df

    # Key the path on the full OSI symbol, not just the root — otherwise two contracts of the
    # same underlying (different strike/expiry) would overwrite each other.
    symbol = "_".join(contract.split())
    out = (
        Path(storage)
        / f"underlying={contract.split()[0].strip()}"
        / f"contract={symbol}"
        / f"frequency={frequency}"
        / "data.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    print(f"  stored {df.height} {'consolidated ' if consolidate else ''}bars -> {out}")
    if "n_venues" in df.columns:
        print(f"  (n_venues spans {df['n_venues'].min()}..{df['n_venues'].max()} venues per bar)")
    print(df.head(10))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--underlying", default="SPX", help="root for the chain (SPX or SPXW)")
    p.add_argument(
        "--contract", default=None, help='OSI 21-char symbol, e.g. "SPX   250321C05800000"'
    )
    p.add_argument("--frequency", default="daily", help="daily | hourly | minute (for --download)")
    p.add_argument("--start", default="2025-03-03")
    p.add_argument("--end", default="2025-03-21")
    p.add_argument("--storage", default="/tmp/ml4t-opra-demo")
    p.add_argument(
        "--download",
        action="store_true",
        help="After the quote, download the single --contract's OHLCV (BILLED).",
    )
    p.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Keep per-venue bars instead of consolidating to one bar per timestamp.",
    )
    p.add_argument(
        "--definition",
        action="store_true",
        help="Dump the chain reference: put/call, strike, expiration (cheap).",
    )
    p.add_argument("--expiry", default=None, help="filter the chain to one expiry, YYYY-MM-DD")
    p.add_argument("--spot", type=float, default=None, help="underlying price (for --moneyness)")
    p.add_argument(
        "--moneyness",
        type=float,
        default=None,
        help="keep strikes within +/- this fraction of --spot, e.g. 0.05",
    )
    p.add_argument("--right", default="both", help="C | P | both (chain filter)")
    p.add_argument(
        "--quotes",
        action="store_true",
        help="Pull consolidated bid/ask for --contract (needs 2025-02-20+).",
    )
    p.add_argument("--quote-schema", default="cbbo-1m", help="cbbo-1m | cbbo-1s | tcbbo | cmbp-1")
    args = p.parse_args()

    if "DATABENTO_API_KEY" not in os.environ:
        raise SystemExit("Set DATABENTO_API_KEY in your environment first.")

    try:
        provider = DataBentoProvider()
    except AuthenticationError as e:
        raise SystemExit(str(e))

    if args.definition:
        print("=" * 72)
        print(f"  Chain definitions (reference layer): {args.underlying}.OPT")
        print("=" * 72)
        try:
            definition_mode(
                provider,
                args.underlying,
                args.start,
                args.end,
                expiry=args.expiry,
                spot=args.spot,
                moneyness=args.moneyness,
                right=args.right,
            )
        except ValueError as e:
            raise SystemExit(f"  {e}")
        if not (args.quotes or args.download):
            return

    if args.quotes:
        if not args.contract:
            raise SystemExit("--quotes requires --contract.")
        print("\n" + "=" * 72)
        print(f"  Consolidated quotes (BILLED): {args.contract!r}  {args.quote_schema}")
        print("=" * 72)
        quotes_mode(provider, args.contract, args.quote_schema, args.start, args.end)
        if not args.download:
            return

    print("=" * 72)
    print("  Pre-flight cost quote (Databento OPRA metadata — FREE)")
    print("=" * 72)
    cost_check(provider, args.contract, args.underlying, args.start, args.end)

    if not args.download:
        return
    if not args.contract:
        raise SystemExit(
            "--download requires --contract (chain downloads are intentionally not automated)."
        )

    print("\n" + "=" * 72)
    print(f"  Download single contract OHLCV (BILLED): {args.contract!r}  {args.frequency}")
    print("=" * 72)
    download_contract(
        provider,
        args.contract,
        args.frequency,
        args.start,
        args.end,
        args.storage,
        consolidate=not args.no_consolidate,
    )


if __name__ == "__main__":
    main()
