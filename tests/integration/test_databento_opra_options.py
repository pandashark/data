"""Integration tests for Databento OPRA options (real API calls).

Two tiers, both gated on DATABENTO_API_KEY and deselected by default:

- ``@pytest.mark.integration``: FREE metadata only — chain (definition) fetch,
  ``get_billable_size``, ``get_cost_quote``, ``_schema_available_from`` (including per-schema
  consolidated-quote availability, e.g. cbbo-1m vs cbbo-1s). Quotes at ~$0.
- ``@pytest.mark.paid_tier``: BILLED single-contract OHLCV and consolidated quotes (one day,
  minimal cost). Requires an OPRA.PILLAR entitlement on the key.

Requirements:
    - DATABENTO_API_KEY environment variable (key starts with "db-").
    - Get a key at https://databento.com/

These tests are deselected by the default pytest lane and only run under ``-m integration``
(free tier) or ``-m paid_tier`` (billed tier).
"""

import os

import polars as pl
import pytest

from ml4t.data.core.schemas import OptionQuoteSchema
from ml4t.data.providers.databento import DataBentoProvider

DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")

pytestmark = pytest.mark.skipif(
    not DATABENTO_API_KEY,
    reason="DATABENTO_API_KEY not set - get key at https://databento.com/",
)

# Realistic OPRA params (see examples/opra_options_databento.py).
UNDERLYING = "SPX"
CONTRACT = "SPX   250321C05800000"  # OSI 21-char: SPX + 250321 + C + 05800000 (strike 5800)
START = "2025-03-03"
END = "2025-03-04"


@pytest.fixture
def provider():
    """Live Databento provider (default dataset stays GLBX.MDP3; OPRA passed per call)."""
    return DataBentoProvider(api_key=DATABENTO_API_KEY)


@pytest.mark.integration
class TestOPRAFreeMetadata:
    """Free (unbilled) metadata: definition chain + cost/availability helpers."""

    def test_fetch_option_chain_returns_definitions(self, provider):
        chain = provider.fetch_option_chain(UNDERLYING, START, END)
        assert chain.height > 0
        assert {"raw_symbol", "instrument_class", "strike_price", "expiration"}.issubset(
            chain.columns
        )
        # Non-OHLCV frame (R1): no price/volume columns.
        assert "open" not in chain.columns
        assert "close" not in chain.columns
        # Only calls/puts in the class column.
        assert set(chain["instrument_class"].unique().to_list()) <= {"C", "P"}

    def test_get_billable_size_is_positive_int(self, provider):
        # Strictly positive: get_billable_size degrades to a 0 sentinel on any SDK/network
        # error, so `>= 0` would pass green even if the real metadata call failed. A real
        # 1-day OHLCV pull on a valid contract bills a non-zero number of records.
        size = provider.get_billable_size(
            symbols=[CONTRACT], schema="ohlcv-1d", start=START, end=END
        )
        assert isinstance(size, int)
        assert size > 0

    def test_get_cost_quote_is_positive_float(self, provider):
        # Strictly positive: get_cost_quote degrades to a 0.0 sentinel on error, so `>= 0.0`
        # would mask a failed quote. A real quote for billable data is a positive dollar amount.
        cost = provider.get_cost_quote(symbols=[CONTRACT], schema="ohlcv-1d", start=START, end=END)
        assert isinstance(cost, float)
        assert cost > 0.0

    def test_schema_available_from(self, provider):
        # cbbo-1m has deep history on OPRA; a real date string is expected.
        avail = provider._schema_available_from("cbbo-1m")
        assert isinstance(avail, str)
        assert len(avail) == 10  # YYYY-MM-DD
        # A nonsense schema degrades to None (not an error).
        assert provider._schema_available_from("not-a-real-schema") is None

    def test_quote_schema_availability_is_per_schema(self, provider):
        # Per-schema availability is the load-bearing detail for quotes: cbbo-1m goes back to
        # ~2013, but cbbo-1s only to 2025-02-20. ISO "YYYY-MM-DD" strings compare chronologically.
        cbbo_1m = provider._schema_available_from("cbbo-1m")
        cbbo_1s = provider._schema_available_from("cbbo-1s")
        assert cbbo_1m is not None and cbbo_1m <= "2013-12-31"
        assert cbbo_1s is not None and cbbo_1s >= "2025-02-20"
        # The two schemas have materially different earliest dates.
        assert cbbo_1m < cbbo_1s

    def test_get_billable_size_for_quote_schema_is_positive(self, provider):
        # Proves the consolidated-quote path is reachable (free) without downloading: a real
        # cbbo-1m query on a valid contract/window bills a non-zero number of records. The
        # sentinel-on-error is 0, so strictly-positive guards against a silently-failed call.
        size = provider.get_billable_size(
            symbols=[CONTRACT], schema="cbbo-1m", start=START, end=END
        )
        assert isinstance(size, int)
        assert size > 0

    def test_dataset_not_mutated_by_chain(self, provider):
        provider.fetch_option_chain(UNDERLYING, START, END)
        assert provider.dataset == "GLBX.MDP3"  # R2


@pytest.mark.integration
@pytest.mark.paid_tier
class TestOPRAPaidOHLCV:
    """Billed: a single-contract, single-day OHLCV pull (kept tiny to minimize cost)."""

    def test_fetch_option_ohlcv(self, provider, cost_tracker):
        df = provider.fetch_option_ohlcv(CONTRACT, START, END, "daily")
        cost_tracker.record_request("databento", 0.001)

        assert not df.is_empty()
        assert df["symbol"].unique().to_list() == [CONTRACT]
        # OHLC invariants.
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        # Consolidated -> one bar per timestamp with an n_venues column.
        assert "n_venues" in df.columns
        assert df["timestamp"].n_unique() == df.height
        # R2: the OPRA dataset was passed per call, not stored.
        assert provider.dataset == "GLBX.MDP3"


@pytest.mark.integration
@pytest.mark.paid_tier
class TestOPRAPaidQuotes:
    """Billed: single-contract consolidated bid/ask quotes (one day, kept tiny)."""

    # The shared 2025-03-03..04 window is valid for BOTH schemas (cbbo-1m back to 2013, cbbo-1s
    # to 2025-02-20) — this directly exercises "use a date range valid for the schema under test".
    @pytest.mark.parametrize("schema", ["cbbo-1m", "cbbo-1s"])
    def test_fetch_option_quotes(self, provider, cost_tracker, schema):
        # Record the free cost estimate for the billed pull (also exercises the cost helper).
        est = provider.get_cost_quote(symbols=[CONTRACT], schema=schema, start=START, end=END)
        q = provider.fetch_option_quotes(CONTRACT, START, END, schema=schema)
        cost_tracker.record_request("databento", est)

        assert not q.is_empty()
        # Conforms to the non-OHLCV OptionQuoteSchema.
        assert set(OptionQuoteSchema.SCHEMA.keys()).issubset(q.columns)
        assert OptionQuoteSchema.validate(q) is True
        # R1: it is NOT an OHLCV frame.
        assert "open" not in q.columns
        assert "close" not in q.columns
        assert "symbol" not in q.columns
        # bid/ask sanity and the spread relationship on rows with both sides present.
        nn = q.drop_nulls(["bid_px_00", "ask_px_00"])
        assert nn.height > 0
        assert (nn["ask_px_00"] >= nn["bid_px_00"]).all()
        assert (nn["spread"] >= 0).all()
        max_err = nn.select(
            (pl.col("ask_px_00") - pl.col("bid_px_00") - pl.col("spread")).abs().max()
        ).item()
        assert max_err < 1e-9
        # Sampling clock is ts_recv, returned ascending.
        assert q["timestamp"].is_sorted()
        # R2: the OPRA dataset was passed per call, not stored.
        assert provider.dataset == "GLBX.MDP3"
