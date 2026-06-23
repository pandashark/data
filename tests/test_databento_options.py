"""Unit tests for the Databento options helpers and provider option methods.

The ``consolidate_publishers`` helper tests are pure/offline. The ``DataBentoProvider`` option
tests run in the default lane with a mocked ``Historical`` client (no network).
"""

import builtins
import importlib
import os
import sys
from datetime import UTC, date, datetime
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl
import pytest
from databento.common.error import BentoClientError

from ml4t.data.core.exceptions import (
    AuthenticationError,
    CostLimitError,
    DataNotAvailableError,
    NetworkError,
)
from ml4t.data.core.schemas import (
    OptionChainQuoteSchema,
    OptionChainSchema,
    OptionQuoteSchema,
)
from ml4t.data.providers.databento import (
    OPRA_DATASET,
    DataBentoProvider,
    consolidate_publishers,
)


class TestConsolidatePublishers:
    """Test multi-publisher OPRA bar consolidation."""

    @pytest.fixture
    def multi_venue_df(self):
        """Two timestamps, each with multiple per-venue rows."""
        t0 = datetime(2024, 1, 5, 14, 30)
        t1 = datetime(2024, 1, 5, 14, 31)
        return pl.DataFrame(
            {
                "ts_event": [t0, t0, t0, t1, t1],
                "publisher_id": [1, 2, 3, 1, 2],
                "open": [10.0, 11.0, 9.5, 20.0, 21.0],
                "high": [12.0, 11.5, 10.0, 22.0, 21.5],
                "low": [9.0, 10.5, 9.0, 19.0, 20.5],
                "close": [11.0, 11.2, 9.8, 21.0, 21.2],
                "volume": [100, 50, 25, 80, 200],
            }
        )

    def test_one_bar_per_timestamp(self, multi_venue_df):
        """Output has exactly one row per distinct ts_event."""
        result = consolidate_publishers(multi_venue_df)
        assert result.height == 2
        assert result["ts_event"].n_unique() == 2

    def test_column_order(self, multi_venue_df):
        """Output columns are in the canonical pre-normalization order."""
        result = consolidate_publishers(multi_venue_df)
        assert result.columns == [
            "ts_event",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "n_venues",
        ]

    def test_high_low_volume_aggregation(self, multi_venue_df):
        """high=max, low=min, volume=sum across venues per timestamp."""
        result = consolidate_publishers(multi_venue_df).sort("ts_event")
        # t0 group: high max(12,11.5,10)=12; low min(9,10.5,9)=9; vol 100+50+25=175
        assert result["high"][0] == 12.0
        assert result["low"][0] == 9.0
        assert result["volume"][0] == 175
        # t1 group: high max(22,21.5)=22; low min(19,20.5)=19; vol 80+200=280
        assert result["high"][1] == 22.0
        assert result["low"][1] == 19.0
        assert result["volume"][1] == 280

    def test_open_close_from_highest_volume_venue(self, multi_venue_df):
        """open/close come from the highest-volume row in each group."""
        result = consolidate_publishers(multi_venue_df).sort("ts_event")
        # t0 dominant venue = publisher 1 (vol 100): open 10.0, close 11.0
        assert result["open"][0] == 10.0
        assert result["close"][0] == 11.0
        # t1 dominant venue = publisher 2 (vol 200): open 21.0, close 21.2
        assert result["open"][1] == 21.0
        assert result["close"][1] == 21.2

    def test_n_venues_count(self, multi_venue_df):
        """n_venues is the distinct publisher count per bar."""
        result = consolidate_publishers(multi_venue_df).sort("ts_event")
        assert result["n_venues"][0] == 3
        assert result["n_venues"][1] == 2

    def test_no_symbol_or_timestamp_column(self, multi_venue_df):
        """Result is pre-normalization: keyed on ts_event, no symbol/timestamp."""
        result = consolidate_publishers(multi_venue_df)
        assert "symbol" not in result.columns
        assert "timestamp" not in result.columns
        assert "ts_event" in result.columns

    def test_guard_no_publisher_id_returns_unchanged(self):
        """A frame without publisher_id is returned unchanged (identity)."""
        df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 5)],
                "open": [10.0],
                "high": [12.0],
                "low": [9.0],
                "close": [11.0],
                "volume": [100],
            }
        )
        result = consolidate_publishers(df)
        assert result is df

    def test_guard_non_ohlcv_returns_unchanged(self):
        """A trades-like frame (publisher_id but no OHLCV) is returned unchanged."""
        df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 5), datetime(2024, 1, 5)],
                "publisher_id": [1, 2],
                "price": [10.0, 10.5],
                "size": [5, 3],
            }
        )
        result = consolidate_publishers(df)
        assert result is df

    def test_single_venue_per_timestamp(self):
        """One row per timestamp passes through with n_venues == 1."""
        df = pl.DataFrame(
            {
                "ts_event": [datetime(2024, 1, 5, 14, 30), datetime(2024, 1, 5, 14, 31)],
                "publisher_id": [1, 1],
                "open": [10.0, 20.0],
                "high": [12.0, 22.0],
                "low": [9.0, 19.0],
                "close": [11.0, 21.0],
                "volume": [100, 80],
            }
        )
        result = consolidate_publishers(df).sort("ts_event")
        assert result.height == 2
        assert result["n_venues"].to_list() == [1, 1]
        assert result["open"].to_list() == [10.0, 20.0]
        assert result["close"].to_list() == [11.0, 21.0]

    def test_empty_ohlcv_frame(self):
        """An empty (zero-row) OHLCV frame consolidates to an empty result."""
        df = pl.DataFrame(
            schema={
                "ts_event": pl.Datetime,
                "publisher_id": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        result = consolidate_publishers(df)
        assert result.height == 0
        assert result.columns == [
            "ts_event",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "n_venues",
        ]

    def test_volume_tie_broken_by_lowest_publisher_id(self):
        """On a volume tie, open/close come from the lowest publisher_id (deterministic)."""
        ts = datetime(2024, 1, 5, 14, 30)
        # Both venues share the max volume (100); publisher 2 listed first to prove
        # the result does not depend on input row order.
        df = pl.DataFrame(
            {
                "ts_event": [ts, ts],
                "publisher_id": [2, 1],
                "open": [99.0, 10.0],
                "high": [99.0, 12.0],
                "low": [98.0, 9.0],
                "close": [99.0, 11.0],
                "volume": [100, 100],
            }
        )
        result = consolidate_publishers(df)
        assert result.height == 1
        # publisher 1 wins the tie -> open 10.0, close 11.0
        assert result["open"][0] == 10.0
        assert result["close"][0] == 11.0
        assert result["n_venues"][0] == 2


class TestFetchOptionOHLCV:
    """Test DataBentoProvider.fetch_option_ohlcv (mocked, offline)."""

    CONTRACT = "SPX   250321C05800000"

    @pytest.fixture
    def provider(self):
        """Provider with a mocked Historical client."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client
            return provider

    @staticmethod
    def _multi_venue_response():
        """A .to_df()-bearing mock response: two timestamps, two venues each."""
        t0 = datetime(2025, 3, 3, 14, 30)
        t1 = datetime(2025, 3, 4, 14, 30)
        idx = pd.DatetimeIndex([t0, t0, t1, t1], name="ts_event")
        frame = pd.DataFrame(
            {
                "publisher_id": [1, 2, 1, 2],
                "open": [10.0, 10.5, 20.0, 20.5],
                "high": [12.0, 11.5, 22.0, 22.5],
                "low": [9.0, 9.5, 19.0, 20.0],
                "close": [11.0, 11.2, 21.0, 21.5],
                "volume": [100, 50, 80, 200],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        return resp

    def test_session_adjust_skipped_for_opra(self):
        """adjust_session_dates (a CME futures shift) must NOT move OPRA option queries a day back."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_historical.return_value = Mock()
            provider = DataBentoProvider(
                api_key="test_key", adjust_session_dates=True, session_start_hour_utc=22
            )
            provider.client = mock_historical.return_value
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()

        provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")

        start = provider.client.timeseries.get_range.call_args.kwargs["start"]
        # OPRA keeps the requested start at hour 0, not shifted back to 2025-03-02 @ hour 22.
        assert (start.year, start.month, start.day, start.hour) == (2025, 3, 3, 0)

    def test_consolidated_one_bar_per_timestamp(self, provider):
        """Multi-venue input collapses to one consolidated bar per timestamp."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()

        result = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")

        assert result.height == 2
        result = result.sort("timestamp")
        # t0 dominant venue = publisher 1 (vol 100): open 10, close 11
        # high=max(12,11.5)=12; low=min(9,9.5)=9; volume=150; n_venues=2
        assert result["open"].to_list() == [10.0, 20.5]
        assert result["close"].to_list() == [11.0, 21.5]
        assert result["high"].to_list() == [12.0, 22.5]
        assert result["low"].to_list() == [9.0, 19.0]
        assert result["volume"].to_list() == [150.0, 280.0]
        assert result["n_venues"].to_list() == [2, 2]

    def test_symbol_is_osi_contract(self, provider):
        """The symbol column is the OSI contract string."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()
        result = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")
        assert result["symbol"].unique().to_list() == [self.CONTRACT]

    def test_n_venues_survives_canonical_reorder(self, provider):
        """FR1: n_venues is the trailing column after the canonical reorder."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()
        result = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")
        assert result.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "n_venues",
        ]

    def test_uses_opra_dataset_and_raw_symbol_without_mutating(self, provider):
        """R2: the call targets OPRA with raw_symbol and self.dataset is untouched."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()

        provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")

        call_args = provider.client.timeseries.get_range.call_args
        assert call_args.kwargs["dataset"] == OPRA_DATASET
        assert call_args.kwargs["stype_in"] == "raw_symbol"
        assert call_args.kwargs["symbols"] == [self.CONTRACT]
        assert provider.dataset == "GLBX.MDP3"

    def test_unconsolidated_returns_canonical_schema(self, provider):
        """consolidate=False returns the plain canonical OHLCV schema (no leaked metadata)."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()

        result = provider.fetch_option_ohlcv(
            self.CONTRACT, "2025-03-03", "2025-03-04", "daily", consolidate=False
        )

        # No n_venues, and raw per-venue metadata (publisher_id) must not leak through.
        assert result.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        # _validate_ohlcv dedups duplicate timestamps -> one (unspecified) venue per ts_event.
        assert result.height == 2
        # Which venue survives the dedup is intentionally unspecified, but each surviving bar
        # must be a valid OHLC bar (high>=low etc.).
        assert (result["high"] >= result["low"]).all()
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()

    def test_timestamp_is_utc_aware(self, provider):
        """Populated frame timestamp is UTC-aware, matching the empty-frame schema."""
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()
        result = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")
        assert result.schema["timestamp"] == pl.Datetime("ns", "UTC")

    @staticmethod
    def _empty_response():
        resp = Mock()
        resp.to_df.return_value = pd.DataFrame(
            {
                "publisher_id": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            index=pd.DatetimeIndex([], name="ts_event"),
        )
        return resp

    def test_empty_response_consolidated_matches_populated_schema(self, provider):
        """No rows + consolidate=True -> empty frame still carries n_venues (matches populated)."""
        provider.client.timeseries.get_range.return_value = self._empty_response()

        result = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")

        assert result.is_empty()
        # Column set MUST match the populated consolidated path so per-contract slices concat.
        assert result.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "n_venues",
        ]
        # A populated consolidated frame can be vstacked onto the empty one without a schema error.
        provider.client.timeseries.get_range.return_value = self._multi_venue_response()
        populated = provider.fetch_option_ohlcv(self.CONTRACT, "2025-03-03", "2025-03-04", "daily")
        assert pl.concat([result, populated]).height == populated.height

    def test_empty_response_unconsolidated_returns_canonical(self, provider):
        """No rows + consolidate=False -> plain canonical OHLCV frame (no n_venues)."""
        provider.client.timeseries.get_range.return_value = self._empty_response()

        result = provider.fetch_option_ohlcv(
            self.CONTRACT, "2025-03-03", "2025-03-04", "daily", consolidate=False
        )

        assert result.is_empty()
        assert result.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]


class TestFetchOptionChain:
    """Test DataBentoProvider.fetch_option_chain (mocked, offline)."""

    @pytest.fixture
    def provider(self):
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client
            return provider

    @staticmethod
    def _definition_response():
        """A .to_df()-bearing mock definition response with two contracts + extra metadata."""
        idx = pd.DatetimeIndex([datetime(2025, 3, 3), datetime(2025, 3, 3)], name="ts_recv")
        frame = pd.DataFrame(
            {
                "raw_symbol": ["SPX   250321C05800000", "SPX   250321P05800000"],
                "instrument_class": ["C", "P"],
                "strike_price": [5800.0, 5800.0],
                "expiration": [pd.Timestamp("2025-03-21"), pd.Timestamp("2025-03-21")],
                "instrument_id": [111, 222],
                # raw Databento metadata that must be dropped:
                "rtype": [19, 19],
                "publisher_id": [1, 1],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        return resp

    def test_returns_option_chain_schema_not_ohlcv(self, provider):
        """R1: the frame is the chain schema, never coerced to OHLCV."""
        provider.client.timeseries.get_range.return_value = self._definition_response()

        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")

        assert result.columns == [
            "raw_symbol",
            "instrument_class",
            "strike_price",
            "expiration",
            "instrument_id",
        ]
        assert "open" not in result.columns
        assert "close" not in result.columns
        assert "symbol" not in result.columns
        assert OptionChainSchema.validate(result) is True

    def test_collapses_per_day_definition_duplicates(self, provider):
        """Databento emits one definition row per day; the chain dedups to one row per contract."""
        idx = pd.DatetimeIndex(
            [
                datetime(2025, 3, 3),
                datetime(2025, 3, 3),
                datetime(2025, 3, 4),
                datetime(2025, 3, 4),
            ],
            name="ts_recv",
        )
        frame = pd.DataFrame(
            {
                "raw_symbol": [
                    "SPX   250321C05800000",
                    "SPX   250321P05800000",
                    "SPX   250321C05800000",
                    "SPX   250321P05800000",
                ],
                "instrument_class": ["C", "P", "C", "P"],
                "strike_price": [5800.0, 5800.0, 5800.0, 5800.0],
                "expiration": [pd.Timestamp("2025-03-21")] * 4,
                "instrument_id": [111, 222, 111, 222],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-05")

        assert result.height == 2
        assert result["raw_symbol"].n_unique() == 2
        assert sorted(result["raw_symbol"].to_list()) == [
            "SPX   250321C05800000",
            "SPX   250321P05800000",
        ]

    def test_bypasses_validate_ohlcv(self, provider):
        """R1: _validate_ohlcv is never called for a chain fetch."""
        provider.client.timeseries.get_range.return_value = self._definition_response()
        with patch.object(provider, "_validate_ohlcv") as spy:
            provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")
        spy.assert_not_called()

    def test_uses_definition_schema_only(self, provider):
        """R3: a single get_range with schema=definition/stype_in=parent against the .OPT parent."""
        provider.client.timeseries.get_range.return_value = self._definition_response()

        provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")

        provider.client.timeseries.get_range.assert_called_once()
        call_args = provider.client.timeseries.get_range.call_args
        assert call_args.kwargs["schema"] == "definition"
        assert call_args.kwargs["stype_in"] == "parent"
        assert call_args.kwargs["symbols"] == ["SPX.OPT"]
        assert call_args.kwargs["dataset"] == OPRA_DATASET
        assert provider.dataset == "GLBX.MDP3"

    def test_expiration_normalized_to_utc(self, provider):
        """expiration is normalized to UTC microsecond datetime (matches schema)."""
        provider.client.timeseries.get_range.return_value = self._definition_response()
        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")
        assert result.schema["expiration"] == pl.Datetime("us", "UTC")

    def test_shape_option_chain_accepts_date_expiration(self):
        """A Date-typed expiration casts to midnight UTC us (not an error)."""
        df = pl.DataFrame(
            {
                "raw_symbol": ["SPX   250321C05800000"],
                "instrument_class": ["C"],
                "strike_price": [5800.0],
                "expiration": [date(2025, 3, 21)],
                "instrument_id": [111],
            }
        )
        out = DataBentoProvider._shape_option_chain(df)
        assert out.schema["expiration"] == pl.Datetime("us", "UTC")
        assert OptionChainSchema.validate(out) is True

    def test_shape_option_chain_rejects_unexpected_expiration_dtype(self):
        """A non-integer, non-Date, non-Datetime expiration raises loudly."""
        df = pl.DataFrame(
            {
                "raw_symbol": ["SPX   250321C05800000"],
                "instrument_class": ["C"],
                "strike_price": [5800.0],
                "expiration": ["2025-03-21"],  # string -> unexpected dtype
                "instrument_id": [111],
            }
        )
        with pytest.raises(ValueError, match="Unexpected expiration dtype"):
            DataBentoProvider._shape_option_chain(df)

    def test_expiration_epoch_int_is_normalized(self, provider):
        """A non-Datetime (epoch-ns int) expiration is cast to UTC us, not left as Int64."""
        exp_ns = pd.Timestamp("2025-03-21").value  # int nanoseconds since epoch
        idx = pd.DatetimeIndex([datetime(2025, 3, 3)], name="ts_recv")
        frame = pd.DataFrame(
            {
                "raw_symbol": ["SPX   250321C05800000"],
                "instrument_class": ["C"],
                "strike_price": [5800.0],
                "expiration": [exp_ns],
                "instrument_id": [111],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")

        assert result.schema["expiration"] == pl.Datetime("us", "UTC")
        assert OptionChainSchema.validate(result) is True

    def test_empty_response_returns_empty_chain(self, provider):
        """An empty definition response yields a well-formed empty chain frame."""
        resp = Mock()
        resp.to_df.return_value = pd.DataFrame(
            {
                "raw_symbol": [],
                "instrument_class": [],
                "strike_price": [],
                "expiration": [],
                "instrument_id": [],
            }
        )
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")

        assert result.is_empty()
        assert "raw_symbol" in result.columns
        assert "strike_price" in result.columns

    def test_no_data_error_returns_empty_chain(self, provider):
        """A no-data BentoClientError degrades to an empty chain frame (not a raise)."""
        provider.client.timeseries.get_range.side_effect = BentoClientError("no data found")

        result = provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")

        assert result.is_empty()
        assert result.columns == [
            "raw_symbol",
            "instrument_class",
            "strike_price",
            "expiration",
            "instrument_id",
        ]

    def test_authentication_error_propagates(self, provider):
        """Auth failures are NOT swallowed as 'no data'."""
        provider.client.timeseries.get_range.side_effect = BentoClientError(
            "Unauthorized: Invalid API key"
        )
        with pytest.raises(AuthenticationError):
            provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")


class TestOptionChainFilters:
    """Test the expiry / right / moneyness filters on the chain frame."""

    @staticmethod
    def _chain_frame():
        """A shaped chain frame: two expirations, both rights, a range of strikes."""
        mar = datetime(2025, 3, 21, tzinfo=UTC)
        jun = datetime(2025, 6, 20, tzinfo=UTC)
        return pl.DataFrame(
            {
                "raw_symbol": ["A", "B", "C", "D", "E", "F"],
                "instrument_class": ["C", "P", "C", "P", "C", "P"],
                "strike_price": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0],
                "expiration": [mar, mar, mar, mar, jun, jun],
                "instrument_id": [1, 2, 3, 4, 5, 6],
            }
        ).with_columns(pl.col("expiration").cast(pl.Datetime("us", "UTC")))

    def _filter(self, **kwargs):
        defaults = {"expiry": None, "spot": None, "moneyness": None, "right": "both"}
        defaults.update(kwargs)
        return DataBentoProvider._filter_option_chain(self._chain_frame(), **defaults)

    def test_no_filters_returns_all_sorted(self):
        """No filters -> full chain, sorted by (expiration, strike, class)."""
        result = self._filter()
        assert result.height == 6
        # March rows (strikes 90-105) sort before June rows (110-115).
        assert result["strike_price"].to_list() == [90.0, 95.0, 100.0, 105.0, 110.0, 115.0]

    def test_expiry_filter(self):
        """expiry keeps only rows whose expiration calendar date matches."""
        result = self._filter(expiry=date(2025, 3, 21))
        assert result.height == 4
        assert set(result["raw_symbol"].to_list()) == {"A", "B", "C", "D"}

    def test_right_call(self):
        result = self._filter(right="C")
        assert result["instrument_class"].unique().to_list() == ["C"]
        assert result.height == 3

    def test_right_put(self):
        result = self._filter(right="P")
        assert result["instrument_class"].unique().to_list() == ["P"]
        assert result.height == 3

    def test_right_lowercase_normalized(self):
        """Lower-case right is normalized via .upper()."""
        assert self._filter(right="c")["instrument_class"].unique().to_list() == ["C"]

    def test_right_both_is_noop(self):
        assert self._filter(right="both").height == 6

    def test_moneyness_inclusive_bounds(self):
        """moneyness keeps strikes in [spot*(1-m), spot*(1+m)] inclusive of both bounds."""
        result = self._filter(spot=100.0, moneyness=0.05)
        # [95, 105] inclusive -> strikes 95, 100, 105
        assert result["strike_price"].to_list() == [95.0, 100.0, 105.0]

    def test_combined_filters_and_compose(self):
        """expiry + right + moneyness AND-compose."""
        result = self._filter(expiry=date(2025, 3, 21), right="C", spot=100.0, moneyness=0.05)
        # March calls within [95, 105] -> only strike 100 (row C)
        assert result["raw_symbol"].to_list() == ["C"]

    def test_moneyness_without_spot_raises_valueerror(self):
        """moneyness without spot raises ValueError (NOT SystemExit), before any fetch."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client

            with pytest.raises(ValueError, match="moneyness filter requires a spot price"):
                provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04", moneyness=0.05)
            # Guard fires before any network call.
            mock_client.timeseries.get_range.assert_not_called()

    def test_invalid_right_raises_valueerror(self):
        """An unrecognized right (typo) raises ValueError before any fetch, not a silent no-op."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client

            with pytest.raises(ValueError, match="right must be 'C', 'P', or 'both'"):
                provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04", right="call")
            mock_client.timeseries.get_range.assert_not_called()

    def test_partial_expiry_raises_valueerror(self):
        """A partial/malformed expiry (e.g. "2025-03") raises before any fetch."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client

            with pytest.raises(ValueError):
                provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04", expiry="2025-03")
            mock_client.timeseries.get_range.assert_not_called()

    def test_empty_chain_with_filters_returns_empty_frame(self):
        """An empty definition response short-circuits to a well-formed empty chain even when
        filters are supplied (the short-circuit returns before _filter_option_chain runs)."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client

            empty = Mock()
            empty.to_df.return_value = pd.DataFrame(
                {
                    "raw_symbol": [],
                    "instrument_class": [],
                    "strike_price": [],
                    "expiration": [],
                    "instrument_id": [],
                }
            )
            mock_client.timeseries.get_range.return_value = empty

            result = provider.fetch_option_chain(
                "SPX", "2025-03-03", "2025-03-04", expiry="2025-03-21", right="C"
            )
            assert result.is_empty()
            assert {"raw_symbol", "instrument_class", "strike_price", "expiration"}.issubset(
                result.columns
            )


class TestDefaultLaneGuarantees:
    """Offline guarantees enforced in the default pytest lane (no markers)."""

    def test_option_methods_make_no_live_calls(self):
        """The option methods route through the injected mock client, never a live one."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client

            # OHLCV response (one venue, one bar).
            ohlcv = Mock()
            ohlcv.to_df.return_value = pd.DataFrame(
                {
                    "publisher_id": [1],
                    "open": [10.0],
                    "high": [12.0],
                    "low": [9.0],
                    "close": [11.0],
                    "volume": [100],
                },
                index=pd.DatetimeIndex([datetime(2025, 3, 3)], name="ts_event"),
            )
            # Definition response (one contract).
            definition = Mock()
            definition.to_df.return_value = pd.DataFrame(
                {
                    "raw_symbol": ["SPX   250321C05800000"],
                    "instrument_class": ["C"],
                    "strike_price": [5800.0],
                    "expiration": [pd.Timestamp("2025-03-21")],
                    "instrument_id": [1],
                },
                index=pd.DatetimeIndex([datetime(2025, 3, 3)], name="ts_recv"),
            )
            mock_client.timeseries.get_range.side_effect = [ohlcv, definition]
            mock_client.metadata.get_billable_size.return_value = 123

            provider.fetch_option_ohlcv("SPX   250321C05800000", "2025-03-03", "2025-03-04")
            provider.fetch_option_chain("SPX", "2025-03-03", "2025-03-04")
            provider.get_billable_size(
                symbols=["SPX   250321C05800000"],
                schema="ohlcv-1d",
                start="2025-03-03",
                end="2025-03-04",
            )

            # Every databento interaction went through the mock; the real client was only
            # constructed via the patched Historical (no live network client).
            mock_historical.assert_called_once_with("test_key")
            assert mock_client.timeseries.get_range.call_count == 2
            mock_client.metadata.get_billable_size.assert_called_once()

    def test_library_imports_without_databento(self, monkeypatch):
        """[databento]-uninstalled gating: the library still imports; provider is None."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "databento" or name.startswith("databento."):
                raise ImportError("simulated: databento not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Drop cached databento + provider modules so the import guard re-runs under absence.
        for mod in list(sys.modules):
            if (
                mod == "databento"
                or mod.startswith("databento.")
                or mod in {"ml4t.data.providers", "ml4t.data.providers.databento"}
            ):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        providers = importlib.import_module("ml4t.data.providers")

        # The databento-dependent provider degrades to None; the rest of the package works.
        assert providers.DataBentoProvider is None
        assert providers.MockProvider is not None


class TestFetchOptionQuotes:
    """Test DataBentoProvider.fetch_option_quotes (mocked, offline)."""

    CONTRACT = "SPX   250321C05800000"

    @pytest.fixture
    def provider(self):
        """Provider with a mocked Historical client."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client
            return provider

    @staticmethod
    def _quote_response(*, recv, bid_px, ask_px, bid_sz, ask_sz, event=None, clock_name="ts_recv"):
        """A .to_df()-bearing mock: the sampling clock on a named index, quote cols as columns.

        ``.to_df()`` puts the record timestamp on the index, so the method does ``.reset_index()``
        to materialize the clock column; ``event`` (when given) is an extra ``ts_event`` column.
        """
        idx = pd.DatetimeIndex(recv, name=clock_name)
        data = {
            "bid_px_00": bid_px,
            "ask_px_00": ask_px,
            "bid_sz_00": bid_sz,
            "ask_sz_00": ask_sz,
            "publisher_id": [30] * len(bid_px),
        }
        if event is not None:
            data["ts_event"] = event
        resp = Mock()
        resp.to_df.return_value = pd.DataFrame(data, index=idx)
        return resp

    def test_bypasses_validate_ohlcv_and_returns_quote_frame(self, provider):
        """R1: never routes through _validate_ohlcv; returns a non-OHLCV quote frame."""
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30), datetime(2025, 3, 3, 14, 31)],
            event=[datetime(2025, 3, 3, 14, 29), datetime(2025, 3, 3, 14, 30)],
            bid_px=[1.0, 2.0],
            ask_px=[1.5, 2.5],
            bid_sz=[10, 20],
            ask_sz=[11, 21],
        )
        provider.client.timeseries.get_range.return_value = resp

        with patch.object(provider, "_validate_ohlcv") as mock_validate:
            result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")

        mock_validate.assert_not_called()
        # Quote columns present, OHLCV columns absent, no forced symbol literal.
        assert "bid_px_00" in result.columns
        assert "spread" in result.columns
        assert "open" not in result.columns
        assert "close" not in result.columns
        assert "symbol" not in result.columns
        assert OptionQuoteSchema.validate(result) is True

    def test_spread_is_ask_minus_bid(self, provider):
        """spread = ask_px_00 - bid_px_00 when both price sides are present."""
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30), datetime(2025, 3, 3, 14, 31)],
            bid_px=[1.0, 2.0],
            ask_px=[1.5, 2.5],
            bid_sz=[1, 1],
            ask_sz=[1, 1],
        )
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        assert result["spread"].to_list() == [0.5, 0.5]

    def test_spread_guard_missing_price_side_yields_conformant_frame(self, provider):
        """A response missing a price side does not raise; the frame still conforms to
        OptionQuoteSchema, with the absent ask price and the uncomputable spread back-filled null."""
        idx = pd.DatetimeIndex([datetime(2025, 3, 3, 14, 30)], name="ts_recv")
        frame = pd.DataFrame(
            {"bid_px_00": [1.0], "bid_sz_00": [10], "ask_sz_00": [11], "publisher_id": [30]},
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        # Producer conforms to its own schema even on the sparse path.
        assert result.columns == list(OptionQuoteSchema.SCHEMA.keys())
        assert OptionQuoteSchema.validate(result) is True
        assert result["bid_px_00"].to_list() == [1.0]
        assert result["ask_px_00"].to_list() == [None]  # back-filled
        assert result["spread"].to_list() == [None]  # uncomputable -> null, not dropped

    def test_uses_ts_recv_clock_and_sorts(self, provider):
        """The sampling clock is ts_recv (not ts_event), and the frame is sorted by it."""
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 31), datetime(2025, 3, 3, 14, 30)],  # unsorted
            event=[datetime(2020, 1, 1), datetime(2020, 1, 1)],  # stale, clearly different
            bid_px=[2.0, 1.0],
            ask_px=[2.5, 1.5],
            bid_sz=[1, 1],
            ask_sz=[1, 1],
        )
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        # timestamp derives from ts_recv (2025, not the 2020 ts_event) and is sorted ascending.
        assert result["timestamp"].dt.year().to_list() == [2025, 2025]
        assert result["timestamp"].dt.minute().to_list() == [30, 31]
        # Rows moved with the sort: the 14:30 row (bid 1.0) comes first.
        assert result["bid_px_00"].to_list() == [1.0, 2.0]

    def test_falls_back_to_ts_event_when_no_ts_recv(self, provider):
        """With no ts_recv column, the method falls back to ts_event for the clock."""
        idx = pd.DatetimeIndex([datetime(2025, 3, 3, 14, 30)], name="ts_event")
        frame = pd.DataFrame(
            {
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
                "publisher_id": [30],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        assert "timestamp" in result.columns
        assert result["timestamp"].dt.year().to_list() == [2025]

    def test_tz_aware_clock_is_converted_to_utc(self, provider):
        """A tz-aware ts_recv (as Databento .to_df() returns) is converted to UTC microseconds."""
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2025-03-03 09:30", tz="America/New_York")], name="ts_recv"
        )
        frame = pd.DataFrame(
            {
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
                "publisher_id": [30],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        assert result.schema["timestamp"] == pl.Datetime("us", "UTC")
        # 09:30 America/New_York (EST, UTC-5 on 2025-03-03) -> 14:30 UTC.
        assert result["timestamp"].to_list() == [datetime(2025, 3, 3, 14, 30, tzinfo=UTC)]
        assert OptionQuoteSchema.validate(result) is True

    def test_shape_quotes_missing_clock_raises(self):
        """_shape_option_quotes raises when neither ts_recv nor ts_event is present, rather than
        silently returning a frame missing the required timestamp column."""
        df = pl.DataFrame(
            {"bid_px_00": [1.0], "ask_px_00": [1.5], "bid_sz_00": [1], "ask_sz_00": [1]}
        )
        with pytest.raises(ValueError, match="neither a ts_recv nor ts_event"):
            DataBentoProvider._shape_option_quotes(df)

    def test_shape_option_quotes_epoch_int_is_normalized(self):
        """A non-Datetime (epoch-ns int) ts_recv is coerced to UTC us, not left as Int64."""
        recv_ns = pd.Timestamp("2025-03-03 14:30").value  # int nanoseconds since epoch
        df = pl.DataFrame(
            {
                "ts_recv": [recv_ns],
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
            }
        )
        result = DataBentoProvider._shape_option_quotes(df)
        assert result.schema["timestamp"] == pl.Datetime("us", "UTC")
        # Instant (not just dtype) is correct: epoch-ns for 14:30 -> 14:30 UTC.
        assert result["timestamp"].to_list() == [datetime(2025, 3, 3, 14, 30, tzinfo=UTC)]
        assert OptionQuoteSchema.validate(result) is True

    def test_shape_option_quotes_accepts_date_clock(self):
        """A Date clock casts to midnight UTC us (mirrors the chain's Date handling)."""
        df = pl.DataFrame(
            {
                "ts_recv": [date(2025, 3, 3)],
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
            }
        )
        result = DataBentoProvider._shape_option_quotes(df)
        assert result.schema["timestamp"] == pl.Datetime("us", "UTC")
        assert result["timestamp"].to_list() == [datetime(2025, 3, 3, 0, 0, tzinfo=UTC)]

    def test_shape_option_quotes_rejects_unexpected_clock_dtype(self):
        """A non-integer, non-Date, non-Datetime clock raises loudly (not a silent Int64)."""
        df = pl.DataFrame(
            {
                "ts_recv": ["2025-03-03T14:30:00"],  # string -> unexpected dtype
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
            }
        )
        with pytest.raises(ValueError, match="Unexpected timestamp dtype"):
            DataBentoProvider._shape_option_quotes(df)

    def test_shape_option_quotes_ts_event_fallback_epoch_int(self):
        """With only ts_event as an epoch int, it is selected, coerced, and renamed to timestamp."""
        recv_ns = pd.Timestamp("2025-03-03 14:30").value
        df = pl.DataFrame(
            {
                "ts_event": [recv_ns],
                "bid_px_00": [1.0],
                "ask_px_00": [1.5],
                "bid_sz_00": [1],
                "ask_sz_00": [1],
            }
        )
        result = DataBentoProvider._shape_option_quotes(df)
        assert "timestamp" in result.columns
        assert result.schema["timestamp"] == pl.Datetime("us", "UTC")
        assert result["timestamp"].to_list() == [datetime(2025, 3, 3, 14, 30, tzinfo=UTC)]

    def test_availability_too_early_raises_and_skips_fetch(self, provider):
        """A start earlier than the schema's availability raises before any billed fetch."""
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"cbbo-1s": {"start": "2025-02-20T00:00:00Z"}}
        }
        with pytest.raises(DataNotAvailableError, match="cbbo-1s"):
            provider.fetch_option_quotes(
                self.CONTRACT, "2024-01-01", "2024-01-02", schema="cbbo-1s"
            )
        provider.client.timeseries.get_range.assert_not_called()

    def test_availability_ok_proceeds_to_fetch(self, provider):
        """A start within the schema's availability proceeds to the fetch."""
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"cbbo-1m": {"start": "2013-07-15T00:00:00Z"}}
        }
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30)],
            bid_px=[1.0],
            ask_px=[1.5],
            bid_sz=[1],
            ask_sz=[1],
        )
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        provider.client.timeseries.get_range.assert_called_once()
        assert not result.is_empty()

    def test_availability_is_cached_across_calls(self, provider):
        """Availability is memoized per (dataset, schema): a repeated quotes pull for the same
        schema issues at most one get_dataset_range, not one per call."""
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"cbbo-1m": {"start": "2013-07-15T00:00:00Z"}}
        }
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30)],
            bid_px=[1.0],
            ask_px=[1.5],
            bid_sz=[1],
            ask_sz=[1],
        )
        provider.client.timeseries.get_range.return_value = resp

        for _ in range(3):
            provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        provider.client.metadata.get_dataset_range.assert_called_once()

    def test_availability_unknown_does_not_block(self, provider):
        """An unparseable get_dataset_range payload degrades to None and does not block."""
        provider.client.metadata.get_dataset_range.return_value = {"unexpected": "shape"}
        resp = self._quote_response(
            recv=[datetime(2020, 1, 1)],
            bid_px=[1.0],
            ask_px=[1.5],
            bid_sz=[1],
            ask_sz=[1],
        )
        provider.client.timeseries.get_range.return_value = resp

        provider.fetch_option_quotes(self.CONTRACT, "2020-01-01", "2020-01-02")
        provider.client.timeseries.get_range.assert_called_once()

    def test_call_shape_passes_opra_overrides(self, provider):
        """The low-level call carries the OPRA dataset, raw_symbol stype, and default schema."""
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30)],
            bid_px=[1.0],
            ask_px=[1.5],
            bid_sz=[1],
            ask_sz=[1],
        )
        provider.client.timeseries.get_range.return_value = resp

        provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        kwargs = provider.client.timeseries.get_range.call_args.kwargs
        assert kwargs["dataset"] == OPRA_DATASET
        assert kwargs["stype_in"] == "raw_symbol"
        assert kwargs["schema"] == "cbbo-1m"
        assert kwargs["symbols"] == [self.CONTRACT]

    def test_custom_schema_passed_through(self, provider):
        """A non-default schema flows through to the low-level call."""
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"tcbbo": {"start": "2023-01-01T00:00:00Z"}}
        }
        resp = self._quote_response(
            recv=[datetime(2025, 3, 3, 14, 30)],
            bid_px=[1.0],
            ask_px=[1.5],
            bid_sz=[1],
            ask_sz=[1],
        )
        provider.client.timeseries.get_range.return_value = resp

        provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04", schema="tcbbo")
        assert provider.client.timeseries.get_range.call_args.kwargs["schema"] == "tcbbo"

    def test_empty_response_returns_empty_quote_frame(self, provider):
        """An empty quote response yields a well-formed empty OptionQuoteSchema frame."""
        resp = Mock()
        resp.to_df.return_value = pd.DataFrame(
            {"ts_recv": [], "bid_px_00": [], "ask_px_00": [], "bid_sz_00": [], "ask_sz_00": []}
        )
        provider.client.timeseries.get_range.return_value = resp

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        assert result.is_empty()
        assert result.columns == list(OptionQuoteSchema.SCHEMA.keys())
        assert OptionQuoteSchema.validate(result) is True

    def test_no_data_error_returns_empty_quote_frame(self, provider):
        """A no-data BentoClientError degrades to an empty quote frame (not a raise)."""
        provider.client.timeseries.get_range.side_effect = BentoClientError("no data found")

        result = provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")
        assert result.is_empty()
        assert result.columns == list(OptionQuoteSchema.SCHEMA.keys())

    def test_auth_error_propagates(self, provider):
        """An auth error is not "no data" — it propagates."""
        provider.client.timeseries.get_range.side_effect = BentoClientError(
            "Unauthorized: Invalid API key"
        )
        with pytest.raises(AuthenticationError):
            provider.fetch_option_quotes(self.CONTRACT, "2025-03-03", "2025-03-04")


class TestFetchOptionChainQuotes:
    """Targeted regressions for fetch_option_chain_quotes (the broad suite is TASK-006)."""

    @pytest.fixture
    def provider(self):
        """Provider with a mocked Historical client; availability resolved for cbbo-1m."""
        with patch("ml4t.data.providers.databento.Historical") as mock_historical:
            mock_client = Mock()
            mock_historical.return_value = mock_client
            provider = DataBentoProvider(api_key="test_key")
            provider.client = mock_client
            provider.client.metadata.get_dataset_range.return_value = {
                "schema": {"cbbo-1m": {"start": "2013-07-15T00:00:00Z"}}
            }
            return provider

    @staticmethod
    def _parent_cbbo_response():
        """A parent cbbo response: many contracts interleaved, instrument_id as uint32.

        A real cbbo response carries instrument_id as uint32 (the CBBOMsg struct field). The
        definition chain's key is Int64, so the shaper must reconcile the dtypes before the join.
        """
        idx = pd.DatetimeIndex([datetime(2025, 3, 3, 19, 50)] * 3, name="ts_recv")
        frame = pd.DataFrame(
            {
                "instrument_id": [111, 222, 999],
                "bid_px_00": [1.0, 2.0, 3.0],
                "ask_px_00": [1.5, 2.5, 3.5],
                "bid_sz_00": [10, 20, 30],
                "ask_sz_00": [11, 21, 31],
                "publisher_id": [30, 30, 30],
            },
            index=idx,
        )
        # cbbo carries instrument_id as uint32; cast AFTER construction so the column keeps its
        # positional values (a pd.Series with its own index would align to the DatetimeIndex -> NaN).
        frame["instrument_id"] = frame["instrument_id"].astype("uint32")
        resp = Mock()
        resp.to_df.return_value = frame
        return resp

    @staticmethod
    def _definition_response():
        """A definition response (Int64 instrument_id) covering two of the three contracts."""
        idx = pd.DatetimeIndex([datetime(2025, 3, 3), datetime(2025, 3, 3)], name="ts_recv")
        frame = pd.DataFrame(
            {
                "raw_symbol": ["SPX   250321C05800000", "SPX   250321P05800000"],
                "instrument_class": ["C", "P"],
                "strike_price": [5800.0, 5800.0],
                "expiration": [pd.Timestamp("2025-03-21"), pd.Timestamp("2025-03-21")],
                "instrument_id": [111, 222],
                "rtype": [19, 19],
                "publisher_id": [1, 1],
            },
            index=idx,
        )
        resp = Mock()
        resp.to_df.return_value = frame
        return resp

    def test_cost_guard_fails_closed_when_estimate_unavailable(self, provider):
        """If the free cost-metadata call fails (get_cost_quote -> 0.0), the guard fails CLOSED:
        raise NetworkError and never reach the billed fetch."""
        # Simulate a 504 on the metadata call -> get_cost_quote swallows it to 0.0.
        provider.client.metadata.get_cost.side_effect = Exception("504 gateway timeout")

        with pytest.raises(NetworkError, match="could not verify query cost"):
            provider.fetch_option_chain_quotes(
                "SPX", "2025-03-03T19:50", "2025-03-03T20:10", max_cost_usd=0.25
            )
        # No billed fetch happened.
        provider.client.timeseries.get_range.assert_not_called()

    def test_cost_guard_absent_does_not_quote_or_block(self, provider):
        """With max_cost_usd unset, no cost quote is taken and the pull proceeds."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        result = provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")
        provider.client.metadata.get_cost.assert_not_called()
        assert not result.is_empty()

    def test_raw_symbol_resolved_across_mismatched_id_dtypes(self, provider):
        """The definition join resolves raw_symbol even though cbbo instrument_id is uint32 and
        the chain key is Int64; an unmatched id keeps its row with a null raw_symbol (D2)."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        result = provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")

        assert OptionChainQuoteSchema.validate(result) is True
        assert result.schema["instrument_id"] == pl.Int64
        by_id = {row["instrument_id"]: row["raw_symbol"] for row in result.iter_rows(named=True)}
        assert by_id[111] == "SPX   250321C05800000"
        assert by_id[222] == "SPX   250321P05800000"
        # The third contract has no definition match -> identifiable by id, raw_symbol null.
        assert by_id[999] is None
        # Rows are NOT deduped across contracts.
        assert result.height == 3

    @staticmethod
    def _empty_cbbo_response():
        """An empty parent cbbo response (no rows)."""
        resp = Mock()
        resp.to_df.return_value = pd.DataFrame(
            {
                "ts_recv": [],
                "instrument_id": [],
                "bid_px_00": [],
                "ask_px_00": [],
                "bid_sz_00": [],
                "ask_sz_00": [],
            }
        )
        return resp

    def test_shape_conforms_and_computes_spread(self, provider):
        """The shaped frame conforms to OptionChainQuoteSchema, carries both id columns, and
        computes spread = ask - bid; the call issues exactly two get_range pulls (cbbo + def)."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        result = provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")

        assert result.columns == list(OptionChainQuoteSchema.SCHEMA.keys())
        assert OptionChainQuoteSchema.validate(result) is True
        assert result.schema["instrument_id"] == pl.Int64
        assert result.schema["raw_symbol"] == pl.Utf8
        # bid/ask were [1.0,2.0,3.0]/[1.5,2.5,3.5] -> every spread is 0.5.
        assert result["spread"].to_list() == [0.5, 0.5, 0.5]
        assert provider.client.timeseries.get_range.call_count == 2

    def test_empty_response_returns_empty_chain_quote_frame(self, provider):
        """An empty cbbo response yields a well-formed empty frame BEFORE any definition pull."""
        provider.client.timeseries.get_range.return_value = self._empty_cbbo_response()

        result = provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")

        assert result.is_empty()
        assert result.columns == list(OptionChainQuoteSchema.SCHEMA.keys())
        assert OptionChainQuoteSchema.validate(result) is True
        # No definition pull fires once the cbbo frame is empty (early return).
        assert provider.client.timeseries.get_range.call_count == 1

    def test_cost_over_cap_raises_and_skips_fetch(self, provider):
        """An estimate above max_cost_usd raises CostLimitError and never bills a fetch."""
        provider.client.metadata.get_cost.return_value = 5.0

        with pytest.raises(CostLimitError, match="exceeds limit"):
            provider.fetch_option_chain_quotes(
                "SPX", "2025-03-03T19:50", "2025-03-03T20:10", max_cost_usd=0.25
            )
        provider.client.timeseries.get_range.assert_not_called()
        # The guard cost-quotes the parent symbol (free metadata), never billable_size.
        assert provider.client.metadata.get_cost.call_args.kwargs["stype_in"] == "parent"
        provider.client.metadata.get_billable_size.assert_not_called()

    def test_cost_under_cap_proceeds_to_fetch(self, provider):
        """An estimate at or below max_cost_usd proceeds to the billed fetch."""
        provider.client.metadata.get_cost.return_value = 0.05
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        result = provider.fetch_option_chain_quotes(
            "SPX", "2025-03-03T19:50", "2025-03-03T20:10", max_cost_usd=0.25
        )
        provider.client.metadata.get_cost.assert_called_once()
        assert not result.is_empty()
        assert provider.client.timeseries.get_range.call_count == 2

    def test_availability_too_early_raises_and_skips_fetch(self, provider):
        """A start before the schema's availability raises DataNotAvailableError before billing."""
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"cbbo-1s": {"start": "2025-02-20T00:00:00Z"}}
        }
        with pytest.raises(DataNotAvailableError, match="cbbo-1s"):
            provider.fetch_option_chain_quotes(
                "SPX", "2025-01-01T19:50", "2025-01-01T20:10", schema="cbbo-1s"
            )
        provider.client.timeseries.get_range.assert_not_called()

    def test_availability_gate_handles_unhyphenated_iso_start(self, provider):
        """Regression: a basic (unhyphenated) ISO start before availability is still gated.

        The gate compares PARSED bounds; a raw-string compare would have let "20250101" through
        because "20250101" < "2025-02-20" is False lexicographically (the '0' vs '-' at index 4).
        """
        provider.client.metadata.get_dataset_range.return_value = {
            "schema": {"cbbo-1s": {"start": "2025-02-20T00:00:00Z"}}
        }
        with pytest.raises(DataNotAvailableError, match="cbbo-1s"):
            provider.fetch_option_chain_quotes("SPX", "20250101", "20250102", schema="cbbo-1s")
        provider.client.timeseries.get_range.assert_not_called()

    def test_intraday_window_reaches_sdk_unfloored(self, provider):
        """An ISO datetime window flows to the SDK with its HH:MM intact (TASK-001 plumbing)."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50:00", "2025-03-03T20:10:00")

        cbbo_call = provider.client.timeseries.get_range.call_args_list[0].kwargs
        assert (cbbo_call["start"].hour, cbbo_call["start"].minute) == (19, 50)
        assert (cbbo_call["end"].hour, cbbo_call["end"].minute) == (20, 10)

    def test_date_only_bounds_floor_and_ceil(self, provider):
        """Date-only bounds keep the whole-day window: start floors to 00:00, end ceils to 23:59."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        provider.fetch_option_chain_quotes("SPX", "2025-03-03", "2025-03-04")

        cbbo_call = provider.client.timeseries.get_range.call_args_list[0].kwargs
        assert (cbbo_call["start"].hour, cbbo_call["start"].minute) == (0, 0)
        assert (cbbo_call["end"].hour, cbbo_call["end"].minute, cbbo_call["end"].second) == (
            23,
            59,
            59,
        )

    def test_windowed_billable_forwards_distinct_window_and_day_bounds(self, provider):
        """The free billable-size estimate forwards the intraday window and the full-day bounds
        to the SDK as given (the cost lever is the narrower window; the actual window<day byte
        comparison is asserted live in the @integration tier, not against a dictated mock)."""
        provider.client.metadata.get_billable_size.return_value = 120
        for start, end in (
            ("2025-03-03T19:50:00", "2025-03-03T20:10:00"),
            ("2025-03-03", "2025-03-04"),
        ):
            provider.get_billable_size(
                symbols="SPX.OPT",
                schema="cbbo-1m",
                start=start,
                end=end,
                stype_in="parent",
                dataset=OPRA_DATASET,
            )
        calls = provider.client.metadata.get_billable_size.call_args_list
        assert len(calls) == 2
        # Window bounds carry the intraday time; the full-day bounds are date-only — the helper
        # forwards both verbatim with the parent stype and OPRA dataset.
        assert (calls[0].kwargs["start"], calls[0].kwargs["end"]) == (
            "2025-03-03T19:50:00",
            "2025-03-03T20:10:00",
        )
        assert (calls[1].kwargs["start"], calls[1].kwargs["end"]) == ("2025-03-03", "2025-03-04")
        for call in calls:
            assert call.kwargs["stype_in"] == "parent"
            assert call.kwargs["dataset"] == OPRA_DATASET

    def test_definition_join_call_shape_and_no_mutation(self, provider):
        """Both pulls target OPRA with stype_in='parent'; the second is the definition pull;
        self.dataset is never mutated (R2)."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")

        calls = provider.client.timeseries.get_range.call_args_list
        assert len(calls) == 2
        cbbo, definition = calls[0].kwargs, calls[1].kwargs
        assert cbbo["dataset"] == OPRA_DATASET
        assert cbbo["stype_in"] == "parent"
        assert cbbo["schema"] == "cbbo-1m"
        assert cbbo["symbols"] == ["SPX.OPT"]
        assert definition["dataset"] == OPRA_DATASET
        assert definition["stype_in"] == "parent"
        assert definition["schema"] == "definition"
        assert definition["symbols"] == ["SPX.OPT"]
        # R2: the OPRA dataset is passed per call; the instance default is untouched.
        assert provider.dataset == "GLBX.MDP3"

    def test_custom_schema_passed_through(self, provider):
        """A non-default quote schema reaches the cbbo pull; the definition pull stays definition."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        provider.fetch_option_chain_quotes(
            "SPX", "2025-03-03T19:50", "2025-03-03T20:10", schema="tcbbo"
        )
        calls = provider.client.timeseries.get_range.call_args_list
        assert calls[0].kwargs["schema"] == "tcbbo"
        assert calls[1].kwargs["schema"] == "definition"

    def test_no_data_error_returns_empty_chain_quote_frame(self, provider):
        """A no-data BentoClientError degrades to an empty frame (not a raise)."""
        provider.client.timeseries.get_range.side_effect = BentoClientError("no data found")

        result = provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")
        assert result.is_empty()
        assert result.columns == list(OptionChainQuoteSchema.SCHEMA.keys())

    def test_auth_error_propagates(self, provider):
        """An auth error is not "no data" — it propagates."""
        provider.client.timeseries.get_range.side_effect = BentoClientError(
            "Unauthorized: Invalid API key"
        )
        with pytest.raises(AuthenticationError):
            provider.fetch_option_chain_quotes("SPX", "2025-03-03T19:50", "2025-03-03T20:10")

    def test_bypasses_validate_ohlcv(self, provider):
        """R1: never routes through _validate_ohlcv; OHLCV columns are absent."""
        provider.client.timeseries.get_range.side_effect = [
            self._parent_cbbo_response(),
            self._definition_response(),
        ]
        with patch.object(provider, "_validate_ohlcv") as mock_validate:
            result = provider.fetch_option_chain_quotes(
                "SPX", "2025-03-03T19:50", "2025-03-03T20:10"
            )
        mock_validate.assert_not_called()
        assert "bid_px_00" in result.columns
        assert "spread" in result.columns
        assert "open" not in result.columns
        assert "close" not in result.columns


_DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")


@pytest.mark.skipif(
    not _DATABENTO_API_KEY,
    reason="DATABENTO_API_KEY not set - get key at https://databento.com/",
)
@pytest.mark.integration
class TestFetchOptionChainQuotesFreeMetadata:
    """Gated, deselected-by-default LIVE tier — FREE metadata only (never bills).

    Deselected by the default ``-m "not integration ..."`` lane; run with ``-m integration``.
    Asserts strictly-positive values because the helpers degrade to a 0/0.0 sentinel on error,
    so ``>= 0`` would mask a silently-failed call. NEVER calls fetch_option_chain_quotes here.
    """

    @pytest.fixture
    def provider(self):
        return DataBentoProvider(api_key=_DATABENTO_API_KEY)

    def test_parent_cost_quote_is_positive(self, provider):
        cost = provider.get_cost_quote(
            symbols="SPX.OPT",
            schema="cbbo-1m",
            start="2024-05-15T19:50:00",
            end="2024-05-15T20:10:00",
            stype_in="parent",
            dataset=OPRA_DATASET,
        )
        assert cost > 0.0

    def test_windowed_bills_fewer_bytes_than_full_day(self, provider):
        window = provider.get_billable_size(
            symbols="SPX.OPT",
            schema="cbbo-1m",
            start="2024-05-15T19:50:00",
            end="2024-05-15T20:10:00",
            stype_in="parent",
            dataset=OPRA_DATASET,
        )
        day = provider.get_billable_size(
            symbols="SPX.OPT",
            schema="cbbo-1m",
            start="2024-05-15",
            end="2024-05-16",
            stype_in="parent",
            dataset=OPRA_DATASET,
        )
        assert 0 < window < day

    def test_cbbo_1m_available_in_deep_history(self, provider):
        avail = provider._schema_available_from("cbbo-1m", dataset=OPRA_DATASET)
        assert avail is not None
        assert avail <= "2014-01-01"


@pytest.mark.skipif(
    not _DATABENTO_API_KEY,
    reason="DATABENTO_API_KEY not set - get key at https://databento.com/",
)
@pytest.mark.integration
@pytest.mark.paid_tier
class TestFetchOptionChainQuotesPaid:
    """Gated, deselected-by-default LIVE tier — BILLED. Run with ``-m paid_tier``.

    A tiny 1-minute windowed whole-chain pull, capped with a max_cost_usd guard to bound spend.
    """

    @pytest.fixture
    def provider(self):
        return DataBentoProvider(api_key=_DATABENTO_API_KEY)

    def test_tiny_windowed_chain_pull(self, provider):
        df = provider.fetch_option_chain_quotes(
            "SPX",
            "2024-05-15T19:59:00",
            "2024-05-15T20:00:00",
            max_cost_usd=2.0,
        )
        assert df.columns == list(OptionChainQuoteSchema.SCHEMA.keys())
        assert OptionChainQuoteSchema.validate(df) is True
        if not df.is_empty():
            # A parent pull interleaves many distinct contracts.
            assert df["instrument_id"].n_unique() > 1


class TestCoerceEpochDatetime:
    """Direct tests for the shared epoch/Date -> Datetime drift guard."""

    def test_int_epoch_ns_to_datetime(self):
        """An Int64 column is interpreted as epoch nanoseconds and cast to Datetime."""
        df = pl.DataFrame({"t": [pd.Timestamp("2025-03-03 14:30").value]})
        out = DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")
        assert isinstance(out.schema["t"], pl.Datetime)
        assert out["t"].to_list() == [datetime(2025, 3, 3, 14, 30)]

    def test_date_to_datetime_midnight(self):
        """A Date column casts to midnight Datetime."""
        df = pl.DataFrame({"t": [date(2025, 3, 3)]})
        out = DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")
        assert isinstance(out.schema["t"], pl.Datetime)
        assert out["t"].to_list() == [datetime(2025, 3, 3, 0, 0)]

    def test_datetime_passthrough_returns_unchanged(self):
        """An already-Datetime column is returned unchanged (no-op)."""
        df = pl.DataFrame({"t": [datetime(2025, 3, 3, 14, 30)]}).with_columns(
            pl.col("t").cast(pl.Datetime("us", "UTC"))
        )
        out = DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")
        assert out.schema["t"] == pl.Datetime("us", "UTC")
        assert out["t"].to_list() == df["t"].to_list()

    def test_unexpected_dtype_raises_with_label(self):
        """A string column raises a labelled ValueError, not a silent misread."""
        df = pl.DataFrame({"t": ["2025-03-03"]})
        with pytest.raises(ValueError, match="Unexpected t dtype for x"):
            DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")

    def test_empty_int_frame_coerces_without_error(self):
        """An empty Int64 column coerces (cast on an empty column is safe)."""
        df = pl.DataFrame(schema={"t": pl.Int64})
        out = DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")
        assert isinstance(out.schema["t"], pl.Datetime)
        assert out.height == 0

    def test_empty_datetime_frame_no_op(self):
        """An empty already-Datetime column passes through unchanged."""
        df = pl.DataFrame(schema={"t": pl.Datetime("us", "UTC")})
        out = DataBentoProvider._coerce_epoch_datetime(df, "t", label="x")
        assert out.schema["t"] == pl.Datetime("us", "UTC")
        assert out.height == 0
