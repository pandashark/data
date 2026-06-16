"""Databento provider implementation for futures, equities, and OPRA options data.

This provider supports:
- Multiple schemas (ohlcv-1m, ohlcv-1h, ohlcv-1d)
- Continuous futures contracts (symbol.v.0)
- CME session date logic for futures
- OPRA options (OPRA.PILLAR): single-contract OHLCV with multi-venue
  consolidation, plus chain discovery via the definition schema
- Free metadata cost/availability checks before any billed pull
- Native Polars output
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from typing import Any, ClassVar

import polars as pl
import structlog
from databento import Historical
from databento.common.error import BentoClientError, BentoServerError

from ml4t.data.core.exceptions import (
    AuthenticationError,
    CostLimitError,
    DataNotAvailableError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.core.schemas import (
    OptionChainQuoteSchema,
    OptionChainSchema,
    OptionQuoteSchema,
)
from ml4t.data.providers.base import BaseProvider, ProviderCapabilities

logger = structlog.get_logger()

# Consolidated U.S. options feed (OPRA). Used as a per-call dataset override by the
# option fetch methods so the default futures/equities dataset (self.dataset) is never mutated.
OPRA_DATASET = "OPRA.PILLAR"

_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


def consolidate_publishers(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse OPRA's per-exchange bars into one bar per timestamp.

    OPRA is a consolidated feed: each contract gets one bar PER reporting venue
    (~17 options exchanges), so raw bars share timestamps and volume is split
    across venues. We aggregate per ``ts_event``: high=max, low=min, volume=sum,
    and take open/close from the highest-volume venue for that bar. Ties in volume
    are broken deterministically by the lowest ``publisher_id`` so the selection is
    independent of input order and engine sort-stability.

    The result is keyed on ``ts_event`` with no ``symbol`` column — it is a
    pre-normalization frame intended to be passed through ``_transform_data``
    (which renames ``ts_event`` -> ``timestamp`` and adds ``symbol``). The trailing
    ``n_venues`` column rides through as a tolerated extra column.

    Returns the frame unchanged for non-OHLCV schemas (e.g. trades), which carry
    ``publisher_id`` but no open/high/low/close/volume to aggregate, and for frames
    lacking ``publisher_id`` entirely.

    Args:
        df: Raw Databento OPRA OHLCV frame with per-venue rows.

    Returns:
        One consolidated bar per ``ts_event`` with columns
        ``ts_event, open, high, low, close, volume, n_venues``; or ``df`` unchanged
        if it is not a multi-venue OHLCV frame.
    """
    if "publisher_id" not in df.columns or not _OHLCV_COLS.issubset(df.columns):
        return df
    by = ["volume", "publisher_id"]
    descending = [True, False]
    return (
        df.group_by("ts_event")
        .agg(
            [
                pl.col("open").sort_by(by, descending=descending).first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").sort_by(by, descending=descending).first(),
                pl.col("volume").sum(),
                pl.col("publisher_id").n_unique().alias("n_venues"),
            ]
        )
        .sort("ts_event")
        .select(["ts_event", "open", "high", "low", "close", "volume", "n_venues"])
    )


class DataBentoProvider(BaseProvider):
    """Thin wrapper around databento.Historical for API consistency and incremental updates.

    **When to use this wrapper:**
    - Automated data pipelines with incremental updates
    - Cross-provider comparisons (Yahoo vs Databento vs EODHD)
    - OHLCV bars only (daily/hourly/minute)
    - OPRA options: single-contract OHLCV (``fetch_option_ohlcv``) and chain
      discovery (``fetch_option_chain``), with free cost checks
      (``get_billable_size`` / ``get_cost_quote``) before any billed pull
    - Consistent Polars DataFrame output

    **When to use databento.Historical directly:**
    - Advanced schemas: trades, MBO, MBP-10, quotes, imbalance, statistics
    - Symbology API: symbol resolution, contract specifications
    - Cost estimation beyond the wrapped ``get_billable_size``/``get_cost_quote`` helpers
    - Live streaming: WebSocket real-time data
    - Batch operations: multi-symbol, multi-schema requests

    **Quick start with native SDK:**
        >>> import databento as db
        >>> client = db.Historical(api_key)
        >>> # Get continuous front month futures
        >>> data = client.timeseries.get_range(
        ...     dataset='GLBX.MDP3',
        ...     symbols='ES.c.0',  # Continuous front month
        ...     schema='ohlcv-1d',
        ...     stype_in='continuous',
        ...     start='2024-01-01',
        ...     end='2024-12-31'
        ... )
        >>> import polars as pl
        >>> df = pl.from_pandas(data.to_df())

    **This wrapper exposes the native client:**
        >>> provider = DataBentoProvider(api_key)
        >>> provider.client  # Access databento.Historical directly
        >>> # Use for advanced features while keeping incremental update infrastructure

    See: https://docs.databento.com/ for full native SDK capabilities.
    """

    # Databento has generous rate limits
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (100, 1.0)

    # Schema mappings
    SCHEMA_MAPPING = {
        "ohlcv-1m": "ohlcv-1m",
        "ohlcv-1h": "ohlcv-1h",
        "ohlcv-1d": "ohlcv-1d",
        "trades": "trades",
        "quotes": "tbbo",
        "mbo": "mbo",
    }

    def __init__(
        self,
        api_key: str | None = None,
        dataset: str = "GLBX.MDP3",
        rate_limit: tuple[int, float] | None = None,
        adjust_session_dates: bool = False,
        session_start_hour_utc: int = 0,
    ):
        """Initialize Databento provider.

        Args:
            api_key: Databento API key (or set DATABENTO_API_KEY env var)
            dataset: Default dataset to use (e.g., GLBX.MDP3, XNAS.ITCH)
            rate_limit: Optional custom rate limit (calls, period_seconds)
            adjust_session_dates: Whether to adjust dates for CME session logic
            session_start_hour_utc: Hour in UTC when trading session starts (for futures)
        """
        super().__init__(rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT)

        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                provider="databento",
                message="Databento API key not provided. "
                "Set DATABENTO_API_KEY environment variable or pass api_key parameter.",
            )

        try:
            self.client = Historical(self.api_key)
        except Exception as e:
            raise AuthenticationError(
                provider="databento",
                message=f"Failed to initialize Databento client: {e}",
            )

        self.dataset = dataset
        self.default_schema = "ohlcv-1m"
        self.adjust_session_dates = adjust_session_dates
        self.session_start_hour_utc = session_start_hour_utc
        # Memoizes _schema_available_from per (dataset, schema). Availability is effectively
        # static within a session, so a per-contract quotes loop hits get_dataset_range once.
        self._availability_cache: dict[tuple[str, str], str] = {}

        self.logger.info(
            "Initialized Databento provider",
            dataset=dataset,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "databento"

    def capabilities(self) -> ProviderCapabilities:
        """Return Databento provider capabilities."""
        return ProviderCapabilities(
            supports_intraday=True,
            supports_futures=True,
            supports_options=True,
            requires_api_key=True,
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Return empty DataFrame with correct OHLCV schema."""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns", "UTC"),
                "symbol": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    def _map_frequency_to_schema(self, frequency: str) -> str:
        """Map frequency parameter to Databento schema."""
        freq_lower = frequency.lower()

        if freq_lower in ["daily", "day", "1d", "d"]:
            return "ohlcv-1d"
        if freq_lower in ["hourly", "hour", "1h", "h"]:
            return "ohlcv-1h"
        if freq_lower in ["minute", "min", "1m", "m"]:
            return "ohlcv-1m"
        if freq_lower in ["tick", "trades"]:
            return "trades"
        if freq_lower in ["quote", "quotes", "tbbo"]:
            return "tbbo"
        if freq_lower in ["mbo"]:
            return "mbo"

        # Default to daily
        return "ohlcv-1d"

    def _fetch_raw_data(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        dataset: str | None = None,
        stype_in: str | None = None,
        schema: str | None = None,
    ) -> Any:
        """Fetch raw data from Databento API.

        Args:
            symbol: Symbol / raw contract string to fetch.
            start: Start bound — ISO-8601 date (``YYYY-MM-DD``, floored to 00:00:00 UTC) or
                datetime (``YYYY-MM-DDTHH:MM[:SS][+offset]``, honored verbatim for an intraday
                window).
            end: End bound — ISO-8601 date (``YYYY-MM-DD``, ceiled to 23:59:59 UTC) or datetime
                (honored verbatim), same form as ``start``.
            frequency: Human frequency; mapped to a schema when ``schema`` is not given.
            dataset: Optional per-call dataset override (e.g. ``OPRA_DATASET``). Defaults to
                ``self.dataset``; the instance attribute is never mutated (R2).
            stype_in: Optional symbology type override (e.g. ``"parent"``). Defaults to
                ``"raw_symbol"``.
            schema: Optional explicit schema override. When given, it wins over the
                frequency mapping.
        """
        # Explicit schema override wins; otherwise map from frequency (never None).
        resolved_schema = schema if schema is not None else self._map_frequency_to_schema(frequency)
        # Per-call dataset override without mutating self.dataset (R2).
        resolved_dataset = dataset if dataset is not None else self.dataset
        resolved_stype_in = stype_in if stype_in is not None else "raw_symbol"

        # Parse start/end as ISO-8601 date OR datetime. A date-only bound keeps the historical
        # whole-day window (start floored to 00:00:00, end ceiled to 23:59:59 UTC); an explicit
        # time component is honored verbatim so callers can request an intraday window (e.g. the
        # ~20-min cbbo-1m snapshot around the cash close) instead of a full session day.
        start_dt = self._resolve_request_bound(start, is_end=False)
        end_dt = self._resolve_request_bound(end, is_end=True)

        # Adjust for session dates if enabled (for futures with CME session logic)
        # The CME maintenance-break shift is futures-only; never apply it to OPRA option calls.
        if self.adjust_session_dates and resolved_dataset != OPRA_DATASET:
            from datetime import timedelta

            # Move start back by one day and set to session start hour
            start_dt = (start_dt - timedelta(days=1)).replace(hour=self.session_start_hour_utc)
            # End stays at end of requested day
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        try:
            self.logger.debug(
                "Fetching from Databento",
                symbol=symbol,
                dataset=resolved_dataset,
                schema=resolved_schema,
            )

            response = self.client.timeseries.get_range(
                dataset=resolved_dataset,
                start=start_dt,
                end=end_dt,
                symbols=[symbol],
                schema=resolved_schema,
                stype_in=resolved_stype_in,
            )

            return response

        except BentoClientError as e:
            if "unauthorized" in str(e).lower():
                raise AuthenticationError(
                    provider=self.name,
                    message=f"Databento authentication failed: {e}",
                )
            if "rate limit" in str(e).lower():
                raise RateLimitError(provider=self.name)
            raise DataNotAvailableError(self.name, f"Client error: {e}")

        except BentoServerError as e:
            raise NetworkError(
                provider=self.name,
                message=f"Databento server error: {e}",
            )

        except Exception as e:
            self.logger.error("Error fetching from Databento", error=str(e), symbol=symbol)
            raise NetworkError(
                provider=self.name,
                message=f"Failed to fetch data from Databento: {e}",
            )

    @staticmethod
    def _resolve_request_bound(value: str, *, is_end: bool) -> datetime:
        """Parse an ISO-8601 date or datetime into a UTC request bound.

        A date-only string (no ``T``/space time separator) keeps the historical whole-day
        window: the start bound floors to ``00:00:00`` and the end bound ceils to ``23:59:59``.
        An explicit time component is honored verbatim — a naive value is treated as UTC, a
        tz-aware value (``Z``/offset) is converted to UTC — so a caller can request an intraday
        window instead of a full session day. A bare date and an explicit midnight datetime
        parse to the same value, so the date-only floor/ceil is decided from the INPUT STRING,
        not the parsed datetime.
        """
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            return dt.astimezone(UTC)
        if "T" not in value and " " not in value:
            dt = (
                dt.replace(hour=23, minute=59, second=59)
                if is_end
                else dt.replace(hour=0, minute=0, second=0)
            )
        return dt.replace(tzinfo=UTC)

    def _transform_data(self, raw_data: Any, symbol: str) -> pl.DataFrame:
        """Transform Databento data to standard schema."""
        try:
            # Convert Databento response to DataFrame
            if hasattr(raw_data, "to_df"):
                df_pandas = raw_data.to_df()

                # Databento uses timestamp as DataFrame index
                df_pandas = df_pandas.reset_index()

                # Rename index column to timestamp
                if "index" in df_pandas.columns:
                    df_pandas = df_pandas.rename(columns={"index": "timestamp"})
                elif "ts_event" in df_pandas.columns:
                    df_pandas = df_pandas.rename(columns={"ts_event": "timestamp"})

                df = pl.from_pandas(df_pandas)
            else:
                df = pl.DataFrame(raw_data)

            # Ensure timestamp column exists and is datetime
            if "timestamp" in df.columns:
                if df["timestamp"].dtype == pl.Int64:
                    # Convert nanoseconds to datetime
                    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))

            # Add symbol column
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(symbol).alias("symbol"))

            # For OHLCV data, ensure proper column types
            ohlcv_columns = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort("timestamp")

            # For OHLCV data, select columns in standard order
            required_ohlcv = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_ohlcv):
                # Keep any extra columns after the standard ones
                extra_cols = [c for c in df.columns if c not in required_ohlcv]
                df = df.select(required_ohlcv + extra_cols)

            return df

        except Exception as e:
            self.logger.error("Failed to transform Databento data", error=str(e), symbol=symbol)
            raise DataNotAvailableError(self.name, f"Failed to transform data for {symbol}: {e}")

    def fetch_continuous_futures(
        self,
        root_symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
        version: int = 0,
    ) -> pl.DataFrame:
        """Fetch continuous futures contract data.

        Databento supports continuous futures with the .v.N notation where
        N is the version/roll number (0 = front month).

        Args:
            root_symbol: Root futures symbol (e.g., "ES", "CL")
            start: Start date
            end: End date
            frequency: Data frequency
            version: Contract version (0 = front month, 1 = second month, etc.)

        Returns:
            DataFrame with continuous contract data
        """
        continuous_symbol = f"{root_symbol}.v.{version}"

        self.logger.info(
            "Fetching continuous futures",
            root=root_symbol,
            version=version,
            symbol=continuous_symbol,
        )

        return self.fetch_ohlcv(continuous_symbol, start, end, frequency)

    @staticmethod
    def _response_to_frame(raw: Any) -> pl.DataFrame:
        """Convert a raw Databento response to a polars frame.

        Databento puts the record timestamp on the index, so the ``to_df()`` result is
        ``reset_index()``-ed to materialize it as a column. Falls back to a direct
        ``pl.DataFrame`` for a response object that does not expose ``to_df`` (mirrors the
        defensive branch in ``_transform_data``).
        """
        if hasattr(raw, "to_df"):
            return pl.from_pandas(raw.to_df().reset_index())
        return pl.DataFrame(raw)

    def fetch_option_ohlcv(
        self,
        contract: str,
        start: str,
        end: str,
        frequency: str = "daily",
        *,
        consolidate: bool = True,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars for a single OPRA option contract.

        OPRA is a consolidated feed: each contract gets one bar per reporting venue, so the
        raw response has multiple rows per timestamp. With ``consolidate=True`` (default)
        those per-venue bars are collapsed into one bar per timestamp before normalization
        (high=max, low=min, volume summed, open/close from the highest-volume venue), and an
        extra ``n_venues`` column records how many venues fed each bar. With
        ``consolidate=False`` the per-venue rows are passed through and OHLCV validation
        collapses duplicate timestamps to a single (unspecified) venue's bar — the result is
        the plain canonical OHLCV schema with no ``n_venues`` and no leaked venue metadata.

        Design: this normalizes directly rather than delegating to ``fetch_ohlcv`` because the
        public template cannot carry the OPRA dataset override or the pre-normalization
        consolidation step. It explicitly reuses the same building blocks the template uses
        (input validation, rate limiting, circuit breaker, ``_transform_data``,
        ``_validate_ohlcv``). It deliberately does NOT wrap the fetch in the tenacity
        ``@retry`` that decorates ``fetch_ohlcv``; transient errors surface as
        ``NetworkError``/``RateLimitError`` for the caller to handle. ``self.dataset`` is
        never mutated (the OPRA dataset is passed per call, R2).

        This targets ``dataset="OPRA.PILLAR"`` with ``stype_in="raw_symbol"`` (a single
        contract). The ``contract`` is an OSI 21-char symbol: root padded to 6 + YYMMDD + C/P +
        strike*1000 zero-padded to 8 (e.g. ``"SPX   250321C05800000"`` = SPX, 2025-03-21, call,
        strike 5800). Discover valid contracts with ``fetch_option_chain`` and feed its
        ``raw_symbol`` values here.

        Cost discipline: single-contract OHLCV is cheap (a year of one contract's bars quotes
        at ~$0), but OPRA is the high-volume schema family — call the free ``get_cost_quote`` /
        ``get_billable_size`` before any billed pull as a matter of habit. Never widen this to a
        whole-chain ``trades`` pull (orders of magnitude larger; see ``fetch_option_chain``).

        Args:
            contract: OSI 21-char raw_symbol, e.g. "SPX   250321C05800000".
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            frequency: Bar frequency (daily/hourly/minute), mapped to a Databento schema.
            consolidate: Collapse per-venue bars into one bar per timestamp.

        Returns:
            Canonical OHLCV frame [timestamp, symbol, open, high, low, close, volume] with a
            trailing ``n_venues`` column when consolidated; ``symbol`` is the OSI contract and
            ``timestamp`` is UTC-aware. The empty (well-formed) frame returned when no rows come
            back carries the same columns (including ``n_venues`` under ``consolidate=True``), so
            per-contract slices concatenate cleanly. Unlike ``fetch_option_chain`` /
            ``fetch_option_quotes``, a ``DataNotAvailableError`` (including a no-data condition
            Databento delivers as a client error) PROPAGATES here rather than degrading to an
            empty frame, mirroring the base ``fetch_ohlcv``. NOTE: option OHLCV ``timestamp`` is
            ``Datetime("ns", UTC)`` (the library-wide OHLCV convention), whereas
            ``fetch_option_quotes`` / ``fetch_option_chain`` use ``Datetime("us", UTC)`` — cast
            the join key when joining option OHLCV against quotes/chain.
        """
        self.logger.info(
            "Fetching option OHLCV",
            contract=contract,
            start=start,
            end=end,
            frequency=frequency,
            consolidate=consolidate,
            provider=self.name,
        )

        self._validate_inputs(contract, start, end, frequency)
        self._acquire_rate_limit()

        def _fetch_and_process() -> pl.DataFrame:
            raw = self._fetch_raw_data(
                contract,
                start,
                end,
                frequency,
                dataset=OPRA_DATASET,
                stype_in="raw_symbol",
            )
            df = self._response_to_frame(raw)
            if df.is_empty():
                # Match the populated path's columns so per-contract slices concat cleanly:
                # the consolidated path carries a typed n_venues (UInt32).
                empty = self._create_empty_dataframe()
                if consolidate:
                    empty = empty.with_columns(pl.lit(None, dtype=pl.UInt32).alias("n_venues"))
                return empty
            if consolidate:
                # Collapse venues BEFORE normalization; otherwise _validate_ohlcv's
                # unique(subset=["timestamp"]) would drop all-but-one venue and corrupt volume.
                df = consolidate_publishers(df)
            # _transform_data only auto-renames ts_event->timestamp in its .to_df() branch; a
            # pre-built polars frame skips that, so rename here before normalizing.
            if "ts_event" in df.columns and "timestamp" not in df.columns:
                df = df.rename({"ts_event": "timestamp"})
            # Coerce an epoch-int timestamp (pretty_ts=False) before normalizing, symmetric with
            # the chain/quote paths, so _transform_data never casts it to a tz-naive Datetime.
            if "timestamp" in df.columns:
                df = self._coerce_epoch_datetime(df, "timestamp", label="option ohlcv")
            # Normalize to UTC BEFORE validation so ordering/dedup in _validate_ohlcv operates
            # on a consistent tz-aware dtype throughout the pipeline (not just at the return).
            df = self._to_utc(df, "timestamp")
            normalized = self._transform_data(df, contract)
            validated = self._validate_ohlcv(normalized, self.name)
            # Keep only canonical OHLCV + derived n_venues, dropping the raw per-venue metadata
            # (publisher_id, rtype, instrument_id, ...) that _transform_data passes through.
            keep = [
                c
                for c in (
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "n_venues",
                )
                if c in validated.columns
            ]
            return validated.select(keep)

        return self._with_circuit_breaker(_fetch_and_process)

    @staticmethod
    def _to_utc(df: pl.DataFrame, col: str, *, unit: str | None = None) -> pl.DataFrame:
        """Normalize a Datetime column to UTC, optionally casting its time unit.

        A tz-naive column is interpreted as UTC; a tz-aware one is converted to UTC. When
        ``unit`` is given (e.g. ``"us"``) the column is also cast to that precision so the
        populated frame matches the canonical UTC-microsecond convention of the empty frames.
        Returns the frame unchanged if ``col`` is absent or not a ``Datetime``.
        """
        if col not in df.columns or not isinstance(df.schema[col], pl.Datetime):
            return df
        expr = pl.col(col)
        expr = (
            expr.dt.replace_time_zone("UTC")
            if df.schema[col].time_zone is None
            else expr.dt.convert_time_zone("UTC")
        )
        if unit is not None:
            expr = expr.dt.cast_time_unit(unit)
        return df.with_columns(expr)

    @staticmethod
    def _coerce_epoch_datetime(df: pl.DataFrame, col: str, *, label: str) -> pl.DataFrame:
        """Coerce a non-Datetime epoch/Date column to ``Datetime``, or raise on an odd dtype.

        A drift guard so a shaped frame never silently carries a non-conforming column.
        Integer = raw epoch nanoseconds (``pretty_ts=False``), mirroring the Int64-timestamp
        handling in ``_transform_data``; a ``Date`` casts to midnight. Any other non-Datetime
        dtype is unexpected and raises loudly rather than being misinterpreted (e.g. a
        second/millisecond int read as nanoseconds). Magnitude is NOT validated — the
        ``pretty_ts=True`` default (always a tz-aware datetime) is the real safety net. Returns
        the frame unchanged when ``col`` is already a ``Datetime``.
        """
        dtype = df.schema[col]
        if isinstance(dtype, pl.Datetime):
            return df
        if not (dtype.is_integer() or dtype == pl.Date):
            raise ValueError(f"Unexpected {col} dtype for {label}: {dtype}")
        return df.with_columns(pl.col(col).cast(pl.Datetime("ns")))

    def fetch_option_chain(
        self,
        underlying: str,
        start: str,
        end: str,
        *,
        expiry: str | None = None,
        spot: float | None = None,
        moneyness: float | None = None,
        right: str = "both",
    ) -> pl.DataFrame:
        """Fetch the OPRA option chain (definition records) for an underlying.

        Issues a single ``schema="definition"`` request with ``stype_in="parent"`` against the
        ``<underlying>.OPT`` parent symbol, which returns every listed contract's reference data
        in one call. This is free/cheap metadata — it is NEVER an OHLCV/``trades`` pull (R3).

        The returned frame is the non-OHLCV ``OptionChainSchema`` (raw_symbol, instrument_class,
        strike_price, expiration, plus instrument_id). It deliberately bypasses ``_transform_data``
        and ``_validate_ohlcv`` — those are OHLCV-only and would corrupt or reject a chain frame
        (R1). ``self.dataset`` is never mutated (the OPRA dataset is passed per call, R2).

        The optional filters AND-compose and are applied to the fetched chain. The result is
        sorted by (expiration, strike_price, instrument_class) for a deterministic order. This is
        the discovery layer, not prices: filter the chain, then feed the surviving ``raw_symbol``
        values into ``fetch_option_ohlcv`` for per-contract bars.

        SPX dual-root: SPX index options have TWO roots that are SEPARATE chains — ``SPX``
        (AM-settled monthlies) and ``SPXW`` (PM-settled weeklys / EOM). Query BOTH for full
        coverage; a single root gives only partial coverage.

        Cost discipline: definition records are tiny, so a full chain for a day quotes at ~$0 and
        this method issues ``schema="definition"`` ONLY. It never auto-pulls ``trades`` for a
        whole chain — a single day of all SPX option trades is millions of rows (orders of
        magnitude larger than OHLCV). Keep chain-wide price pulls deliberate and per-contract,
        and check the free ``get_billable_size`` / ``get_cost_quote`` first.

        Args:
            underlying: Underlying root, e.g. "SPX" (AM-settled monthlies) or "SPXW"
                (PM-settled weeklys / EOM). No ".OPT" suffix — the parent symbol is built
                internally.
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            expiry: Keep only this exact expiration date, given as a complete "YYYY-MM-DD"
                string (matched on the calendar date, not as a string prefix).
            spot: Underlying reference price; required when ``moneyness`` is given (option
                definitions carry no underlying price).
            moneyness: Keep strikes within +/- this fraction of ``spot`` (e.g. 0.05 = +/-5%),
                inclusive of both bounds.
            right: "C", "P", or "both" (case-insensitive); "both" keeps calls and puts.

        Returns:
            A non-OHLCV ``OptionChainSchema`` frame with columns [raw_symbol, instrument_class
            ("C"/"P"), strike_price, expiration, instrument_id] — reference data, NOT prices, and
            not to be treated as an OHLCV frame (it bypasses ``_validate_ohlcv``). A well-formed
            empty chain frame when no contracts are available for the underlying/window.
            ``AuthenticationError``/``RateLimitError`` propagate (a missing entitlement or rate
            limit is not "no data"). NOTE: a client-side error (an unknown root, a malformed
            parent symbol, or an invalid schema) also degrades to an empty frame rather than
            raising — double-check ``underlying`` when a chain comes back unexpectedly empty.

        Raises:
            ValueError: If ``moneyness`` is given without ``spot``, if ``expiry`` is not a valid
                "YYYY-MM-DD" date, or if ``right`` is not "C", "P", or "both".
        """
        # Validate filter args up front so a misuse fails before any (free) network call.
        if moneyness is not None and spot is None:
            raise ValueError(
                "moneyness filter requires a spot price; option definitions carry no "
                "underlying price. Pass spot=<underlying price>."
            )
        expiry_date = date.fromisoformat(expiry) if expiry else None
        if right.upper() not in ("C", "P", "BOTH"):
            raise ValueError(
                f"right must be 'C', 'P', or 'both' (case-insensitive); got {right!r}."
            )

        parent = f"{underlying}.OPT"
        self.logger.info(
            "Fetching option chain",
            underlying=underlying,
            parent=parent,
            start=start,
            end=end,
            provider=self.name,
        )
        self._validate_inputs(parent, start, end, "definition")
        self._acquire_rate_limit()

        def _fetch_and_process() -> pl.DataFrame:
            try:
                raw = self._fetch_raw_data(
                    parent,
                    start,
                    end,
                    dataset=OPRA_DATASET,
                    stype_in="parent",
                    schema="definition",
                )
            except DataNotAvailableError:
                # No chain data for this underlying/window -> well-formed empty chain frame.
                # (Auth/rate-limit errors are NOT swallowed; they propagate to the caller.)
                return OptionChainSchema.create_empty(include_optional=True)

            df = self._response_to_frame(raw)
            if df.is_empty():
                return OptionChainSchema.create_empty(include_optional=True)
            shaped = self._shape_option_chain(df)
            return self._filter_option_chain(
                shaped, expiry=expiry_date, spot=spot, moneyness=moneyness, right=right
            )

        return self._with_circuit_breaker(_fetch_and_process)

    @staticmethod
    def _filter_option_chain(
        df: pl.DataFrame,
        *,
        expiry: date | None,
        spot: float | None,
        moneyness: float | None,
        right: str,
    ) -> pl.DataFrame:
        """Apply the expiry / right / moneyness filters to a shaped chain frame.

        Filters AND-compose; each is a no-op when its argument is unset (``right="both"`` keeps
        both rights). ``expiry`` matches on the calendar date of ``expiration`` (not a string
        prefix). The result is sorted by (expiration, strike_price, instrument_class). Filter args
        (``moneyness``-without-``spot``, malformed ``expiry``, unknown ``right``) are validated in
        ``fetch_option_chain`` before this runs.
        """
        f = df
        if expiry is not None:
            f = f.filter(pl.col("expiration").dt.date() == expiry)
        if right.upper() in ("C", "P"):
            f = f.filter(pl.col("instrument_class") == right.upper())
        if moneyness is not None and spot is not None:
            lo, hi = spot * (1 - moneyness), spot * (1 + moneyness)
            f = f.filter(pl.col("strike_price").is_between(lo, hi))
        return f.sort(["expiration", "strike_price", "instrument_class"])

    @staticmethod
    def _shape_option_chain(df: pl.DataFrame) -> pl.DataFrame:
        """Shape a raw Databento definition frame into the OptionChainSchema (non-OHLCV).

        Selects only the chain columns (dropping raw Databento metadata such as rtype /
        publisher_id), casts them to the schema dtypes, normalizes ``expiration`` to a UTC
        microsecond datetime so populated and empty chain frames share one dtype, and collapses
        the per-day definition records Databento emits down to one row per contract.
        """
        if "expiration" in df.columns:
            df = DataBentoProvider._coerce_epoch_datetime(df, "expiration", label="option chain")
            df = DataBentoProvider._to_utc(df, "expiration", unit="us")

        casts = {
            "raw_symbol": pl.Utf8,
            "instrument_class": pl.Utf8,
            "strike_price": pl.Float64,
            "instrument_id": pl.Int64,
        }
        df = df.with_columns([pl.col(c).cast(dt) for c, dt in casts.items() if c in df.columns])

        ordered = [
            c
            for c in (
                "raw_symbol",
                "instrument_class",
                "strike_price",
                "expiration",
                "instrument_id",
            )
            if c in df.columns
        ]
        shaped = df.select(ordered)
        # Collapse to one row per OSI raw_symbol — the per-day definition records are identical
        # for these static fields, so which row survives doesn't matter and the filter re-sorts.
        if "raw_symbol" in shaped.columns:
            shaped = shaped.unique(subset=["raw_symbol"], keep="last", maintain_order=True)
        return shaped

    def fetch_option_quotes(
        self,
        contract: str,
        start: str,
        end: str,
        *,
        schema: str = "cbbo-1m",
    ) -> pl.DataFrame:
        """Fetch consolidated bid/ask quotes for a single OPRA option contract.

        Issues a single per-contract request (``stype_in="raw_symbol"``) against a Databento
        consolidated-quote schema (``cbbo-1m`` default, also ``cbbo-1s``/``tcbbo``/``cmbp-1``).
        OPRA consolidated quotes are a single synthetic book (``publisher_id == 30``), so unlike
        OHLCV there is NO per-venue consolidation — the sampled frame is returned as-is.

        The sampling clock is ``ts_recv`` (the sampling instant); ``ts_event`` is the last
        book-change time and can repeat/stale, so ``ts_recv`` is used when present (falling back
        to ``ts_event``) and renamed to ``timestamp``. The result is sorted ascending by it.

        Schema availability is per-schema and varies a lot (``cbbo-1m`` back to 2013,
        ``cmbp-1``/``tcbbo`` to 2023, ``cbbo-1s`` only to 2025-02-20), so it is checked at runtime
        via the free ``_schema_available_from`` metadata call BEFORE any billed fetch; a ``start``
        earlier than the schema's availability raises ``DataNotAvailableError`` rather than
        silently returning a truncated range.

        This is a non-OHLCV frame: it deliberately bypasses ``_transform_data`` and
        ``_validate_ohlcv`` (which are OHLCV-only and would reject a quote frame, R1).
        ``self.dataset`` is never mutated (the OPRA dataset is passed per call, R2).

        Cost: consolidated quotes are billed but small (~$0.0001 for one contract/day of
        ``cbbo-1m``); the availability metadata call is free. This is per single contract only.
        For the WHOLE chain's quotes, use ``fetch_option_chain_quotes`` (one ``stype_in="parent"``
        request, intraday-windowed and ``max_cost_usd``-guarded) — never auto-loop this
        single-contract method across a chain (that remains a cost trap).

        Args:
            contract: OSI 21-char raw_symbol, e.g. "SPX   250321C05800000".
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            schema: Consolidated-quote schema ("cbbo-1m" default).

        Returns:
            An ``OptionQuoteSchema`` frame [timestamp, bid_px_00, ask_px_00, spread, bid_sz_00,
            ask_sz_00] (only the columns the response carries; ``spread`` is added only when both
            price sides are present). A well-formed empty quote frame when no quotes are
            available for the contract/window. ``AuthenticationError``/``RateLimitError`` and the
            availability ``DataNotAvailableError`` propagate (they are not "no data"). NOTE: a
            client-side error from the fetch (a malformed contract symbol or an invalid schema)
            also degrades to an empty frame rather than raising — verify ``contract`` when a
            populated window comes back empty.

        Raises:
            DataNotAvailableError: If ``schema`` is not available on OPRA as early as ``start``.
        """
        self.logger.info(
            "Fetching option quotes",
            contract=contract,
            start=start,
            end=end,
            schema=schema,
            provider=self.name,
        )
        self._validate_inputs(contract, start, end, schema)

        # Availability gate BEFORE any billed fetch (free metadata). Lexicographic comparison is
        # chronological for "YYYY-MM-DD" ISO strings. avail is None (unknown, or the lookup failed
        # — _schema_available_from logs it) intentionally fails OPEN: proceed to the billed fetch.
        avail = self._schema_available_from(schema, dataset=OPRA_DATASET)
        if avail and start < avail:
            raise DataNotAvailableError(
                self.name,
                f"{schema} is available on OPRA from {avail}; start {start} is too early "
                f"(availability is per-schema: cbbo-1m back to 2013, cbbo-1s only 2025-02-20).",
            )

        self._acquire_rate_limit()

        def _fetch_and_process() -> pl.DataFrame:
            try:
                raw = self._fetch_raw_data(
                    contract,
                    start,
                    end,
                    dataset=OPRA_DATASET,
                    stype_in="raw_symbol",
                    schema=schema,
                )
            except DataNotAvailableError:
                # No quotes for this contract/window -> well-formed empty quote frame.
                # (Auth/rate-limit errors are NOT swallowed; they propagate to the caller.)
                return OptionQuoteSchema.create_empty()

            q = self._response_to_frame(raw)
            if q.is_empty():
                return OptionQuoteSchema.create_empty()
            return self._shape_option_quotes(q)

        return self._with_circuit_breaker(_fetch_and_process)

    @staticmethod
    def _shape_option_quotes(df: pl.DataFrame) -> pl.DataFrame:
        """Shape a raw consolidated-quote frame into the OptionQuoteSchema (non-OHLCV)."""
        # Spread guard: only compute spread when BOTH price columns are present; a sparse
        # schema/response may omit one side, and a blind subtraction would raise.
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            df = df.with_columns((pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread"))

        # Sampling clock is ts_recv; ts_event is the last book-change time (can stale/repeat).
        # A consolidated-quote response always carries one of them; a frame with neither is
        # malformed and would silently drop the required timestamp column, so raise loudly.
        clock = "ts_recv" if "ts_recv" in df.columns else "ts_event"
        if clock not in df.columns:
            raise ValueError(
                "consolidated quote response carried neither a ts_recv nor ts_event timestamp "
                "column; cannot build an OptionQuoteSchema frame."
            )
        df = df.sort(clock).rename({clock: "timestamp"})

        # Coerce a non-Datetime clock (e.g. an epoch-int ts_recv from pretty_ts=False) before
        # normalizing, so we never silently return a non-conforming Int64 timestamp.
        df = DataBentoProvider._coerce_epoch_datetime(df, "timestamp", label="option quotes")
        df = DataBentoProvider._to_utc(df, "timestamp", unit="us")

        # Conform to OptionQuoteSchema: cast the columns present, then back-fill the ones the
        # (sparse) response omitted with typed nulls so a populated frame always validates.
        schema = OptionQuoteSchema.SCHEMA
        df = df.with_columns([pl.col(c).cast(dt) for c, dt in schema.items() if c in df.columns])
        fills = [pl.lit(None, dtype=dt).alias(c) for c, dt in schema.items() if c not in df.columns]
        if fills:
            df = df.with_columns(fills)
        return df.select(list(schema.keys()))

    def fetch_option_chain_quotes(
        self,
        underlying: str,
        start: str,
        end: str,
        *,
        schema: str = "cbbo-1m",
        max_cost_usd: float | None = None,
    ) -> pl.DataFrame:
        """Fetch consolidated bid/ask quotes for an ENTIRE OPRA option chain in one request.

        This is the sanctioned, windowed, cost-guarded way to pull whole-chain quotes — the
        responsible counterpart to ``fetch_option_quotes`` (single contract). It issues a
        SINGLE ``stype_in="parent"`` request against the ``<underlying>.OPT`` parent symbol, so
        every listed contract's consolidated quotes come back together. ``self.dataset`` is
        never mutated (the OPRA dataset is passed per call, R2).

        Intraday windowing is the cost lever. Pass ISO *datetimes* in ``start``/``end`` (e.g.
        ``"2025-03-03T19:50:00"`` .. ``"2025-03-03T20:10:00"``) to bill only the ~20-min window
        around the cash close instead of a whole session day — Databento bills on bytes
        returned, so a full-day whole-chain pull is ~20x a windowed one. Date-only bounds keep
        the whole-day window (see ``_fetch_raw_data``).

        Row identity: a parent ``cbbo`` response identifies each contract by the integer
        ``instrument_id`` (NOT the OSI symbol). The OSI ``raw_symbol`` is resolved by a join
        against the definition chain (``fetch_option_chain``) for the SAME window — OPRA
        ``instrument_id``s roll at session/publication boundaries, so the definition pull must
        cover the window the quotes came from. Each row carries BOTH ``instrument_id`` (always)
        and ``raw_symbol`` (null when an id has no definition match). Rows are NOT deduped across
        ``instrument_id`` — the parent pull interleaves many contracts.

        Availability is checked (free metadata) BEFORE any billed fetch; a ``start`` earlier than
        the schema's availability raises ``DataNotAvailableError`` (``cbbo-1m`` back to 2013,
        ``cbbo-1s`` only 2025-02-20). When ``max_cost_usd`` is given, the query is cost-quoted
        (free) BEFORE billing and a ``CostLimitError`` is raised if the estimate exceeds the cap.
        The cost guard fails CLOSED: if the free cost-metadata call cannot return an estimate
        (e.g. a transient gateway timeout, which surfaces as a non-positive estimate), a
        ``NetworkError`` is raised rather than proceeding to an unguarded billed pull.

        This is a non-OHLCV frame: it deliberately bypasses ``_transform_data`` /
        ``_validate_ohlcv`` (which are OHLCV-only).

        SPX dual-root: SPX index options have TWO roots (``SPX`` AM-settled monthlies and
        ``SPXW`` PM-settled weeklys / EOM) — call once per root for full coverage.

        Args:
            underlying: Underlying root, e.g. "SPX" or "SPXW". No ".OPT" suffix — the parent
                symbol is built internally.
            start: Start bound — ISO-8601 date (whole-day) or datetime (intraday window),
                inclusive.
            end: End bound, same ISO date-or-datetime form as ``start``.
            schema: Consolidated-quote schema ("cbbo-1m" default; also "cbbo-1s"/"tcbbo"/"cmbp-1").
            max_cost_usd: Optional hard cost ceiling in USD. When set, the query is cost-quoted
                before billing and ``CostLimitError`` is raised if the estimate exceeds it.
                ``None`` (default) skips the guard.

        Returns:
            An ``OptionChainQuoteSchema`` frame [timestamp, instrument_id, raw_symbol, bid_px_00,
            ask_px_00, spread, bid_sz_00, ask_sz_00], one row per (contract, sampled timestamp),
            sorted ascending by the sampling clock. A well-formed empty frame when no quotes are
            available for the chain/window. ``AuthenticationError``/``RateLimitError`` and the
            availability/cost ``DataNotAvailableError``/``CostLimitError`` propagate (they are not
            "no data"). NOTE: a client-side error from the billed fetch (a malformed root or an
            invalid schema) degrades to an empty frame — double-check ``underlying``/``schema``
            when a populated window comes back empty.

        Raises:
            DataNotAvailableError: If ``schema`` is not available on OPRA as early as ``start``.
            CostLimitError: If ``max_cost_usd`` is set and the estimate exceeds it.
            NetworkError: If ``max_cost_usd`` is set but the cost estimate cannot be obtained
                (the guard fails closed rather than billing an unverified pull).
        """
        parent = f"{underlying}.OPT"
        self.logger.info(
            "Fetching option chain quotes",
            underlying=underlying,
            parent=parent,
            start=start,
            end=end,
            schema=schema,
            max_cost_usd=max_cost_usd,
            provider=self.name,
        )
        self._validate_inputs(parent, start, end, schema)

        # Availability gate BEFORE any billed fetch (free metadata). Lexicographic comparison is
        # chronological for ISO start strings (the date prefix dominates). avail is None
        # (unknown/lookup failed) fails OPEN: proceed to the billed fetch.
        avail = self._schema_available_from(schema, dataset=OPRA_DATASET)
        if avail and start < avail:
            raise DataNotAvailableError(
                self.name,
                f"{schema} is available on OPRA from {avail}; start {start} is too early "
                f"(availability is per-schema: cbbo-1m back to 2013, cbbo-1s only 2025-02-20).",
            )

        # Cost guard BEFORE any billed fetch (free metadata). Whole-chain quotes are the
        # documented cost trap; max_cost_usd is the opt-in rail. None -> no quote, no guard.
        if max_cost_usd is not None:
            estimated = self.get_cost_quote(
                symbols=parent,
                schema=schema,
                start=start,
                end=end,
                stype_in="parent",
                dataset=OPRA_DATASET,
            )
            # get_cost_quote returns 0.0 when the free metadata call fails (e.g. a 504 gateway
            # timeout) — indistinguishable from a genuine $0. A whole-chain cbbo pull is never
            # genuinely $0, so a non-positive estimate means the cost could NOT be verified. The
            # caller set max_cost_usd precisely to avoid a surprise bill, so fail CLOSED rather
            # than fall through to an unguarded billed pull (failing open is the expensive
            # direction). Callers who do not want this can omit max_cost_usd (opt out of the rail).
            # Note this intentionally treats a legitimate $0 estimate (if Databento ever priced a
            # query at zero) as "unverified" too — an accepted false positive, since a whole-chain
            # cbbo quote is never genuinely free and conflating the two is the safe direction.
            if estimated <= 0.0:
                self.logger.warning(
                    "Cost estimate unavailable; refusing billed fetch under max_cost_usd",
                    parent=parent,
                    schema=schema,
                    max_cost_usd=max_cost_usd,
                    provider=self.name,
                )
                raise NetworkError(
                    provider=self.name,
                    message=(
                        f"could not verify query cost against max_cost_usd=${max_cost_usd:.2f} "
                        "(the free cost-metadata call returned no estimate); refusing the billed "
                        "fetch. Retry, or omit max_cost_usd to bypass the cost guard."
                    ),
                )
            if estimated > max_cost_usd:
                raise CostLimitError(self.name, estimated, max_cost_usd)

        self._acquire_rate_limit()

        def _fetch_and_process() -> pl.DataFrame:
            try:
                raw = self._fetch_raw_data(
                    parent,
                    start,
                    end,
                    dataset=OPRA_DATASET,
                    stype_in="parent",
                    schema=schema,
                )
            except DataNotAvailableError:
                # No quotes for this chain/window -> well-formed empty frame.
                # (Auth/rate-limit errors are NOT swallowed; they propagate to the caller.)
                return OptionChainQuoteSchema.create_empty()

            q = self._response_to_frame(raw)
            if q.is_empty():
                return OptionChainQuoteSchema.create_empty()

            # Resolve OSI raw_symbol via a definition join (cbbo records carry only
            # instrument_id). Pull definitions over the calendar DAY(S) of the window, not the
            # intraday window itself: OPRA emits definition records once at session start
            # (~00:00 UTC), so a late-day window like 19:50-20:10 returns zero definitions. The
            # id->raw_symbol mapping is stable within the day (TASK-004), so the same-day chain
            # is the correct, free (schema="definition") source. start/end are ISO strings, so
            # the date prefix is the first 10 chars (YYYY-MM-DD) for both date and datetime forms.
            chain = self.fetch_option_chain(underlying, start[:10], end[:10])
            return self._shape_chain_quotes(q, chain)

        return self._with_circuit_breaker(_fetch_and_process)

    def _shape_chain_quotes(self, df: pl.DataFrame, chain: pl.DataFrame) -> pl.DataFrame:
        """Shape a raw parent consolidated-quote frame into the OptionChainQuoteSchema.

        Like ``_shape_option_quotes`` (spread guard, ts_recv/ts_event clock, UTC microsecond
        normalization), but carries a per-contract identity: ``instrument_id`` (native to the
        ``cbbo`` response) plus the OSI ``raw_symbol`` resolved by a left-join against the
        definition ``chain`` on ``instrument_id``. Rows are NOT deduped across ``instrument_id``
        — the parent pull interleaves many contracts and each is a distinct row.
        """
        # Spread guard: only compute when BOTH price columns are present (sparse-schema safe).
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            df = df.with_columns((pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread"))

        # Sampling clock is ts_recv; ts_event is the last book-change time (can stale/repeat).
        clock = "ts_recv" if "ts_recv" in df.columns else "ts_event"
        if clock not in df.columns:
            raise ValueError(
                "consolidated quote response carried neither a ts_recv nor ts_event timestamp "
                "column; cannot build an OptionChainQuoteSchema frame."
            )
        df = df.sort(clock).rename({clock: "timestamp"})

        df = DataBentoProvider._coerce_epoch_datetime(df, "timestamp", label="option chain quotes")
        df = DataBentoProvider._to_utc(df, "timestamp", unit="us")

        # Resolve raw_symbol by a left-join on instrument_id (TASK-004 decision: definition join,
        # not map_symbols — a parent pull resolves map_symbols to the useless parent string). An
        # empty/partial chain leaves raw_symbol null; a row without a definition match keeps its
        # instrument_id (do not drop it). Back-fill a typed-null raw_symbol if either side lacks
        # the join key so the conform step below always produces the column.
        if "instrument_id" in df.columns and {"instrument_id", "raw_symbol"} <= set(chain.columns):
            # Cast the join key to Int64 on BOTH sides so the join never relies on Polars
            # coercing mismatched integer dtypes (a cbbo response carries instrument_id as
            # uint32, while the definition chain's key is Int64 per OptionChainSchema) — that
            # coercion is version-dependent and stricter Polars raises a join-key schema error.
            df = df.with_columns(pl.col("instrument_id").cast(pl.Int64))
            id_map = (
                chain.select(["instrument_id", "raw_symbol"])
                .with_columns(pl.col("instrument_id").cast(pl.Int64))
                .unique(subset=["instrument_id"])
            )
            df = df.join(id_map, on="instrument_id", how="left")
        elif "raw_symbol" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("raw_symbol"))

        # Conform to OptionChainQuoteSchema: cast present columns, back-fill missing with typed
        # nulls, then select in schema order.
        schema = OptionChainQuoteSchema.SCHEMA
        df = df.with_columns([pl.col(c).cast(dt) for c, dt in schema.items() if c in df.columns])
        fills = [pl.lit(None, dtype=dt).alias(c) for c, dt in schema.items() if c not in df.columns]
        if fills:
            df = df.with_columns(fills)
        return df.select(list(schema.keys()))

    def fetch_multiple_schemas(
        self,
        symbol: str,
        start: str,
        end: str,
        schemas: list[str],
    ) -> dict[str, pl.DataFrame]:
        """Fetch data for multiple schemas at once.

        Args:
            symbol: Symbol to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            schemas: List of schemas to fetch (e.g., ["ohlcv-1m", "trades"])

        Returns:
            Dictionary mapping schema names to DataFrames
        """
        results = {}
        for schema in schemas:
            # Map schema back to frequency
            if schema == "ohlcv-1d":
                frequency = "daily"
            elif schema == "ohlcv-1h":
                frequency = "hourly"
            elif schema == "ohlcv-1m":
                frequency = "minute"
            elif schema == "trades":
                frequency = "trades"
            elif schema == "tbbo":
                frequency = "quotes"
            else:
                frequency = schema

            try:
                # Use low-level methods to avoid OHLCV validation for non-OHLCV schemas
                raw_data = self._fetch_raw_data(symbol, start, end, frequency)
                df = self._transform_data(raw_data, symbol)
                results[schema] = df
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch schema",
                    schema=schema,
                    symbol=symbol,
                    error=str(e),
                )
                results[schema] = None

        return results

    def get_available_datasets(self) -> list[str]:
        """Get list of available datasets.

        Returns:
            List of dataset names (e.g., ["GLBX.MDP3", "XNAS.ITCH"])
        """
        try:
            return self.client.metadata.list_datasets()
        except Exception as e:
            self.logger.error("Failed to list datasets", error=str(e))
            return []

    def get_available_schemas(self, dataset: str | None = None) -> list[str]:
        """Get list of available schemas for a dataset.

        Args:
            dataset: Dataset name (uses self.dataset if not provided)

        Returns:
            List of schema names (e.g., ["ohlcv-1m", "trades", "tbbo"])
        """
        dataset = dataset or self.dataset
        try:
            return self.client.metadata.list_schemas(dataset=dataset)
        except Exception as e:
            self.logger.error("Failed to list schemas", dataset=dataset, error=str(e))
            return []

    def get_billable_size(
        self,
        *,
        symbols: list[str] | str,
        schema: str,
        start: str,
        end: str,
        stype_in: str = "raw_symbol",
        dataset: str = OPRA_DATASET,
    ) -> int:
        """Free metadata: billable bytes for a query (no data is billed).

        Use this to size a query before committing to a billed fetch. NOTE: ``dataset`` defaults
        to OPRA (``OPRA.PILLAR``), NOT the provider's configured ``self.dataset`` — pass
        ``dataset=`` explicitly to size a futures/equities (e.g. GLBX.MDP3) query.

        Args:
            symbols: Symbol(s) to query.
            schema: Databento schema (e.g. "ohlcv-1d", "definition").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            stype_in: Symbology type ("raw_symbol" for a single contract, "parent"
                for a whole chain).
            dataset: Dataset to query (defaults to the OPRA options feed).

        Returns:
            Billable size in bytes, or 0 if the metadata call fails.
        """
        try:
            return self.client.metadata.get_billable_size(
                dataset=dataset,
                symbols=symbols,
                stype_in=stype_in,
                schema=schema,
                start=start,
                end=end,
            )
        except Exception as e:
            self.logger.error(
                "Failed to get billable size", error=str(e), dataset=dataset, schema=schema
            )
            return 0

    def get_cost_quote(
        self,
        *,
        symbols: list[str] | str,
        schema: str,
        start: str,
        end: str,
        stype_in: str = "raw_symbol",
        dataset: str = OPRA_DATASET,
    ) -> float:
        """Free metadata: estimated cost in USD for a query (no data is billed).

        Wraps the SDK's ``metadata.get_cost`` (named ``get_cost_quote`` here to avoid
        colliding with the SDK method name). The SDK feed ``mode`` is left at its default. NOTE:
        ``dataset`` defaults to OPRA (``OPRA.PILLAR``), NOT the provider's configured
        ``self.dataset`` — pass ``dataset=`` explicitly to quote a futures/equities query.

        Args:
            symbols: Symbol(s) to query.
            schema: Databento schema (e.g. "ohlcv-1d", "definition").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            stype_in: Symbology type ("raw_symbol" or "parent").
            dataset: Dataset to query (defaults to the OPRA options feed).

        Returns:
            Estimated cost in USD, or 0.0 if the metadata call fails.
        """
        try:
            return self.client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                stype_in=stype_in,
                schema=schema,
                start=start,
                end=end,
            )
        except Exception as e:
            self.logger.error(
                "Failed to get cost quote", error=str(e), dataset=dataset, schema=schema
            )
            return 0.0

    def _schema_available_from(self, schema: str, *, dataset: str = OPRA_DATASET) -> str | None:
        """Earliest 'YYYY-MM-DD' a schema is available on ``dataset``.

        Per-schema availability varies a lot (e.g. cbbo-1m goes back to 2013 but cbbo-1s
        only to 2025), so never hardcode a single date for a family of schemas. This is
        advisory only — the isinstance guards degrade any unexpected payload shape (the
        ``get_dataset_range`` schema has drifted across SDK versions) to None, and a broad
        except degrades SDK/network failures to None too, matching the never-raise posture
        of the sibling metadata helpers.

        Args:
            schema: Schema name to look up (e.g. "cbbo-1m").
            dataset: Dataset to query (defaults to the OPRA options feed).

        Returns:
            The earliest available date as 'YYYY-MM-DD', or None if unknown. Successful
            (non-None) results are memoized per ``(dataset, schema)`` for the session, so a
            per-contract loop issues at most one ``get_dataset_range`` per schema; a None
            (unknown/failed) result is not cached, so it is retried on the next call.
        """
        cache_key = (dataset, schema)
        if cache_key in self._availability_cache:
            return self._availability_cache[cache_key]
        try:
            rng = self.client.metadata.get_dataset_range(dataset=dataset)
            sch = rng.get("schema") if isinstance(rng, dict) else None
            entry = sch.get(schema) if isinstance(sch, dict) else None
            start = entry.get("start") if isinstance(entry, dict) else None
            result = start[:10] if isinstance(start, str) else None
        except Exception as e:
            self.logger.error(
                "Failed to get schema availability", error=str(e), dataset=dataset, schema=schema
            )
            return None
        if result is not None:
            self._availability_cache[cache_key] = result
        return result
