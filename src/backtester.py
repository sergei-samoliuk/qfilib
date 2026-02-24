import calendar
import datetime as dt
import logging
from collections import defaultdict
from enum import Enum
from typing import Optional

import luigi
import numpy as np
import pandas as pd
from moexalgo import CandlePeriod

from config import LUIGI_OUTPUT_DIR
from src.auction_model import AuctionModelInference, RecalibrationMode, TargetType
from src.graph import InMemoryTask, DailyMixin
from src.qr import get_mos_dates_range
from src.tasks import AuctionResultsHistory, BondsDescription, BondMetricsAtVWAP, BondType, Board, Candles

log = logging.getLogger(__name__)


class ADV(InMemoryTask, DailyMixin):

    def requires(self):
        return [self.clone(Candles, date=date, board=Board.TQOB, period=CandlePeriod.ONE_HOUR) for date in get_mos_dates_range(self.prev_date, 10)]

    def produce_output(self):
        value_df = pd.concat(t.read_output()['volume'].unstack().T.between_time(dt.time(10), dt.time(19)) for t in self.requires())
        return value_df.groupby(value_df.index.to_series().dt.date).sum().mean().to_frame('adv') * 1e3

class Side(Enum):
    BUY = 1
    SELL = -1

    def worsen_price(self, price: float, offset: float) -> float:
        return price + offset * self.value


class CostModel:
    def cost_bps(self, years_to_maturity: float, adv: float, size: float) -> float:
        assert size > 0
        lots = size / 1000
        if years_to_maturity >= 10:
            if lots >= 500_000: return 3.
            elif lots >= 350_000: return 2.
            elif lots >= 100_000: return 1.5
            else: return 1.
        else:
            if lots >= 500_000: return 4.
            elif lots >= 350_000: return 3.
            elif lots >= 100_000: return 2.
            else: return 1.


class MultCostModel:
    def __init__(self, base_model: CostModel, mult: float):
        self.base_model = base_model
        self.mult = mult

    def cost_bps(self, years_to_maturity: float, adv: float, size: float) -> float:
        return self.base_model.cost_bps(years_to_maturity, adv, size) * self.mult



class Portfolio:

    def __init__(self):
        self.positions: dict[str, float] = defaultdict(float)
        self.cash: float = 0.0
        self.trades: list[dict] = []

    def get_position(self, secid: str) -> float:
        return self.positions.get(secid, 0.0)

    def snapshot_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def capture_trade(self, date: dt.date, secid: str, size: float, price: float, source: str):
        self.positions[secid] += size
        if abs(self.positions[secid]) < 1:
            self.positions.pop(secid)
        value = size * price / 100
        self.cash -= value
        self.trades.append({
            'date': date, 'secid': secid,
            'size': size, 'price': price,
            'value': value, 'source': source,
        })

    def market_value(self, price_map: dict[str, float]) -> float:
        mv = self.cash
        for secid, size in self.positions.items():
            mv += size * price_map[secid] / 100
        return mv

    def calc_dv01(self, universe_df: pd.DataFrame) -> float:
        return sum(
            size * universe_df.loc[secid, 'dv01']
            for secid, size in self.positions.items())


class CandleProcessor:

    def __init__(self, candles_df: pd.DataFrame):
        df = candles_df.copy()
        df['pxv'] = df['weighted_avg'] * df['volume']
        grouped = df.groupby('secid')[['pxv', 'volume']].sum()
        self.vwap: pd.Series = (grouped['pxv'] / grouped['volume']).rename('vwap')
        self.close: pd.Series = candles_df.reset_index().groupby('secid')['weighted_avg'].last()


class AuctionPriceSource:
    """Loads full auction history once; provides price lookups and auctioned-secid resolution."""

    def __init__(self):
        luigi.build([AuctionResultsHistory(out_dir=LUIGI_OUTPUT_DIR)], local_scheduler=True)
        data = AuctionResultsHistory(out_dir=LUIGI_OUTPUT_DIR).read_output()
        self._data = data[['date', 'series', 'average_price']].dropna().copy()
        self._data['date'] = self._data['date'].dt.date

    def get_price(self, date: dt.date, series: str) -> Optional[float]:
        rows = self._data[(self._data['date'] == date) & (self._data['series'] == series)]
        return float(rows['average_price'].iloc[0]) if not rows.empty else None

    def get_auctioned_secids(self, date: dt.date, universe_df: pd.DataFrame) -> set[str]:
        """Returns secids of bonds that went to auction today, filtered to current FIX universe."""
        auctioned_series = set(self._data.loc[self._data['date'] == date, 'series'])
        return set(universe_df[universe_df['series'].isin(auctioned_series)].index)


class LOBExecutor:
    """Fills at daily VWAP adjusted for market impact."""

    def __init__(self, cost_model):
        self.cost_model = cost_model
        self._vwap: Optional[pd.Series] = None

    def set_vwap(self, vwap: pd.Series):
        self._vwap = vwap

    def execute(self, size: float, side: Side, secid: str, bond_info: pd.Series) -> float:
        vwap = self._vwap.loc[secid]
        cost_bps = self.cost_model.cost_bps(bond_info['years_to_maturity'], bond_info['adv'], abs(size))
        cost_price = vwap * bond_info['modified_duration'] * cost_bps * 1e-4
        return side.worsen_price(vwap, cost_price)


class DelegatingExecutor:
    """Routes buy orders to auction when available; everything else goes to the LOB."""

    def __init__(self, cost_model):
        self.auction = AuctionPriceSource()
        self.lob = LOBExecutor(cost_model)

    def set_vwap(self, vwap: pd.Series):
        self.lob.set_vwap(vwap)

    def get_auctioned_secids(self, date: dt.date, universe_df: pd.DataFrame) -> set[str]:
        return self.auction.get_auctioned_secids(date, universe_df)

    def execute(self, date: dt.date, size: float, secid: str, bond_info: pd.Series) -> tuple[float, str]:
        side = Side.BUY if size > 0 else Side.SELL
        if side == Side.BUY:
            auction_price = self.auction.get_price(date, bond_info['series'])
            if auction_price is not None:
                return auction_price, 'auction'
        return self.lob.execute(size, side, secid, bond_info), 'lob'


class OnTheRunBondsStrategy:
    """
    Produces a short-side target for ALL on-the-run bonds based on auction model inference.
    Returns None on non-action days so the backtester skips rebalancing entirely.
    """

    def __init__(self, action_by_date: dict[calendar.Day, str], short_notionals: list[float]):
        self.action_by_date = action_by_date
        self.short_notionals = short_notionals

    def build_short_target(
        self, date: dt.date, universe_df: pd.DataFrame
    ) -> Optional[tuple[dict[str, float], pd.DataFrame]]:
        """
        Returns (short_target, inference_df) where short_target maps secid → target size (≤ 0) for all onr bonds.
        """
        action = self.action_by_date.get(date.weekday())
        if action is None:
            return None

        inference = AuctionModelInference(
            date=date, out_dir=LUIGI_OUTPUT_DIR, time_of_day=dt.timedelta(hours=9),
            recalibration_mode=RecalibrationMode.QUARTERLY,
            target_type=TargetType.FIRST, loss_function='QuerySoftMax',
            eval_metric='QuerySoftMax',
        )
        luigi.build([inference], local_scheduler=True)
        inference_df = inference.read_output().set_index('secid')

        if action == 'model_target':
            inf_sorted = inference_df.sort_values(
                ['score', 'years_to_maturity'], ascending=[False, False]
            )
            top = inf_sorted.head(len(self.short_notionals)).index.tolist()
        elif action == 'true_target':
            top = (
                inference_df.query('target == 1')
                .sort_values('years_to_maturity', ascending=False)
                .index.tolist()
            )
        else:
            assert action == 'close'
            top = []

        short_target: dict[str, float] = {}
        for secid in inference_df.index:
            if secid not in universe_df.index:
                continue
            if  not universe_df.loc[secid, 'has_adv']:
                # log.warning('ignoring %s as no adv on %s', secid, date)
                continue
            size = self.short_notionals[top.index(secid)] if secid in top else 0.0
            short_target[secid] = -size

        return short_target, inference_df

class BaseHedgeTargetBuilder:
    """
    Shared base for hedge target builders.

    Provides:
      - Constructor injection of portfolio (stable across backtest).
      - Daily universe refresh via set_universe().
      - Instrument-finding helpers used by both hedge strategies.
    """

    def __init__(self, portfolio: Portfolio):
        self._portfolio = portfolio
        self._universe_df: Optional[pd.DataFrame] = None

    def set_universe(self, universe_df: pd.DataFrame):
        self._universe_df = universe_df

    def _find_auction_hedge(
        self,
        auctioned_secids: set[str],
        excluded_secids: set[str],
        inference_df: pd.DataFrame,
        universe_df: pd.DataFrame,
    ) -> Optional[str]:
        """
        Best auction candidate: highest remaining_size_mln among auctioned bonds
        that are not excluded (i.e. not in the new short list), present in the
        FIX universe, and have a valid DV01.
        """
        candidates = [
            s for s in auctioned_secids
            if s not in excluded_secids
            and s in universe_df.index
            and s in inference_df.index
            and not np.isnan(universe_df.loc[s, 'dv01'])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: inference_df.loc[s, 'remaining_size_mln'])

    def _find_inference_hedge(
        self,
        inference_df: pd.DataFrame,
        excluded_secids: set[str],
        universe_df: pd.DataFrame,
    ) -> Optional[str]:
        """
        Best non-auction candidate: lowest model score among on-the-run bonds
        with remaining_size_mln ≥ 10, not excluded, present in the FIX universe.
        """
        mask = (
            (~inference_df.index.isin(excluded_secids))
            & (inference_df['remaining_size_mln'] >= 10)
            & (inference_df.index.isin(universe_df.index))
        )
        candidates = inference_df[mask].copy()
        valid = [s for s in candidates.index if not np.isnan(universe_df.loc[s, 'dv01'])]
        if not valid:
            return None
        return candidates.loc[valid].sort_values('score').index[0]

    def _flatten_all(self, short_target: dict[str, float]) -> dict[str, float]:
        """Last-resort: flatten all positions rather than run unhedged."""
        log.warning("No hedge instrument found — flattening all positions")
        return {
            secid: 0.0
            for secid in set(short_target) | set(self._portfolio.positions)
        }

    # ------------------------------------------------------------------

    def _find_auction_hedge(
        self,
        auctioned_secids: set[str],
        new_short_secids: set[str],
        inference_df: pd.DataFrame,
        universe_df: pd.DataFrame,
    ) -> Optional[str]:
        candidates = [
            s for s in auctioned_secids
            if s not in new_short_secids
            and s in universe_df.index
            and s in inference_df.index
            and not np.isnan(universe_df.loc[s, 'dv01'])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: inference_df.loc[s, 'remaining_size_mln'])

    def _find_inference_hedge(
        self,
        inference_df: pd.DataFrame,
        new_short_secids: set[str],
        universe_df: pd.DataFrame,
    ) -> Optional[str]:
        mask = (
            (~inference_df.index.isin(new_short_secids))
            & (inference_df['remaining_size_mln'] >= 50)
            & (inference_df.index.isin(universe_df.index))
        )
        candidates = inference_df[mask].copy()
        valid = [s for s in candidates.index if universe_df.loc[s, 'has_adv']]
        if not valid:
            return None
        return candidates.loc[valid].sort_values('score').index[0]


class HedgeTargetBuilder(BaseHedgeTargetBuilder):
    """
    Naive hedge: always buys a fresh hedge instrument sized to offset the full
    short-book DV01, ignoring any existing long positions.

    Priority for the hedge instrument:
      1. Auctioned bond with highest remaining_size_mln (not in new shorts)
      2. Inference-universe bond with smallest model score, remaining_size_mln ≥ 10
      3. Flatten everything if neither found

    Also closes positions in bonds that have left the inference universe.
    """

    def build(
        self,
        short_target: dict[str, float],
        inference_df: pd.DataFrame,
        auctioned_secids: set[str],
    ) -> dict[str, float]:
        universe_df = self._universe_df
        portfolio = self._portfolio
        hedge_target: dict[str, float] = {}

        # Close positions in bonds that have fallen out of the inference universe
        for secid in portfolio.snapshot_positions():
            if secid not in inference_df.index:
                hedge_target[secid] = 0.0

        required_long_dv01 = -sum(
            qty * universe_df.loc[secid, 'dv01']
            for secid, qty in short_target.items()
            if secid in universe_df.index
        )

        if required_long_dv01 <= 0:
            return hedge_target

        new_short_secids = {s for s, q in short_target.items() if q < 0}

        hedge_secid = (
            self._find_auction_hedge(auctioned_secids, new_short_secids, inference_df, universe_df)
            or self._find_inference_hedge(inference_df, new_short_secids, universe_df)
        )

        if hedge_secid is not None:
            hedge_target[hedge_secid] = required_long_dv01 / universe_df.loc[hedge_secid, 'dv01']
        else:
            hedge_target.update(self._flatten_all(short_target))

        return hedge_target


class IncrementalHedgeTargetBuilder(BaseHedgeTargetBuilder):
    """
    Incremental hedge: reuses existing long positions where possible to avoid
    unnecessary turnover, only trading the marginal DV01 gap.

    Logic:
      - Existing longs still present in inference_df (still on-the-run) are
        retained. They are trimmed if they overhedge, or topped up if they
        underhedge, rather than being closed and replaced.
      - Existing longs that have left inference_df are closed (stale off-run).
      - When trimming overhedge: reduce longs in ascending remaining_size_mln
        order (least attractive first) until DV01-neutral.
      - When topping up underhedge: buy the same priority of hedge instrument
        as HedgeTargetBuilder (auction first, then inference universe).
    """

    def build(
        self,
        short_target: dict[str, float],
        inference_df: pd.DataFrame,
        auctioned_secids: set[str],
    ) -> dict[str, float]:
        universe_df = self._universe_df
        portfolio = self._portfolio
        hedge_target: dict[str, float] = {}

        new_short_secids = {s for s, q in short_target.items() if q < 0}

        # Partition existing longs (not part of the short target) into
        # retained (still on-the-run) and stale (fell off inference universe).
        existing_longs = {
            secid: qty
            for secid, qty in portfolio.snapshot_positions().items()
            if qty > 0 and secid not in new_short_secids
        }
        retained = {
            secid: qty for secid, qty in existing_longs.items()
            if secid in inference_df.index and secid in universe_df.index
        }
        stale = {secid for secid in existing_longs if secid not in retained}

        for secid in stale:
            hedge_target[secid] = 0.0

        required_long_dv01 = -sum(
            qty * universe_df.loc[secid, 'dv01']
            for secid, qty in short_target.items()
            if secid in universe_df.index
        )

        if required_long_dv01 <= 0:
            # No longs needed at all: close all retained longs too
            for secid in retained:
                hedge_target[secid] = 0.0
            return hedge_target

        retained_dv01 = sum(
            qty * universe_df.loc[secid, 'dv01']
            for secid, qty in retained.items()
        )

        gap_dv01 = required_long_dv01 - retained_dv01

        if gap_dv01 < 0:
            # Overhedged: trim retained longs, cheapest-to-hold last
            # (lowest remaining_size_mln trimmed first).
            hedge_target.update(self._trim_overhedge(retained, -gap_dv01, inference_df, universe_df))
        elif gap_dv01 > 0:
            # Underhedged: buy additional hedge instrument for the gap.
            # Exclude new shorts; retained longs are valid top-up candidates.
            hedge_secid = (
                self._find_auction_hedge(auctioned_secids, new_short_secids, inference_df, universe_df)
                or self._find_inference_hedge(inference_df, new_short_secids, universe_df)
            )
            if hedge_secid is not None:
                additional_qty = gap_dv01 / universe_df.loc[hedge_secid, 'dv01']

                # if it is existing long - top - up
                current_qty = max(portfolio.get_position(hedge_secid), 0)
                hedge_target[hedge_secid] = current_qty + additional_qty
            else:
                hedge_target.update(self._flatten_all(short_target))

        hedge_target = {**retained, **hedge_target}

        return hedge_target

    def _trim_overhedge(
        self,
        retained: dict[str, float],
        excess_dv01: float,
        inference_df: pd.DataFrame,
        universe_df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Reduce retained longs in ascending remaining_size_mln order until
        excess_dv01 is absorbed. Returns only the longs whose target qty
        changes (partial trims and full closes).
        """
        trim_target: dict[str, float] = {}
        sorted_longs = sorted(
            retained.items(),
            key=lambda x: inference_df.loc[x[0], 'remaining_size_mln']
        )
        for secid, qty in sorted_longs:
            if excess_dv01 <= 0:
                break
            dv01_per_unit = universe_df.loc[secid, 'dv01']
            pos_dv01 = qty * dv01_per_unit
            if pos_dv01 <= excess_dv01:
                trim_target[secid] = 0.0
                excess_dv01 -= pos_dv01
            else:
                trim_qty = excess_dv01 / dv01_per_unit
                trim_target[secid] = qty - trim_qty
                excess_dv01 = 0.0
        return trim_target

# ---------------------------------------------------------------------------
# Metrics Calculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Decomposes daily P&L into three components:

      holding_pnl      — (close_t − close_{t-1}) / 100 × position_sod
                         Captures mark-to-market drift on existing positions.

      auction_exec_pnl — (close_t − auction_price) / 100 × size  [buys only]
                         Usually positive: buying at auction is typically cheaper than close.

      lob_exec_pnl     — (close_t − lob_price) / 100 × size
                         Usually negative: LOB fills are worse than VWAP/close.

    Together these reconcile to the change in equity:
      Δequity = holding_pnl + auction_exec_pnl + lob_exec_pnl

    close_prices are set daily via set_close_prices() before record() is called.
    """

    def __init__(self, portfolio: Portfolio):
        self._portfolio = portfolio
        self._history: list[dict] = []
        self._prev_close: dict[str, float] = {}
        self._close: dict[str, float] = {}

    def set_close_prices(self, close: pd.Series, fallback_close: dict[str, float]):
        self._close = {**fallback_close, **close.to_dict()}
        assert all(np.isfinite(p) for p in self._close.values())

    def record(self, date: dt.date, positions_sod: dict[str, float], universe_df: pd.DataFrame):
        portfolio = self._portfolio
        close_map = self._close
        today_trades = [t for t in portfolio.trades if t['date'] == date]

        holding_pnl = sum(
            size * (close_map[secid] - self._prev_close[secid]) / 100
            for secid, size in positions_sod.items()
        )

        auction_exec_pnl = sum(
            t['size'] * (close_map[t['secid']] - t['price']) / 100
            for t in today_trades
            if t['source'] == 'auction'
        )

        lob_exec_pnl = sum(
            t['size'] * (close_map[t['secid']] - t['price']) / 100
            for t in today_trades
            if t['source'] == 'lob'
        )

        net_dv01, longs_dv01, shorts_dv01, gross_notional, longs_notional, shorts_notional = (self._compute_risk(universe_df))

        self._history.append({
            'date': date,
            'equity': portfolio.market_value(close_map),
            'holding_pnl': holding_pnl,
            'auction_exec_to_close_pnl': auction_exec_pnl,
            'lob_exec_to_close_pnl': lob_exec_pnl,
            'close_map': close_map,
            'today_trades': today_trades,
            'net_dv01': net_dv01,
            'longs_dv01': longs_dv01,
            'shorts_dv01': shorts_dv01,
            'gross_notional': gross_notional,
            'longs_notional': longs_notional,
            'shorts_notional': shorts_notional,
        })

        self._prev_close = close_map

    def get_history(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._history)
        frame['date'] = pd.to_datetime(frame['date'])
        return frame.set_index('date')

    def _compute_risk(self, universe_df: pd.DataFrame):
        net_dv01 = longs_dv01 = shorts_dv01 = 0.0
        gross_notional = longs_notional = shorts_notional = 0.0
        for secid, size in self._portfolio.positions.items():
            dv01 = universe_df.loc[secid, 'dv01']
            net_dv01 += size * dv01
            longs_dv01 += max(size, 0) * dv01
            shorts_dv01 += min(size, 0) * dv01
            gross_notional += abs(size)
            longs_notional += max(size, 0)
            shorts_notional += min(size, 0)
        return net_dv01, longs_dv01, shorts_dv01, gross_notional, longs_notional, shorts_notional


# ---------------------------------------------------------------------------
# Daily Data Loader
# ---------------------------------------------------------------------------

class DailyDataLoader:
    """
    Loads and prepares all market data for a single date.
    Centralises luigi task execution and universe construction so the
    backtester's run loop stays pure orchestration.
    """

    CANDLE_START = dt.time(10)
    CANDLE_END = dt.time(19)

    def load(self, date: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (universe_df, candles_df) ready for consumption.

        universe_df index: secid, columns include modified_duration, dv01,
                           years_to_maturity, adv, series.
        candles_df:        hourly candles filtered to the trading window,
                           with weighted_avg and volume columns.
        """
        tasks = [
            BondsDescription(out_dir=LUIGI_OUTPUT_DIR, date=date),
            Candles(out_dir=LUIGI_OUTPUT_DIR, date=date, period=CandlePeriod.ONE_HOUR, board=Board.TQOB),
            BondMetricsAtVWAP(out_dir=LUIGI_OUTPUT_DIR, date=date, bond_type=BondType.FIX),
            ADV(out_dir=LUIGI_OUTPUT_DIR, date=date),
        ]

        luigi.build(tasks, local_scheduler=True)
        universe_df, candles_df, metrics_df, adv_df = (t.read_output() for t in tasks)

        candles_df = candles_df[candles_df['end'].dt.time.between(self.CANDLE_START, self.CANDLE_END)].copy()

        universe_df = universe_df[universe_df['SHORT_TYPE'] == BondType.FIX.value].copy()
        universe_df['years_to_maturity'] = (universe_df['MATDATE'] - pd.to_datetime(date)) / pd.Timedelta(days=365)
        metrics_df['dv01'] = ((metrics_df['price'] / 100) * metrics_df['modified_duration'] * 1e-4)
        universe_df = universe_df.join(metrics_df[['modified_duration', 'dv01']]).join(adv_df)
        universe_df['has_adv'] = universe_df['adv'] > 0
        universe_df = universe_df.sort_values('years_to_maturity')
        universe_df[['modified_duration', 'dv01']] = universe_df[['modified_duration', 'dv01']].ffill()

        return universe_df, candles_df


class AuctionBacktester:
    """
    Orchestrates the backtest loop.

    Each action day:
      1. Process candles → distribute VWAP to executor, close prices to metrics.
      2. Build full portfolio target (shorts + hedge) in one pass.
      3. Compute per-instrument deltas against current positions.
      4. Execute a single trade per instrument (auction or LOB, auto-routed).
      5. Check DV01 neutrality.
      6. Record metrics.
    """

    def __init__(
        self,
        strategy: OnTheRunBondsStrategy,
        executor: DelegatingExecutor,
        loader: DailyDataLoader,
        hedge_class: BaseHedgeTargetBuilder,
    ):
        self.strategy = strategy
        self.executor = executor
        self.loader = loader
        self.portfolio = Portfolio()
        self.hedge_builder = hedge_class(self.portfolio)
        self.metrics = MetricsCalculator(self.portfolio)

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_full_target(
        self, date: dt.date, universe_df: pd.DataFrame
    ) -> Optional[dict[str, float]]:
        result = self.strategy.build_short_target(date, universe_df)
        if result is None:
            return None

        short_target, inference_df = result
        auctioned_secids = self.executor.get_auctioned_secids(date, universe_df)
        hedge_target = self.hedge_builder.build(short_target, inference_df, auctioned_secids)

        # hedge_target wins on overlap
        return {**short_target, **hedge_target}

    def _compute_deltas(self, full_target: dict[str, float]) -> dict[str, float]:
        """One delta per instrument across both current positions and new target."""
        all_secids = set(full_target) | set(self.portfolio.positions)
        deltas = {secid: full_target.get(secid, 0.0) - self.portfolio.get_position(secid) for secid in all_secids}
        return {secid: delta for secid, delta in deltas.items() if abs(delta) >= 1}

    def _execute_deltas(
        self, date: dt.date, deltas: dict[str, float], universe_df: pd.DataFrame
    ):
        for secid, delta in deltas.items():
            bond_info = universe_df.loc[secid]
            exec_price, source = self.executor.execute(date, delta, secid, bond_info)
            self.portfolio.capture_trade(date, secid, delta, exec_price, source)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, dates: list[dt.date]) -> pd.DataFrame:

        for date in dates:
            universe_df, candles_df = self.loader.load(date)

            candles = CandleProcessor(candles_df)

            self.executor.set_vwap(candles.vwap)
            self.metrics.set_close_prices(candles.close, {s: self.executor.auction.get_price(date, universe_df.loc[s]['series']) for s in self.executor.get_auctioned_secids(date, universe_df)})
            self.hedge_builder.set_universe(universe_df)

            positions_sod = self.portfolio.snapshot_positions()

            full_target = self._build_full_target(date, universe_df)
            if full_target is not None:
                deltas = self._compute_deltas(full_target)
                self._execute_deltas(date, deltas, universe_df)

                net_dv01 = self.portfolio.calc_dv01(universe_df)
                if abs(net_dv01) >= 1:
                    log.error("DV01 not fully neutral on %s: net_dv01=%.4f", date, net_dv01)

            self.metrics.record(date, positions_sod, universe_df)

        return self.metrics.get_history()