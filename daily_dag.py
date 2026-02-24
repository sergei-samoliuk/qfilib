import datetime as dt

import luigi
from moexalgo import CandlePeriod

import config
from config import LUIGI_OUTPUT_DIR as OUT_DIR
from src.auction_model import AuctionModelEmail, AuctionModelCalibration
from src.qr import get_mos_dates_range, is_mos_business_date
from src.tasks import Board, Candles, GCurve, BondsBoards, BondType, YieldCandles, BondMetricsAtVWAP, AuctionDates, \
    GCurveParams

luigi.interface.core.log_level = "INFO"
config.setup_proxies()

def filter_auction_dates(dates):
    luigi.build([AuctionDates(out_dir=OUT_DIR)], local_scheduler=True)
    auction_dates = AuctionDates(out_dir=OUT_DIR).read_output()['date'].dt.date
    intersection = set(auction_dates).intersection(set(dates))
    return intersection


if __name__ == '__main__':

    datetime_now = dt.datetime.now()
    today = datetime_now.date()
    time = datetime_now.time()
    time_of_day = dt.datetime.combine(dt.date.min, time) - dt.datetime.combine(dt.date.min, dt.time.min)

    if is_mos_business_date(today):
        tasks = []
        tasks.append(BondsBoards(date=today, out_dir=OUT_DIR))
        tasks.append(AuctionModelEmail(date=today, out_dir=OUT_DIR, time_of_day=time_of_day))

        dates = [d for d in get_mos_dates_range(today, 2) if d > dt.date(2023, 1, 1)]
        tasks += [YieldCandles(date=date, out_dir=OUT_DIR, board=Board.TQOB, bond_type=BondType.FIX, period=period) for date in dates for period in [CandlePeriod.TEN_MINUTES, CandlePeriod.ONE_HOUR]]
        tasks += [YieldCandles(date=date, out_dir=OUT_DIR, board=Board.PACT, bond_type=BondType.FIX, period=CandlePeriod.ONE_MINUTE) for date in filter_auction_dates(dates)]
        tasks += [Candles(date=date, out_dir=OUT_DIR, board=Board.EQRP, period=CandlePeriod.ONE_HOUR) for date in dates]
        tasks += [BondMetricsAtVWAP(date=date, out_dir=OUT_DIR, bond_type=BondType.FIX) for date in dates]
        tasks += [GCurve(date=date, out_dir=OUT_DIR) for date in dates]
        tasks += [GCurveParams(date=date, out_dir=OUT_DIR) for date in dates]
        tasks += [AuctionModelCalibration(date=date, out_dir=OUT_DIR) for date in dates]

        tasks_to_run = [task for task in tasks if not task.complete() and task.is_final_output()]
        if tasks_to_run:
            luigi.build(tasks_to_run, workers=1, local_scheduler=True)