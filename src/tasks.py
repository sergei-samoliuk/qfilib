import datetime as dt
import json
from enum import Enum, auto
from functools import lru_cache

import moexalgo as ma
import numpy as np
import pandas as pd
import requests
from luigi.util import requires
from moexalgo import Ticker
from moexalgo.session import Session
from moexalgo.utils import CandlePeriod

import luigi

from config import DATA_INPUT_DIR
from src.graph import DailyPickleTask, DailyMixin, InMemoryTask, to_dict_by_class
from src.instruments import symbol_to_series
from src.qr import vq, get_calypso_session, shift_mos_date


class Board(Enum):
    TQOB = auto()
    EQRP = auto()  # repo 1d
    PACT = auto()  # auction main, no additional
    # AUCT = auto() # always empty
    # PSAU = auto() # need to check

class BondType(Enum):
    FIX = 'фикс. купон'
    FLOATER = 'флоутер'
    LINKER = 'ипц-линкер'


def convert_scalar(value, data_type):
    """Convert a scalar value based on data_type."""
    if pd.isna(value):
        return value

    if data_type == 'date':
        return pd.to_datetime(value)
    elif data_type == 'time':
        return pd.to_timedelta(value)
    elif data_type == 'int32' or data_type == 'int64':
        return int(value)
    elif data_type == 'double':
        return float(value)
    elif data_type == 'string':
        return str(value)
    elif data_type == 'boolean':
        return bool(value)
    elif data_type == 'number':
        return float(value)
    else:
        raise ValueError(f"'{data_type}' is not supported")

def convert_series(series, data_type):
    """Convert a scalar value based on data_type."""

    if data_type == 'date':
        return pd.to_datetime(series)
    elif data_type == 'time':
        return pd.to_timedelta(series)
    elif data_type == 'int32' or data_type == 'int64':
        # do float conversion first to handle values like 1.000000e+09
        return series.astype(float).astype(int)
    elif data_type == 'double':
        return series.astype(float)
    elif data_type == 'string':
        return series.astype(str)
    elif data_type == 'boolean':
        return series.fillna(0).astype(int).astype(bool)
    elif data_type == 'number':
        return series.astype(float)
    else:
        raise ValueError(f"'{data_type}' is not supported")


def json_to_dataframe_with_timedelta(json_data):
    """
    Convert JSON data with metadata to pandas DataFrame.
    Convert time columns to timedelta (duration since midnight).
    """
    # Extract metadata and data
    metadata = json_data['metadata']
    columns = json_data['columns']
    rows = json_data['data']

    # Create DataFrame from rows
    df = pd.DataFrame(rows, columns=columns)

    # Convert data types based on metadata
    for col_name, col_info in metadata.items():
        if col_name in df.columns:
            data_type = col_info['type']
            try:
                df[col_name] = convert_series(df[col_name], data_type)
            except:
                print(f'failed to convert {col_name} to {data_type}, values: {df[col_name].value_counts().head()}')
                pass
    return df


@lru_cache(maxsize=None)
def get_moex_secid_data(secid, section):
    def deserializer(data):
        columns = data[section]["columns"]
        metadata = data[section]["metadata"]
        types = [metadata[c]["type"] for c in columns]
        return [dict({c: convert_scalar(v, t) for c, t, v in zip(columns, types, row)}) for row in data[section]["data"]]

    with Session() as client:
        data = client.get_objects(
            f"securities/{secid}",
            deserializer,
        )
    return data


def get_request_response(url):
    headers = {
        'Authorization': f'Bearer {ma.session.TOKEN}',
    }
    response = requests.request("GET", url, headers=headers)
    response_json = json.loads(response.text)
    return response_json


class GCurve(DailyPickleTask):

    def file_name(self):
        return "gcurve"

    def produce_output(self):
        url = f"https://iss.moex.com/iss/engines/stock/zcyc/securities.json?date={self.date.strftime("%Y-%m-%d")}"
        response_json = get_request_response(url)
        df = json_to_dataframe_with_timedelta(response_json['securities'])
        return df

    def postprocess_on_read(self, df: pd.DataFrame) -> pd.DataFrame:
        df['spread'] = df['trdyield'] - df['crtyield']
        return df.set_index('secid', verify_integrity=True)


class GCurveParams(DailyPickleTask):

    def file_name(self):
        return "gcurve_params"

    def produce_output(self):
        url = f"https://iss.moex.com/iss/engines/stock/zcyc/params.json?date={self.date.strftime("%Y-%m-%d")}"
        response_json = get_request_response(url)
        df = json_to_dataframe_with_timedelta(response_json['params'])
        return df

    def calc_spot_yields(self, tenors: np.array):
        def GT(t, beta0, beta1, beta2, tau, g_values):

            exp_term = np.exp(-t / tau)
            factor = tau * (1 - exp_term) / t

            term1 = beta0 + beta1 * factor
            term2 = beta2 * (factor - exp_term)

            a_values, b_values = _compute_ab()

            term3 = 0.0
            for i in range(9):
                if b_values[i] != 0:
                    term3 += g_values[i] * np.exp(-((t - a_values[i]) ** 2) / (b_values[i] ** 2))

            return (term1 + term2 + term3) / 10000

        def _compute_ab():
            k = 1.6

            a_values = np.zeros(9)
            b_values = np.zeros(9)

            a_values[0] = 0.0
            a_values[1] = 0.6
            b_values[0] = a_values[1]  # b[0] = 0.6

            for i in range(1, 9):
                b_values[i] = b_values[i - 1] * k

            for i in range(2, 9):
                a_values[i] = a_values[i - 1] + a_values[1] * k ** (i - 1)

            return a_values, b_values

        def KBD(t, beta0, beta1, beta2, tau, g_values):
            return 100.0 * (np.exp(GT(t, beta0, beta1, beta2, tau, g_values)) - 1.0)

        params = self.read_output().iloc[0]
        beta0, beta1, beta2, tau, *g_values = params[['B1', 'B2', 'B3', 'T1', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']]

        return KBD(tenors, beta0, beta1, beta2, tau, g_values)


@lru_cache(maxsize=1000)
def ticker_candles_quarter(ticker, start, end, period):
    return ticker.candles(start=start, end=end, period=period)


def ticker_candles(ticker, date, period):
    if date < dt.date.today():
        pd_period = pd.Period(date, freq='Q')
        df = ticker_candles_quarter(ticker, pd_period.start_time.date(), pd_period.end_time.date(), period)
        if df.empty:
            return df
        return df[pd.to_datetime(df['begin']).dt.date == date].copy()
    else:
        return ticker.candles(start=date, end=date, period=period)



class RawSecurities(DailyPickleTask):

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'raw_securities'

    def produce_output(self) -> pd.DataFrame:
        assert self.date == dt.date.today()
        url = "https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQOB/securities.json"
        response_json = get_request_response(url)
        df = json_to_dataframe_with_timedelta(response_json['securities'])
        return df.assign(timestamp=pd.Timestamp.now())


def description_postprocess_on_read(df):
    type_dict = df[['name', 'type']].drop_duplicates().set_index('name', verify_integrity=True)['type'].to_dict()
    df = df.set_index(['secid', 'name'])['value'].unstack()
    for c, dtype in type_dict.items():
        df[c] = convert_series(df[c], dtype)

    bond_type_map = {'облигации федерального займа с постоянным купонным доходом': BondType.FIX.value,
                     'облигации федерального займа с переменным купонным доходом': BondType.FLOATER.value,
                     'облигации федерального займа с индексируемым номиналом': BondType.LINKER.value}
    df['SHORT_TYPE'] = df.pop('ISSUENAME').str.lower().map(bond_type_map)
    assert df[['FACEUNIT', 'ISSUEDATE']].notna().all().all()
    # drop non rub, amortizing etc
    df = df[df['FACEUNIT'] == 'SUR'].dropna(subset=['SHORT_TYPE']).copy()
    df['REGISTRY_DATE'] = df['REGISTRY_DATE'].fillna(df['ISSUEDATE'] - pd.Timedelta(days=2))
    df['series'] = symbol_to_series(df.index.to_series())
    return df


@requires(RawSecurities)
class RawSecuritiesDescription(DailyPickleTask):

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'raw_securities_description'

    def produce_output(self) -> pd.DataFrame:
        raw_securities = self.requires().read_output()
        dfs = []
        for secid in raw_securities['SECID']:
            secid_data = get_moex_secid_data(secid, 'description')
            dfs.append(pd.DataFrame(secid_data).assign(secid=secid))
        return pd.concat(dfs)

    def postprocess_on_read(self, df):
        return description_postprocess_on_read(df)

class RawAuctionResultsHistory(DailyPickleTask):

    def file_name(self):
        return 'raw_auction_results_history'

    def produce_output(self):
        url = 'https://www.bondresearch.ru/boards/auctions_table.json'
        response = requests.get(url)
        data = response.json()

        column_names = [
            'Дата',  # 0
            'Серия',  # 1
            'Погашение',  # 2
            'Тип',  # 3
            'Купон, %',  # 4
            'Спрос по номиналу, млн руб.',  # 5
            'Номинальный объем размещения, млн руб.',  # 6
            'Цена отсечения, пп',  # 7
            'Средняя цена, пп',  # 8
            'Дох-ть / спред по цене отсечения, % (бп)',  # 9
            'Средняя дох-ть / cпред, % (бп)',  # 10
            'DV01',  # 11
            'Премия, бп',  # 12
            'check'  # 13
        ]
        auctions = pd.DataFrame(data['demo'], columns=column_names)
        return auctions

    @property
    def numeric_cols(self):
        return ['Купон, %', 'Спрос по номиналу, млн руб.',
                'Номинальный объем размещения, млн руб.',
                'Цена отсечения, пп', 'Средняя цена, пп',
                'Дох-ть / спред по цене отсечения, % (бп)',
                'Средняя дох-ть / cпред, % (бп)', 'DV01', 'Премия, бп']

    @property
    def date_cols(self):
        return ['Дата', 'Погашение']

    def postprocess_on_read(self, df):
        df['Серия'] = df['Серия'].astype(int)

        for col in self.numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='raise').astype(float)

        for col in self.date_cols:
            df[col] = pd.to_datetime(df[col], errors='raise')

        return df


class AuctionResultsHistory(InMemoryTask):

    def requires(self):
        return self.clone(RawAuctionResultsHistory, date=dt.date.today()).clone_previous()

    def produce_output(self):
        df = self.requires().read_output()

        rename = {'Дата': 'date',
                  'Серия': 'series',
                  'Погашение': 'maturity_date',
                  'Спрос по номиналу, млн руб.': 'demand_mln',
                  'Цена отсечения, пп': 'cutoff_price',
                  'Средняя цена, пп': 'average_price',
                  'Номинальный объем размещения, млн руб.': 'placed_size_mln',
                  'Дох-ть / спред по цене отсечения, % (бп)': 'cutoff_yield',
                  'Средняя дох-ть / cпред, % (бп)': 'average_yield',
                  'Премия, бп': 'premium'}
        df = df[list(rename.keys())].rename(columns=rename)

        for c in ['demand_mln', 'placed_size_mln']:
            df[c] = df[c].fillna(0.)

        df = df.sort_values('date').reset_index(drop=True)
        self.add_auction_announce_time(df)
        df['result_release_ts'] = df['date'] + pd.Timedelta(hours=18)
        df['quarter'] = df['date'].dt.to_period('Q')

        return df

    @staticmethod
    def add_auction_announce_time(auctions):
        auctions['announce_date'] = auctions['date'] - pd.Timedelta(days=1)
        auctions['announce_ts'] = auctions['announce_date'] + pd.Timedelta(hours=16)


@requires(AuctionResultsHistory)
class AuctionDates(InMemoryTask):

    @property
    def extra_dates(self):
        # https://minfin.gov.ru/ru/perfomance/public_debt/internal/operations/ofz/auction?id_65=314995-grafik_auktsionov_po_razmeshcheniyu_obligatsii_federalnykh_zaimov_na_i_kvartal_2026_goda
        return [
            pd.to_datetime('2026-01-14'),
            pd.to_datetime('2026-01-21'),
            pd.to_datetime('2026-01-28'),
            pd.to_datetime('2026-02-04'),
            pd.to_datetime('2026-02-11'),
            pd.to_datetime('2026-02-18'),
            pd.to_datetime('2026-02-25'),
            pd.to_datetime('2026-03-04'),
            pd.to_datetime('2026-03-11'),
            pd.to_datetime('2026-03-18'),
            pd.to_datetime('2026-03-25')
        ]

    def produce_output(self) -> pd.DataFrame:
        df = self.requires().read_output()
        dates = df['date'].to_list() + self.extra_dates
        df = pd.Series(dates).drop_duplicates().sort_values().reset_index(drop=True).to_frame('date')
        AuctionResultsHistory.add_auction_announce_time(df)
        df['ordinal'] = range(1, len(df) + 1)
        df['quarter'] = df['date'].dt.to_period('Q')
        return df


class MinfinPlanHistory(InMemoryTask):

    def produce_output(self):
        # chatgpt + https://minfin.gov.ru/ru/perfomance/public_debt/internal/operations/ofz/auction, Таблицы планируемых аукционов
        df = pd.read_excel(DATA_INPUT_DIR / 'minfin_plans.xlsx')
        df['quarter'] = df['quarter'].map(pd.Period)
        df['amount_mln'] = df['amount_bln'] * 1e3
        df['amount_mln_per_auction'] = df['amount_mln'] / df['nb_auctions']
        df['maturity_bucket'] = df.apply(lambda row: pd.Interval(row['min_maturity'], row['max_maturity']), axis=1)
        return df


class MinfinDebtStructure(InMemoryTask, DailyMixin):
    # https://minfin.gov.ru/ru/perfomance/public_debt/internal/structure, Объем ценных бумаг по выпускам

    def produce_output(self):
        df = pd.read_excel(DATA_INPUT_DIR / f'INTERNET_Volume_of_issues_rus_{self.date.strftime('%Y%m%d')}.xlsx', skiprows=3)
        rename = {'Тип ценной бумаги': 'type',
                  'Дата окончания размещения1': 'placement_end_date',
                  'Дата погашения': 'maturity',
                  'Объем эмиссии (объявленный),      млн. руб.2': 'issue_size_mln',
                  'Фактически размещено, \nмлн. руб.2': 'placed_size_mln',
                  'Остаток, доступный для размещения, \nмлн. руб.2,4': 'remaining_size_mln'}
        df = df[df['Тип ценной бумаги'].isin(['ОФЗ-ПД', 'ОФЗ-ПК', 'ОФЗ-ИН'])]
        df['series'] = df['Код выпуска ценной бумаги'].str.slice(0, 5).astype(int)
        df = df[['series'] + list(rename.keys())].rename(columns=rename).groupby('series').first()
        cols = ['issue_size_mln', 'placed_size_mln', 'remaining_size_mln']
        df[cols] = df[cols].astype(float)
        return df


class IssueSizeReopenings(InMemoryTask):
    # chatgpt + tass/interfax

    def produce_output(self):
        df = pd.read_excel(DATA_INPUT_DIR / 'issue_schedule.xlsx')
        reopening_mask = df.pop('type') == 'reopening'
        df = df[reopening_mask].copy()
        return df



@requires(MinfinDebtStructure)
class MinfinDebtStructureDescription(DailyPickleTask):

    def file_name(self):
        return 'minfin_debt_structure_description'

    @staticmethod
    def get_moex_secid_data(series):
        # guessing here
        for i in range(10):
            secid = f'SU{series}RMFS{i}'
            secid_data = get_moex_secid_data(secid, 'description')
            if secid_data:
                return secid, secid_data
        raise LookupError(series)

    def produce_output(self) -> pd.DataFrame:
        series_list = self.requires().read_output().index
        dfs = []
        for series in series_list:
            secid, secid_data = self.get_moex_secid_data(series)
            dfs.append(pd.DataFrame(secid_data).assign(secid=secid))
        return pd.concat(dfs)

    def postprocess_on_read(self, df):
        return description_postprocess_on_read(df)


class BondsDescription(DailyPickleTask):
    # all bonds with REGISTRY_DATE <= self.date, self.date + 1 < MATDATE,
    # should be same as RawSecuritiesDescription run as of dt.date.today()
    # TODO: how to include new bond on day one?

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'description'

    def requires(self):
        if self.date >= dt.date(2026, 2, 3):
            return [self.clone(RawSecuritiesDescription)]
        elif self.date >= dt.date(2026, 1, 1):
            return [self.clone(MinfinDebtStructureDescription, date=dt.date(2025, 12, 31))]
        else:
            return [self.clone(MinfinDebtStructureDescription, date=date) for date in
                    [dt.date(2021, 12, 31), dt.date(2025, 12, 31)]]

    def produce_output(self) -> pd.DataFrame:
        df = pd.concat([t.read_output().reset_index() for t in self.requires()])
        settlement_date = pd.to_datetime(shift_mos_date(self.date, 1))
        mask = (df['REGISTRY_DATE'] <= self.sod_timestamp) & (settlement_date < df['MATDATE'])
        return df[mask].drop_duplicates(subset=['secid']).set_index('secid')


class BondsBoards(DailyPickleTask):

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'boards'

    def requires(self):
        return self.clone(BondsDescription)

    def produce_output(self) -> pd.DataFrame:
        secids = self.requires().read_output().index
        dfs = []
        for secid in secids:
            df = pd.DataFrame(get_moex_secid_data(secid, 'boards'))
            dfs.append(df.assign(secid=secid))
        return pd.concat(dfs).set_index(['boardid', 'secid'], verify_integrity=True)

    def validate_output(self, df):
        if not (self.sod_timestamp <= df['listed_till']).any():
            raise RuntimeError('MOEX static is not up-to-date!')

    def read_filtered_output(self, board: Board):
        df = self.read_output().loc[board.name]
        mask = (df['listed_from'] <= self.sod_timestamp) & (self.sod_timestamp <= df['listed_till'])
        return df[mask]


@requires(BondsBoards)
class Candles(DailyPickleTask):
    board = luigi.EnumParameter(enum=Board)
    period = luigi.EnumParameter(enum=CandlePeriod)

    def file_name(self):
        return f"candles_{self.board.name}_{self.period.name}"

    @staticmethod
    def transform_raw_frame(df):
        for c in ['begin', 'end']:
            df[c] = pd.to_datetime(df[c])
        df = df.set_index(['secid', 'begin'], verify_integrity=True).sort_index()
        return df

    def produce_output(self):
        secids = self.requires().read_filtered_output(self.board).index
        dfs = []
        for secid in secids:
            ticker_object = Ticker(secid, self.board.name)
            df = ticker_candles(ticker_object, self.date, self.period)
            dfs.append(df.assign(secid=secid))
        df = pd.concat(dfs)
        df = self.transform_raw_frame(df)
        return df

    def postprocess_on_read(self, df):
        df['weighted_avg'] = df['value'] / df['volume'] * 0.1
        return df


class QRBondProducts(DailyPickleTask):
    # cache of all calypso qr bonds
    FIRST_DATE = dt.date(2026, 2, 11)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs['date'] < self.FIRST_DATE:
            raise RuntimeError(f"{self.date=} must be >= {self.FIRST_DATE=}")

    @staticmethod
    def floor_date(date):
        return max(date, QRBondProducts.FIRST_DATE)

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'qr_bond_products'

    def requires(self):
        d = {BondsDescription: [self.clone(BondsDescription, date=date) for date in [dt.date(2022, 1, 10), self.date]]}
        if self.use_prev_self():
            d[QRBondProducts] = self.clone_previous()
        return d

    def use_prev_self(self):
        return self.date > self.FIRST_DATE

    def produce_output(self):
        df = pd.concat(t.read_output()[['ISIN', 'series']] for t in self.requires()[BondsDescription]).drop_duplicates()

        if self.use_prev_self():
            product_by_isin = self.requires()[QRBondProducts].read_output().set_index('ISIN')['QR_Bond_Product'].to_dict()
        else:
            product_by_isin = {}

        missing_isins = [isin for isin in df['ISIN'].to_list() if isin not in product_by_isin]
        if len(missing_isins) > 0:
            session = get_calypso_session()
            for isin in missing_isins:
                product_by_isin[isin] = session.get_QR_Bond_Product(isin).decode("utf-8-sig")

        df['QR_Bond_Product'] = df['ISIN'].map(product_by_isin)

        return df


class YieldCandles(DailyPickleTask):
    board = luigi.EnumParameter(enum=Board)
    period = luigi.EnumParameter(enum=CandlePeriod)
    bond_type = luigi.EnumParameter(enum=BondType)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        board = kwargs['board']
        if not board in [Board.TQOB, Board.PACT]:
            raise ValueError(board)

    def requires(self):
        tasks = [self.clone(QRBondProducts, date=QRBondProducts.floor_date(self.date)),
                 self.clone(Candles),
                 self.clone(BondsDescription)]
        return to_dict_by_class(tasks)

    def file_name(self):
        return f"yield_candles_{self.bond_type.name}_{self.board.name}_{self.period.name}"

    def produce_output(self):
        products_df = self.requires()[QRBondProducts].read_output()
        candles_df = self.requires()[Candles].read_output()
        description_df = self.requires()[BondsDescription].read_output()
        secids = description_df[description_df['SHORT_TYPE'] == self.bond_type.value].index
        candles_df = candles_df[candles_df.index.get_level_values(0).isin(secids)]

        yield_dfs = []

        for (secid, candles_df_) in candles_df.groupby('secid'):
            qr_object = vq.LoadObjectFromString(products_df['QR_Bond_Product'].loc[secid])
            var_date = vq.toVariant(self.date)
            var_details = vq.Details({'YieldCalculationMethod': 'TrueAnnualized'}).toVariant()

            def price_to_yield(price):
                return vq.toNumpy(vq.BondPriceToYield(qr_object, vq.toVariant(price), var_date, var_details))

            yield_df = candles_df_.copy()

            price_cols = ['open', 'close', 'high', 'low', 'weighted_avg']

            yield_df[price_cols] = yield_df[price_cols].map(price_to_yield)
            yield_dfs.append(yield_df)

        return pd.concat(yield_dfs, verify_integrity=True)


class BondMetricsAtVWAP(DailyPickleTask):
    bond_type = luigi.EnumParameter(enum=BondType)

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def requires(self):
        tasks = [self.clone(BondsDescription),
                 self.clone(QRBondProducts, date=QRBondProducts.floor_date(self.date)),
                 self.clone_previous(Candles, board=Board.TQOB, period=CandlePeriod.ONE_HOUR)]
        return to_dict_by_class(tasks)

    def file_name(self):
        return f"bond_metrics_at_vwap_{self.bond_type.name}"

    def produce_output(self):
        products_df = self.requires()[QRBondProducts].read_output()
        candles_df = self.requires()[Candles].read_output()
        description_df = self.requires()[BondsDescription].read_output()
        secids = description_df[description_df['SHORT_TYPE'] == self.bond_type.value].index

        df = candles_df.groupby('secid')[['value', 'volume']].sum()
        df['price'] = df['value'] / df['volume'] * 0.1
        df = df.reindex(secids)[['price']]  # nans for new bonds etc

        var_date = vq.toVariant(self.date)
        var_details = vq.Details({'YieldCalculationMethod': 'TrueAnnualized'}).toVariant()

        for secid in df.dropna().index:
            var_price = vq.toVariant(df.loc[secid]['price'])
            qr_object = vq.LoadObjectFromString(products_df.loc[secid]['QR_Bond_Product'])

            df.loc[secid, 'yield'] = vq.toNumpy(vq.BondPriceToYield(qr_object, var_price, var_date, var_details))
            df.loc[secid, 'duration'] = vq.toNumpy(vq.BondPriceToDuration(qr_object, var_price, var_date, var_details))
            df.loc[secid, 'modified_duration'] = vq.toNumpy(vq.BondPriceToModifiedDuration(qr_object, var_price, var_date, var_details))
            df.loc[secid, 'convexity'] = vq.toNumpy(vq.BondPriceToConvexity(qr_object, var_price, var_date, var_details))

        return df


def get_value_ts_df(t0_df, delta_df, snapshot_df, snapshot_ts):
    t0_df = t0_df[['series', 'timestamp']]
    delta_df = delta_df[['series', 'timestamp', 'delta']]
    snapshot_df = snapshot_df[['series', 'value']]

    res_df = pd.concat([t0_df.assign(delta=0.), delta_df], ignore_index=True)
    res_df['total_delta'] = res_df.groupby('series')['delta'].cumsum()
    total_delta_at_snapshot_ts = res_df[res_df['timestamp'] <= snapshot_ts].groupby('series')['total_delta'].last()
    value_at_snapshot_ts = snapshot_df.set_index('series', verify_integrity=True)['value']
    res_df['value'] = res_df['series'].map(value_at_snapshot_ts) + res_df['total_delta'] - res_df['series'].map(total_delta_at_snapshot_ts)
    res_df = res_df[['series', 'timestamp', 'value']].copy()
    return res_df


class IssueSizeHistory(InMemoryTask):

    @property
    def snapshot_dates(self):
        # should be no bonds in between
        return [dt.date(2025, 12, 31), dt.date(2021, 12, 31)]

    def requires(self):
        return {MinfinDebtStructure: [self.clone(MinfinDebtStructure, date=date) for date in self.snapshot_dates],
                MinfinDebtStructureDescription: [self.clone(MinfinDebtStructureDescription, date=date) for date in self.snapshot_dates],
                IssueSizeReopenings: self.clone(IssueSizeReopenings)}

    def produce_output(self) -> pd.DataFrame:
        delta_df = self.requires()[IssueSizeReopenings].read_output().rename(columns={'announcement_ts_msk': 'timestamp', 'volume_mln': 'delta'})[['series', 'timestamp', 'delta']]
        res_dfs = []
        series_processed = []
        for structure_task, desc_task in zip(self.requires()[MinfinDebtStructure], self.requires()[MinfinDebtStructureDescription]):
            t0_df = desc_task.read_output().rename(columns={'REGISTRY_DATE': 'timestamp'})[['series', 'timestamp']]
            snapshot_df = structure_task.read_output().reset_index().rename(columns={'issue_size_mln': 'value'})[['series', 'value']]
            snapshot_ts = structure_task.sod_timestamp
            res_df = get_value_ts_df(t0_df, delta_df, snapshot_df, snapshot_ts)
            res_df = res_df[~res_df['series'].isin(series_processed)].copy()
            series_processed += list(res_df['series'].unique())
            res_dfs.append(res_df)

        return pd.concat(res_dfs).rename(columns={'value': 'issue_size_mln'})


class PlacementEndDateHistory(InMemoryTask):

    @property
    def snapshot_dates(self):
        # should be no bonds in between
        return [dt.date(2025, 12, 31), dt.date(2024, 12, 31), dt.date(2021, 12, 31)]

    def requires(self):
        return {MinfinDebtStructure: [self.clone(MinfinDebtStructure, date=date) for date in self.snapshot_dates],
                MinfinDebtStructureDescription: [self.clone(MinfinDebtStructureDescription, date=date) for date in self.snapshot_dates],
                IssueSizeReopenings: self.clone(IssueSizeReopenings)}

    def produce_output(self) -> pd.DataFrame:
        change_df = self.requires()[IssueSizeReopenings].read_output().rename(columns={'announcement_ts_msk': 'timestamp'})[['series', 'timestamp', 'placement_end_date']]
        res_dfs = [change_df]
        for structure_task, desc_task in zip(self.requires()[MinfinDebtStructure], self.requires()[MinfinDebtStructureDescription]):
            t0_df = desc_task.read_output().rename(columns={'REGISTRY_DATE': 'timestamp'})[['series', 'timestamp']]
            res_dfs.append(t0_df)
            snapshot_df = structure_task.read_output().reset_index().assign(timestamp=structure_task.sod_timestamp)[['series', 'timestamp', 'placement_end_date']]
            res_dfs.append(snapshot_df)

        res_df = pd.concat(res_dfs).drop_duplicates().sort_values(['series', 'timestamp'])
        res_df['placement_end_date'] = res_df.groupby('series')['placement_end_date'].bfill()
        res_df = res_df.drop_duplicates(subset=['series', 'placement_end_date'], keep='first')

        return res_df.reset_index(drop=True)


class PlacedSizeHistory(InMemoryTask):

    @property
    def snapshot_dates(self):
        # should be no bonds in between, snapshots are trustable, prefer later snapshots, hence desc order
        return [dt.date(2025, 12, 31), dt.date(2021, 12, 31)]

    def requires(self):
        return {MinfinDebtStructure: [self.clone(MinfinDebtStructure, date=date) for date in self.snapshot_dates],
                MinfinDebtStructureDescription: [self.clone(MinfinDebtStructureDescription, date=date) for date in self.snapshot_dates],
                AuctionResultsHistory: self.clone(AuctionResultsHistory)}

    def validate_output(self, res_df) -> None:
        first_df = res_df.groupby('series').first()
        # should be zero size on t0 ...
        zero_at_t0 = first_df['placed_size_mln'].abs() < 10
        # ... unless it is
        before_first_date = first_df['timestamp'] < pd.Timestamp(min(self.snapshot_dates))
        is_linker = first_df['type'] == 'ОФЗ-ИН' # linker notional is time varying, so calc is incorrect
        known_issue = first_df.index.to_series() == 29027 # minfin is wrong
        bad_df = first_df[~(zero_at_t0 | before_first_date | is_linker | known_issue)]

        if not bad_df.empty:
            print(bad_df)
            raise RuntimeError()

    def produce_output(self) -> pd.DataFrame:
        delta_df = self.requires()[AuctionResultsHistory].read_output().rename(columns={'date': 'timestamp', 'placed_size_mln': 'delta'})[['series', 'timestamp', 'delta']]
        res_dfs = []
        series_processed = []
        for structure_task, desc_task in zip(self.requires()[MinfinDebtStructure], self.requires()[MinfinDebtStructureDescription]):
            t0_df = desc_task.read_output().rename(columns={'REGISTRY_DATE': 'timestamp'})[['series', 'timestamp']]
            structure_df = structure_task.read_output().reset_index()
            snapshot_df = structure_df.rename(columns={'placed_size_mln': 'value'})[['series', 'value']]
            snapshot_ts = structure_task.sod_timestamp
            res_df = get_value_ts_df(t0_df, delta_df, snapshot_df, snapshot_ts)
            res_df = res_df[~res_df['series'].isin(series_processed)].copy()
            series_processed += list(res_df['series'].unique())
            res_dfs.append(res_df.merge(structure_df[['series', 'type']], on='series'))

        return pd.concat(res_dfs).rename(columns={'value': 'placed_size_mln'})