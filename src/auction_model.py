import datetime as dt
import hashlib
import tempfile
import calendar
from enum import Enum, auto
from pathlib import Path

import optuna
import pandas as pd
import win32com.client as win32
from catboost import Pool, CatBoostClassifier, CatBoostRanker
from moexalgo import CandlePeriod
from sklearn.isotonic import IsotonicRegression

import luigi
from src import pandas_ext
from src.graph import InMemoryTask, DailyMixin, DailyPickleTask, to_dict_by_class
from src.qr import get_mos_dates_range, shift_mos_date
from src.tasks import BondsDescription, AuctionDates, AuctionResultsHistory, IssueSizeHistory, PlacedSizeHistory, \
    MinfinPlanHistory, PlacementEndDateHistory, GCurve, Candles, Board

pandas_ext.register_x_accessor()

N_TRIALS = 500
CALIB_WINDOW = 500


class TargetType(Enum):
    FIRST = 0
    SECOND = 1


class AuctionModelSample(InMemoryTask, DailyMixin):
    target_type = luigi.EnumParameter(enum=TargetType)
    time_of_day = luigi.TimeDeltaParameter()

    def requires(self):
        tasks = [
            self.clone(BondsDescription),
            self.clone(AuctionDates),
            self.clone(AuctionResultsHistory),
            self.clone(IssueSizeHistory),
            self.clone(PlacedSizeHistory),
            self.clone(MinfinPlanHistory),
            self.clone(PlacementEndDateHistory),
            self.clone_previous(GCurve),
            [self.clone(Candles, date=date, board=Board.TQOB, period=CandlePeriod.ONE_HOUR) for date in get_mos_dates_range(self.prev_date, 10)]
        ]
        return to_dict_by_class(tasks)

    @property
    def prediction_ts(self):
        return self.sod_timestamp + pd.to_timedelta(self.time_of_day)

    def enrich_prev_auctions_results(self, x):
        auctions_df = self.requires()[AuctionResultsHistory].read_output()
        auc_dates_df = self.requires()[AuctionDates].read_output()[['date', 'ordinal']]
        auctions_df = auctions_df.merge(auc_dates_df, on='date')
        auc_cols = ['announce_ts', 'date', 'demand_mln', 'placed_size_mln', 'ordinal']

        prev1_auc_df = pd.merge_asof(
            x[['prediction_ts', 'series']],
            auctions_df.x.add_prefix('prev1_auc_', auc_cols, ['series']),
            left_on='prediction_ts',
            right_on='prev1_auc_announce_ts',
            by='series',
            allow_exact_matches=False).drop(columns=['prediction_ts'])
        x = x.merge(prev1_auc_df, how='left', on='series')

        prev2_auc_df = pd.merge_asof(
            prev1_auc_df[['prev1_auc_announce_ts', 'series']].dropna().sort_values('prev1_auc_announce_ts'),
            auctions_df.x.add_prefix('prev2_auc_', auc_cols, ['series']),
            left_on='prev1_auc_announce_ts',
            right_on='prev2_auc_announce_ts',
            by='series',
            allow_exact_matches=False).drop(columns='prev1_auc_announce_ts')
        x = x.merge(prev2_auc_df, how='left', on='series')

        return x

    def enrich_gcurve(self, x):
        x = x.merge(self.requires()[GCurve].read_output()[['correction', 'spread']], how='left', left_on='secid', right_index=True)
        return x

    def enrich_placed_size(self, x):
        placed_size_df = self.requires()[PlacedSizeHistory].read_output()[['series', 'timestamp', 'placed_size_mln']]
        x = pd.merge_asof(x, placed_size_df.sort_values('timestamp'), left_on='prediction_ts', right_on='timestamp', by='series').rename(columns={'timestamp': 'placed_size_ts'})
        # for new bonds it is just zero
        x['placed_size_mln'] = x['placed_size_mln'].fillna(0.)
        return x

    def enrich_issue_size(self, x):
        issue_size_df = self.requires()[IssueSizeHistory].read_output()[['series', 'timestamp', 'issue_size_mln']]
        x = pd.merge_asof(x, issue_size_df.sort_values('timestamp'), left_on='prediction_ts', right_on='timestamp', by='series').rename(columns={'timestamp': 'issue_size_ts'})
        # for new bonds rely on moex description
        x['issue_size_ts'] = x['issue_size_ts'].fillna(x['ISSUEDATE'])
        x['issue_size_mln'] = x['issue_size_mln'].fillna(x['ISSUESIZE'] / 1000)
        return x

    def enrich_placement_end_date(self, x):
        placement_end_date_df = self.requires()[PlacementEndDateHistory].read_output()[['series', 'timestamp', 'placement_end_date']]
        x = pd.merge_asof(x, placement_end_date_df.sort_values('timestamp'), left_on='prediction_ts', right_on='timestamp', by='series').rename(columns={'timestamp': 'placement_end_date_ts'})
        return x

    def enrich_aggregated_auctions_results(self, x):
        auctions_df = self.requires()[AuctionResultsHistory].read_output()
        for timedelta_str in ['90d']:
            timedelta = pd.Timedelta(timedelta_str)
            orig_df = auctions_df[['result_release_ts', 'series', 'placed_size_mln', 'demand_mln']].assign(nb_auctions=1)
            shifted_df = orig_df[['result_release_ts', 'series']].copy()
            shifted_df['result_release_ts'] += timedelta
            combined_df = pd.concat([orig_df, shifted_df], axis=0)
            combined_df = combined_df.drop_duplicates(['result_release_ts', 'series'], keep='first').set_index('result_release_ts').sort_index()

            rolling_stats_df = combined_df.groupby('series').rolling(timedelta).agg({'placed_size_mln': 'mean', 'demand_mln': 'mean', 'nb_auctions': 'count'})
            rolling_stats_df = rolling_stats_df.reset_index()
            rolling_stats_df = rolling_stats_df.sort_values('result_release_ts').x.add_prefix(f'{timedelta_str}_', except_cols=['series'])
            x = pd.merge_asof(x, rolling_stats_df, left_on='prediction_ts', right_on=f'{timedelta_str}_result_release_ts', by='series')
            rolling_stats_columns = rolling_stats_df.select_dtypes(float).columns
            x[rolling_stats_columns] = x[rolling_stats_columns].fillna(0.)
        return x

    def hard_filtering(self, x):
        has_remaining_size = x['remaining_size_mln'] > 5000
        # sometimes newly registered bond not yet available for the next auction
        issue_date_in_future = x['target_date'] >= x['ISSUEDATE']
        not_short = x['years_to_maturity'] >= 2.0
        can_be_placed = x['target_date'] <= x['placement_end_date']
        x = x[has_remaining_size & issue_date_in_future & not_short & can_be_placed].copy()
        return x

    def enrich_target_value(self, x):
        auctions_df = self.requires()[AuctionResultsHistory].read_output()
        xy = x.merge(auctions_df[['series', 'date', 'announce_ts']].assign(target=1).rename(columns={'date': 'target_date', 'announce_ts': 'target_announce_ts'}),
                     on=['series', 'target_date'], how='left')

        xy['target_announce_ts'] = xy.groupby('target_date')['target_announce_ts'].transform('max')
        xy['target'] = xy['target'].fillna(0.).where(xy['target_announce_ts'].notna())
        return xy

    def define_target_properties(self):
        auc_dates_df = self.requires()[AuctionDates].read_output()
        auc_dates_df = auc_dates_df[auc_dates_df['announce_ts'] > self.prediction_ts][['date', 'ordinal', 'quarter']]
        target_properties = auc_dates_df.add_prefix('target_').iloc[self.target_type.value].to_dict()
        return target_properties

    def enrich_placed_and_remaining_size(self, x):
        x = self.enrich_issue_size(x)
        x = self.enrich_placed_size(x)
        x['remaining_size_mln'] = x['issue_size_mln'] - x['placed_size_mln']
        return x

    def enrich_secondary_market_value(self, x):
        value_df = pd.concat(t.read_output()['volume'].unstack().T.between_time(dt.time(10), dt.time(19)) for t in self.requires()[Candles])
        x['mean_volume_mln_per_hour'] = x['secid'].map(value_df.fillna(0).mean()).fillna(0) * 1e-3
        return x

    def enrich_plan_runrate(self, x, target_quarter):
        plan_history_df = self.requires()[MinfinPlanHistory].read_output()[['date', 'quarter', 'nb_auctions', 'maturity_bucket', 'amount_mln', 'amount_mln_per_auction']]
        auctions_df = self.requires()[AuctionResultsHistory].read_output()
        # TODO 2026 plan is in value terms rather than notional, should generalize below
        auctions_df['years_to_maturity'] = (auctions_df['maturity_date'] - auctions_df['date']) / pd.Timedelta(days=365)

        def get_actual_plan(target_quarter_):
            return plan_history_df[(plan_history_df['quarter'] == target_quarter_) & (plan_history_df['date'] <= self.prediction_ts)].query('date == date.max()')

        plan_df = get_actual_plan(target_quarter)
        if plan_df.empty:
            plan_quarter = target_quarter - 1
            self.logger.info('Plans for target are not yet released, falling back to %s', plan_quarter)
            plan_df = get_actual_plan(plan_quarter)
        assert not plan_df.empty

        plan_df = plan_df.set_index('maturity_bucket')

        quarter_auctions_df = auctions_df[(auctions_df['quarter'] == target_quarter) & (auctions_df['result_release_ts'] <= self.prediction_ts)].copy()
        quarter_auctions_df['maturity_bucket'] = pd.cut(quarter_auctions_df['years_to_maturity'], plan_df.index)
        placed_mln = quarter_auctions_df.groupby('maturity_bucket', observed=False)['placed_size_mln'].sum()
        remaining_mln = plan_df['amount_mln'] - placed_mln

        x['nb_auctions_so_far'] = quarter_auctions_df['date'].nunique()
        x['nb_auctions_remaining'] = plan_df['nb_auctions'].iloc[0] - x['nb_auctions_so_far']
        x['plan_amount_mln_per_auction'] = plan_df['amount_mln_per_auction'].sum()
        x['plan_mat_bucket'] = pd.cut(x['years_to_maturity'], plan_df.index)
        x['plan_amount_mln_per_auction_mat_bucket'] = x['plan_mat_bucket'].map(plan_df['amount_mln_per_auction']).astype(float)
        x['required_run_rate_mln'] = remaining_mln.sum() / x['nb_auctions_remaining']
        x['required_run_rate_mln_mat_bucket'] = x['plan_mat_bucket'].map(remaining_mln).astype(float) / x['nb_auctions_remaining']

        x['all_bonds_last_auction_placed_size_mln'] = auctions_df[auctions_df['result_release_ts'] <= self.prediction_ts].query('date == date.max()')['placed_size_mln'].sum()
        return x

    def enrich_derived_metrics(self, x):
        x['years_to_maturity'] = (x['MATDATE'] - x['target_date']) / pd.Timedelta(days=365)
        x['years_since_registry'] = (x['target_date'] - x['REGISTRY_DATE']) / pd.Timedelta(days=365)
        x['years_since_last_issue'] = (x['target_date'] - x['issue_size_ts']) / pd.Timedelta(days=365)
        x['nb_auctions_since_prev1_placement'] = x['target_ordinal'] - x['prev1_auc_ordinal']
        x['nb_auctions_since_prev2_placement'] = x['target_ordinal'] - x['prev2_auc_ordinal']
        return x

    def produce_output(self) -> pd.DataFrame:
        x = self.requires()[BondsDescription].read_output().reset_index()[['series', 'secid', 'REGISTRY_DATE', 'ISSUEDATE', 'ISSUESIZE', 'MATDATE', 'SHORT_TYPE']]
        x['prediction_ts'] = self.prediction_ts

        x = self.enrich_placed_and_remaining_size(x)
        x = self.enrich_placement_end_date(x)
        assert x.notna().all().all()
        x = self.enrich_gcurve(x)
        x = self.enrich_prev_auctions_results(x)
        x = self.enrich_aggregated_auctions_results(x)
        x = self.enrich_secondary_market_value(x)
        target_properties = self.define_target_properties()
        x = x.assign(**target_properties)
        x = self.enrich_derived_metrics(x)
        x = self.hard_filtering(x)
        x = self.enrich_plan_runrate(x, target_properties['target_quarter'])
        xy = self.enrich_target_value(x)
        return xy


class CalibParamsMixin:
    target_type = luigi.EnumParameter(enum=TargetType, default=TargetType.FIRST)
    loss_function = luigi.Parameter(default='QuerySoftMax')
    eval_metric = luigi.Parameter(default='QuerySoftMax')
    features_override = luigi.TupleParameter(default=())
    regime_switch = luigi.BoolParameter(default=True)


class AuctionModelCalibration(DailyPickleTask, CalibParamsMixin):

    def file_name(self):
        to_hash = [self.eval_metric, self.regime_switch] + list(self.features_override)
        to_hash = str(to_hash)
        ending = hashlib.md5(to_hash.encode()).hexdigest()[:5]
        return f'model_{self.target_type.name}_{self.loss_function.replace(':', '_')}_{ending}'

    @staticmethod
    def make_pool(xy, features, cat_features, target_date_float_split=None):
        if target_date_float_split:
            xy = xy.copy()
            xy['target_date_float'] = xy['target_date_float'].clip(target_date_float_split, target_date_float_split + 0.01)
        return Pool(
            data=xy[features],
            label=xy['target'],
            group_id=xy['prediction_id'],
            cat_features=cat_features
        )

    def requires(self):
        dates = get_mos_dates_range(self.date, CALIB_WINDOW)
        return [self.clone(AuctionModelSample, date=date, time_of_day=dt.timedelta(hours=23)) for date in dates if date >= dt.date(2023, 1, 1)]

    @property
    def features_to_use(self):
        if len(self.features_override) == 0:
            return ['nb_auctions_since_prev1_placement',
                    'nb_auctions_since_prev2_placement',
                    'years_since_last_issue',
                    'years_since_registry',
                    'mean_volume_mln_per_hour',
                    'remaining_size_mln',
                    '90d_nb_auctions',
                    '90d_placed_size_mln_rank',
                    '90d_placed_size_mln/90d_demand_mln',
                    'correction',
                    'spread',
                    'all_bonds_last_auction_placed_size_mln/plan_amount_mln_per_auction',
                    'SHORT_TYPE']
        else:
            return list(self.features_override)

    @staticmethod
    def finalize_xy(xy):
        xy['prediction_id'] = xy['prediction_ts'].rank(method='dense').astype(int)
        xy['target_date_id'] = xy['target_date'].rank(method='dense').astype(int)
        xy['years_since_last_issue_rank'] = xy.groupby('prediction_id')['years_since_last_issue'].rank(pct=True)
        xy['years_since_registry_rank'] = xy.groupby('prediction_id')['years_since_registry'].rank(pct=True)
        xy['mean_volume_mln_per_hour_rank'] = xy.groupby('prediction_id')['mean_volume_mln_per_hour'].rank(pct=True)
        xy['remaining_size_mln_rank'] = xy.groupby('prediction_id')['remaining_size_mln'].rank(pct=True)
        xy['90d_placed_size_mln_rank'] = xy.groupby('prediction_id')['90d_placed_size_mln'].rank(pct=True)
        xy['remaining_size_mln/issue_size_mln'] = xy['remaining_size_mln'] / xy['issue_size_mln']
        xy['required_run_rate_mln/plan_amount_mln_per_auction'] = xy['required_run_rate_mln'] / xy['plan_amount_mln_per_auction']
        xy['remaining_size_mln/plan_amount_mln_per_auction'] = xy['remaining_size_mln'] / xy['plan_amount_mln_per_auction']
        xy['90d_placed_size_mln/plan_amount_mln_per_auction'] = xy['90d_placed_size_mln'] / xy['plan_amount_mln_per_auction']
        xy['90d_placed_size_mln/90d_demand_mln'] = xy['90d_placed_size_mln'] / xy['90d_demand_mln']
        xy['90d_placed_size_mln/90d_demand_mln'] = xy['90d_placed_size_mln'] / xy['90d_demand_mln']
        xy['all_bonds_last_auction_placed_size_mln/plan_amount_mln_per_auction'] = xy['all_bonds_last_auction_placed_size_mln'] / xy['plan_amount_mln_per_auction']
        xy['target_date_float'] = (xy['target_date'] - pd.Timestamp('20000101')) / pd.Timedelta(days=365.25)

        xy = xy.fillna({'90d_placed_size_mln/90d_demand_mln': 0.,
                        'correction': 0.,
                        'spread': 0.,
                        'nb_auctions_since_prev1_placement': 1000.,
                        'nb_auctions_since_prev2_placement': 1000.})
        return xy

    def produce_output(self) -> pd.DataFrame:
        xy = pd.concat([t.read_output() for t in self.requires()]).reset_index(drop=True)

        # filter out not yet known data
        xy = xy[xy['target_announce_ts'].notna() & (xy['target_announce_ts'] <= self.eod_timestamp)].copy()

        xy = self.finalize_xy(xy)

        features = self.features_to_use
        if self.regime_switch:
            features += ['target_date_float']

        cat_features = xy[features].select_dtypes(object).columns.to_list()

        # downsampling - taking last prediction_ts for a given target_date, as samples for same target_date are highly correlated
        xy = xy[xy.groupby('target_date')['prediction_ts'].transform('max') == xy['prediction_ts']].copy()

        train_mask = xy['target_date_id'].rank(method='dense', pct=True) <= 0.8
        test_mask = ~train_mask

        for mask_name in ['train_mask', 'test_mask']:
            mask = eval(mask_name)
            self.logger.warning(f'{mask_name}: {xy[mask].shape} {xy[mask]['target_date'].min()} - {xy[mask]['target_date'].max()}')

        CatBoostModel = CatBoostClassifier if self.is_classification() else CatBoostRanker

        def objective(trial):
            params = {
                "loss_function": self.loss_function,
                "iterations": 2000,
                "depth": trial.suggest_int("depth", 4, 8),
                "custom_metric": ["PrecisionAt:top=1", "PrecisionAt:top=2", "PrecisionAt:top=3", "PrecisionAt:top=4", "PrecisionAt:top=5"],
                "eval_metric": self.eval_metric,
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 50.0, log=True),
                "nan_mode": "Forbidden",
                "random_seed": 42,
                "allow_writing_files": False,
                "train_dir": None,
                "verbose": False
            }
            if self.is_classification():
                params['class_weights'] = {0.0: 1, 1.0: 3.0}
            if self.regime_switch:
                target_date_float_split = trial.suggest_float("target_date_float_split", xy[train_mask]['target_date_float'].min(), xy[train_mask]['target_date_float'].max())
            else:
                target_date_float_split = None

            train_pool = self.make_pool(xy[train_mask], features, cat_features, target_date_float_split=target_date_float_split)
            test_pool  = self.make_pool(xy[test_mask],  features, cat_features, target_date_float_split=target_date_float_split)



            model = CatBoostModel(**params)
            model.fit(train_pool, eval_set=test_pool, use_best_model=True, early_stopping_rounds=100)

            feature_importance = model.get_feature_importance(type="LossFunctionChange", data=train_pool)
            feature_importance = pd.Series(feature_importance, index=features).sort_values(ascending=False)

            best_score = model.get_best_score()

            trial.set_user_attr("metrics", best_score)
            trial.set_user_attr("params", params)
            trial.set_user_attr("best_iteration", model.get_best_iteration())
            trial.set_user_attr("tree_count", model.tree_count_)
            trial.set_user_attr("feature_importance", feature_importance)

            return best_score['validation'][self.eval_metric] * (-1 if self.is_classification() else 1)


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=4)

        # fit ranker with best hyperparams
        best_trial = study.best_trial
        model_params = dict(best_trial.user_attrs['params'])
        model_params['iterations'] = best_trial.user_attrs['tree_count']
        ranker = CatBoostModel(**model_params)
        target_date_float_split = best_trial.params['target_date_float_split'] if self.regime_switch else None
        full_pool = self.make_pool(xy, features, cat_features, target_date_float_split=target_date_float_split)

        ranker.fit(full_pool)

        # fit score -> probability, quick and dirty
        raw_scores = ranker.predict(full_pool)
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(raw_scores, full_pool.get_label())

        return ranker, iso_reg, study.trials_dataframe()

    def is_classification(self):
        return self.loss_function == 'Logloss'


class RecalibrationMode(Enum):
    DAILY = auto()
    QUARTERLY = auto()

    def get_calib_date(self, input_date: dt.date) -> dt.date:
        if self == RecalibrationMode.DAILY:
            return shift_mos_date(input_date, -1)
        elif self == RecalibrationMode.QUARTERLY:
            quarter_start = pd.Period(input_date, freq='Q').start_time.date()
            return shift_mos_date(quarter_start, -1)


class AuctionModelInference(InMemoryTask, DailyMixin, CalibParamsMixin):
    time_of_day = luigi.TimeDeltaParameter()
    recalibration_mode = luigi.EnumParameter(enum=RecalibrationMode)

    def requires(self):
        calib_date = self.recalibration_mode.get_calib_date(self.date)
        tasks = [self.clone(AuctionModelSample), self.clone(AuctionModelCalibration, date=calib_date)]
        return to_dict_by_class(tasks)

    def produce_output(self) -> pd.DataFrame:
        sample_df = self.requires()[AuctionModelSample].read_output()
        model, iso_reg = self.requires()[AuctionModelCalibration].read_output()[:2]

        sample_df = AuctionModelCalibration.finalize_xy(sample_df)
        is_ranker = isinstance(model, CatBoostRanker)
        score_calculator = model.predict if is_ranker else lambda s: model.predict_proba(s)[:, list(model.classes_).index(1.)]
        sample_df['score'] = score_calculator(sample_df[model.feature_names_])
        sample_df['proba'] = iso_reg.predict(sample_df['score']) if is_ranker else sample_df['score']
        return sample_df.sort_values(['score', 'years_to_maturity'], ascending=[False, False]).set_index('series', drop=True)

    def is_calibration_complete(self):
        return self.requires()[AuctionModelCalibration].complete()


class AuctionModelEmail(DailyPickleTask, CalibParamsMixin):
    time_of_day = luigi.TimeDeltaParameter()

    @property
    def cutoff_time(self):
        return dt.time(0, 0)

    def file_name(self):
        return 'auction_model_email_was_sent'

    def requires(self):
        return self.clone(AuctionModelInference, recalibration_mode=RecalibrationMode.QUARTERLY)

    def produce_output(self):
        df = self.requires().read_output().rename(columns={'prev1_auc_date': 'prev_auction_date'})
        target_date, = df['target_date'].unique()

        feature_names = self.requires().requires()[AuctionModelCalibration].read_output()[0].feature_names_
        display_cols = ['score', 'years_to_maturity', 'prev_auction_date', 'remaining_size_mln', 'SHORT_TYPE']
        df = df[[c for c in display_cols if c not in feature_names] + feature_names]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / f"Full_Prediction_Report_{target_date.strftime('%Y%m%d')}_asof_{self.date.strftime('%Y%m%d')}.xlsx"
            df.to_excel(file_path)

            outlook = win32.Dispatch('outlook.application')
            mail = outlook.CreateItem(0)

            if self.date.weekday() == calendar.Day.THURSDAY:
                recipients = ['daniil.balkhin@mncap.ru', 'evgeny.garipov@mncap.ru', 'sergei.samoliuk@mncap.ru']
            else:
                recipients = ['sergei.samoliuk@mncap.ru']

            mail.To = "; ".join(recipients)
            mail.Subject = f"Топ-5 ОФЗ к аукциону {target_date.strftime('%Y-%m-%d')}"

            df_display = df[display_cols].head(5).copy()

            df_display["remaining_size_mln"] = df_display["remaining_size_mln"].map('{:,.0f}'.format).str.replace(',', ' ')
            df_display["score"] = df_display["score"].map('{:.3f}'.format)
            df_display["years_to_maturity"] = df_display["years_to_maturity"].map('{:.1f}'.format)
            df_display["prev_auction_date"] = df_display["prev_auction_date"].dt.strftime('%Y-%m-%d').fillna('-')
            df_display.index.name = None
            html_table = df_display.to_html(index=True)

            mail.HTMLBody = f"""
            <html>
            <head>
                <style>
                    table {{ 
                        border-collapse: collapse; 
                        font-family: "Segoe UI", Arial, sans-serif; 
                        font-size: 13px; 
                        border: none;
                    }}
                    th {{ 
                        background-color: #f4f6f9; 
                        border-bottom: 1px solid #dee2e6; 
                        padding: 10px 15px; 
                        text-align: right; 
                        color: #495057; 
                    }}
                    td {{ 
                        border-bottom: 1px solid #ebedf0;
                        padding: 8px 15px; 
                        text-align: right; 
                        white-space: nowrap; 
                        color: #212529;
                    }}
                    tr:last-child td {{
                        border-bottom: 1px solid #ebedf0 !important;
                    }}
                    th:last-child, td:last-child {{ text-align: left; }}
                    td:first-child {{ font-weight: 600; background-color: #fafafa; }}
                    .footer {{ color: #adb5bd; font-size: 11px; margin-top: 15px; }}
                </style>
            </head>
            <body>
                {html_table}
                <p style="color: gray; font-size: 11px; margin-top: 20px;">
                    Autogenerated at {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </body>
            </html>
            """

            mail.Attachments.Add(str(file_path.absolute()))

            mail.Send()

        return True