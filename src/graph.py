import datetime as dt
import logging
import pickle
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import luigi
import pandas as pd

from src.qr import shift_mos_date


def get_task_group_class(task_group):
    if isinstance(task_group, luigi.Task):
        return task_group.__class__
    else:
        classes = set(task.__class__ for task in task_group)
        assert len(classes) == 1
        return next(iter(classes))

def to_dict_by_class(tasks):
    return {get_task_group_class(task_group): task_group for task_group in tasks}


class PickleTarget(luigi.LocalTarget):

    def read(self):
        with open(self.path, 'rb') as f:
            return pickle.load(f)

    def write(self, obj):
        # Ensure parent directory exists
        path = Path(self.path)
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


class InMemoryTarget(luigi.Target):
    _obj = None

    def exists(self):
        return self._obj is not None

    def read(self):
        assert self.exists()
        return self._obj

    def write(self, obj) -> None:
        assert not self.exists()
        self._obj = obj

    def remove(self):
        self._obj = None


class TaskLoggerAdapter(logging.LoggerAdapter):
    """Minimal adapter that adds task repr to log messages"""

    def process(self, msg, kwargs):
        task = self.extra['task']
        return f"[{task}] {msg}", kwargs


class LoggingMixin:

    @cached_property
    def logger(self):
        base_logger = logging.getLogger(__name__)
        return TaskLoggerAdapter(base_logger, {'task': self})

class OutDirMixin:
    out_dir = luigi.PathParameter()

class InMemoryTask(luigi.Task, LoggingMixin, OutDirMixin, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output = InMemoryTarget()

    def output(self):
        return self._output

    def read_output(self):
        return self.output().read().copy()

    def write_output(self, obj) -> None:
        self.validate_output(obj)
        self.output().write(obj)

    def validate_output(self, obj) -> None:
        if isinstance(obj, pd.DataFrame):
            assert not obj.empty, "DataFrame cannot be empty."

    def run(self):
        """Main task execution method."""
        obj = self.produce_output()
        self.write_output(obj)

    @abstractmethod
    def produce_output(self):
        pass


class DailyMixin:
    date = luigi.DateParameter()

    @property
    def sod_timestamp(self):
        return pd.to_datetime(self.date)

    @property
    def eod_timestamp(self):
        return pd.to_datetime(self.date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    @property
    def prev_date(self):
        return shift_mos_date(self.date, -1)

    def clone_previous(self, cls=None, **kwargs):
        kwargs['date'] = self.prev_date
        return self.clone(cls, **kwargs)


class DailyPickleTask(luigi.Task, DailyMixin, LoggingMixin, OutDirMixin, ABC):

    def is_final_output(self):
        return dt.datetime.now() > dt.datetime.combine(self.date, self.cutoff_time)

    @property
    def _output_path(self) -> Path:
        if self.is_final_output():
            output_dir = self.out_dir / str(self.date)
        else:
            output_dir = self.out_dir / 'temp' / str(self.date)
        filename = f"{self.file_name()}.pkl"
        return output_dir / filename

    @property
    def cutoff_time(self):
        return dt.time(23, 15)

    def output(self):
        return PickleTarget(self._output_path)

    @abstractmethod
    def file_name(self):
        pass

    @abstractmethod
    def produce_output(self):
        pass

    def postprocess_on_read(self, obj):
        return obj

    def read_output(self):
        return self.postprocess_on_read(self.output().read())

    def write_output(self, obj) -> None:
        self.validate_output(obj)
        self.output().write(obj)

    def validate_output(self, obj):
        if isinstance(obj, pd.DataFrame):
            assert not obj.empty, "DataFrame cannot be empty."

    def run(self):
        """Main task execution method."""
        df = self.produce_output()
        self.write_output(df)
