import pandas.api.extensions as pd_ext

class XAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def cols(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        return self._obj[columns + list(self._obj.columns.difference(columns, sort=False))]

    def add_prefix(self, prefix, include_cols=None, except_cols=None):
        if except_cols is None:
            except_cols = []

        if include_cols is None:
            include_cols = list(self._obj.columns.difference(except_cols))

        selected_cols = self._obj.columns.intersection(include_cols + except_cols)  # keep original columns order
        rename_dict = {col: prefix + col for col in include_cols}

        return self._obj[self._obj.columns.intersection(selected_cols)].rename(columns=rename_dict)


def register_x_accessor():
    pd_ext.register_dataframe_accessor("x")(XAccessor)