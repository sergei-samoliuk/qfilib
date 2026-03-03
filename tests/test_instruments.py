"""Tests for instruments module."""
import pandas as pd

from src.instruments import symbol_to_series


class TestSymbolToSeries:

    def test_symbol_to_series_conversion(self):
        symbols = pd.Series(['SU26238RMFS4', 'SU26230RMFS1', 'SU25084RMFS3'])
        result = symbol_to_series(symbols)

        expected = pd.Series([26238, 26230, 25084])
        pd.testing.assert_series_equal(result, expected)
