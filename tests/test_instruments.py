"""Tests for instruments module."""
import pandas as pd
import pytest

from src.instruments import OFZ, Id, symbol_to_series


class TestOFZ:
    """Test OFZ bond dictionary."""

    def test_ofz_access_by_number(self):
        """Test accessing OFZ bonds by number."""
        bond = OFZ[26238]
        assert isinstance(bond, Id)
        assert bond.secid == 'SU26238RMFS4'
        assert bond.isin == 'RU000A1038V6'

    def test_ofz_multiple_bonds(self):
        """Test multiple OFZ bonds."""
        assert OFZ[26230].secid == 'SU26230RMFS1'
        assert OFZ[26248].secid == 'SU26248RMFS3'
        assert OFZ[26243].secid == 'SU26243RMFS4'

    def test_ofz_contains_expected_bonds(self):
        """Test that OFZ dictionary contains expected bonds."""
        assert 26238 in OFZ
        assert 26230 in OFZ
        assert 25084 in OFZ
        assert len(OFZ) == 38  # Total number of bonds in the list


class TestId:
    """Test Id dataclass."""

    def test_id_dataclass_properties(self):
        """Test Id dataclass has correct properties."""
        id_obj = Id(secid='SU26238RMFS4', isin='RU000A1038V6')
        assert id_obj.secid == 'SU26238RMFS4'
        assert id_obj.isin == 'RU000A1038V6'


class TestSymbolToSeries:
    """Test symbol_to_series function."""

    def test_symbol_to_series_conversion(self):
        """Test converting symbol series to numeric series."""
        symbols = pd.Series(['SU26238RMFS4', 'SU26230RMFS1', 'SU25084RMFS3'])
        result = symbol_to_series(symbols)

        expected = pd.Series([26238, 26230, 25084])
        pd.testing.assert_series_equal(result, expected)

    def test_symbol_to_series_single_value(self):
        """Test conversion with single value."""
        symbols = pd.Series(['SU26238RMFS4'])
        result = symbol_to_series(symbols)

        assert result.iloc[0] == 26238
        assert result.dtype == int
