"""Tests for Luigi tasks that read from DATA_INPUT_DIR."""
import datetime as dt
import pytest

from src.tasks import MinfinPlanHistory, MinfinDebtStructure, IssueSizeReopenings


class TestDataInputBackedTasks:
    """Test tasks that depend on files in data/input directory."""

    @pytest.fixture
    def test_date(self):
        """Common test date for tasks."""
        return dt.date(2026, 2, 26)

    def test_minfin_plan_history(self, test_date):
        """Test MinfinPlanHistory task runs without failure."""
        task = MinfinPlanHistory(date=test_date)
        output = task.produce_output()
        assert output is not None
        assert len(output) > 0

    def test_minfin_debt_structure(self, test_date):
        """Test MinfinDebtStructure task runs without failure."""
        task = MinfinDebtStructure(date=test_date)
        output = task.produce_output()
        assert output is not None
        assert len(output) > 0

    def test_issue_size_reopenings(self, test_date):
        """Test IssueSizeReopenings task runs without failure."""
        task = IssueSizeReopenings(date=test_date)
        output = task.produce_output()
        assert output is not None
