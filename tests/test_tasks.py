"""Tests for Luigi tasks that read from DATA_INPUT_DIR."""
import datetime as dt

import pytest

from src.tasks import MinfinPlanHistory, MinfinDebtStructure, IssueSizeReopenings


class TestDataInputBackedTasks:

    @pytest.fixture(scope="class")
    def temp_out_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("luigi_output")

    def test_minfin_debt_structure(self, temp_out_dir):
        task = MinfinDebtStructure(date=dt.date(2026, 2, 26), out_dir=temp_out_dir)
        output = task.produce_output()
        assert output is not None

    def test_minfin_plan_history(self, temp_out_dir):
        task = MinfinPlanHistory(out_dir=temp_out_dir)
        output = task.produce_output()
        assert output is not None

    def test_issue_size_reopenings(self, temp_out_dir):
        task = IssueSizeReopenings(out_dir=temp_out_dir)
        output = task.produce_output()
        assert output is not None
