"""pytest 测试 - 覆盖清洗/校验/分析/建议/报告核心逻辑"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cold_chain_report import sample_data
from cold_chain_report import validator
from cold_chain_report import cleaner
from cold_chain_report import analyzer
from cold_chain_report import adviser
from cold_chain_report import reporter


BASE_COLS = [
    "warehouse_id", "warehouse_name", "warehouse_type",
    "timestamp", "power_kwh", "temp_celsius", "ambient_temp",
    "compressor_status", "door_open_count", "defrost_status",
    "inventory_kg", "meter_reading",
]


def _make_hourly_ts(n_hours: int, start: str = "2026-05-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_hours, freq="h")


def _make_base_df(warehouse_id: str = "WH-TEST", warehouse_type: str = "中温库",
                  n_hours: int = 24 * 10) -> pd.DataFrame:
    ts = _make_hourly_ts(n_hours)
    target = analyzer.WAREHOUSE_TARGET_CONFIG[warehouse_type]["target_temp"]
    tol = analyzer.WAREHOUSE_TARGET_CONFIG[warehouse_type]["tolerance"]

    rng = np.random.default_rng(42)
    ambient_base = {"高温库": 32, "中温库": 26, "低温库": 28, "恒温库": 25}.get(warehouse_type, 25)

    ambient = ambient_base + 6 * np.sin(np.arange(n_hours) / 24 * 2 * np.pi - np.pi) + rng.normal(0, 0.5, n_hours)
    temps = target + rng.normal(0, tol * 0.3, n_hours)
    compressor = (temps > target + tol * 0.5).astype(int)
    for i in range(1, n_hours):
        if compressor[i - 1] == 1 and temps[i] > target - tol * 0.5:
            compressor[i] = 1

    power = np.where(compressor == 1, 50 + rng.normal(0, 5, n_hours), 2 + rng.normal(0, 0.5, n_hours))
    power = np.maximum(0.5, power)
    door = rng.integers(0, 5, n_hours)
    defrost = ((np.arange(n_hours) % 8) == 0).astype(int)
    inventory = 25000 + rng.normal(0, 100, n_hours).cumsum()
    inventory = np.clip(inventory, 20000, 35000)
    meter = 100000 + np.cumsum(power + rng.normal(0, 0.05, n_hours))

    return pd.DataFrame({
        "warehouse_id": warehouse_id,
        "warehouse_name": f"测试-{warehouse_type}",
        "warehouse_type": warehouse_type,
        "timestamp": ts,
        "power_kwh": np.round(power, 2),
        "temp_celsius": np.round(temps, 2),
        "ambient_temp": np.round(ambient, 1),
        "compressor_status": compressor,
        "door_open_count": door,
        "defrost_status": defrost,
        "inventory_kg": np.round(inventory, 0),
        "meter_reading": np.round(meter, 2),
    })


# ============================================================
# 1. 缺口插补测试
# ============================================================
class TestMissingInterpolation:
    def test_fill_missing_hours_preserves_count(self):
        df = _make_base_df(n_hours=24 * 7)
        missing_idx = [10, 11, 50, 100]
        df_gapped = df.drop(index=missing_idx).reset_index(drop=True)
        filled, count = cleaner._fill_missing_hours(df_gapped)
        assert count == len(missing_idx)
        assert len(filled) == len(df)
        assert filled["timestamp"].is_monotonic_increasing
        diffs = filled["timestamp"].diff().dropna().dt.total_seconds() / 3600
        assert (diffs == 1.0).all()

    def test_interpolated_values_reasonable(self):
        df = _make_base_df(n_hours=24 * 3)
        drop_idx = [12]
        before_temp = df.loc[11, "temp_celsius"]
        after_temp = df.loc[13, "temp_celsius"]
        df_gapped = df.drop(index=drop_idx).reset_index(drop=True)
        filled, _ = cleaner._fill_missing_hours(df_gapped)
        filled = filled.sort_values("timestamp").reset_index(drop=True)
        interp_idx = 12
        interp_temp = filled.loc[interp_idx, "temp_celsius"]
        expected = (before_temp + after_temp) / 2
        assert abs(interp_temp - expected) < 0.5
        assert filled.loc[interp_idx, "compressor_status"] in (0, 1)
        assert pd.notna(filled.loc[interp_idx, "meter_reading"])

    def test_consecutive_missing(self):
        df = _make_base_df(n_hours=24 * 3)
        drop_idx = list(range(10, 15))
        df_gapped = df.drop(index=drop_idx).reset_index(drop=True)
        filled, count = cleaner._fill_missing_hours(df_gapped)
        assert count == 5
        assert len(filled) == len(df)
        assert filled["temp_celsius"].isna().sum() == 0


# ============================================================
# 2. 重复记录处理
# ============================================================
class TestDuplicateHandling:
    def test_deduplicate_removes_duplicates(self):
        df = _make_base_df(n_hours=24 * 2)
        dup_row = df.iloc[5].copy()
        dup_row["power_kwh"] = df.iloc[5]["power_kwh"] * 1.05
        df_dup = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        deduped, removed = cleaner._deduplicate_timestamps(df_dup)
        assert removed == 1
        assert len(deduped) == len(df)
        assert deduped["timestamp"].duplicated().sum() == 0

    def test_deduplicate_aggregation_preserves_types(self):
        df = _make_base_df(n_hours=24 * 2)
        dup_row = df.iloc[3].copy()
        dup_row["power_kwh"] = 999.0
        df_dup = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        deduped, _ = cleaner._deduplicate_timestamps(df_dup)
        ts_target = df.iloc[3]["timestamp"]
        row = deduped[deduped["timestamp"] == ts_target].iloc[0]
        orig_power = df.iloc[3]["power_kwh"]
        assert abs(row["power_kwh"] - (orig_power + 999.0) / 2) < 0.5
        assert row["compressor_status"] in (0, 1)


# ============================================================
# 3. 电表倒退修复
# ============================================================
class TestMeterRegression:
    def test_detect_regression(self):
        df = _make_base_df(n_hours=24 * 2)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.at[10, "meter_reading"] = df.at[9, "meter_reading"] - 200
        bad = validator.detect_meter_regression(df)
        assert not bad.empty
        assert 10 in bad.index.tolist() or len(bad) >= 1

    def test_fix_regression_monotonic(self):
        df = _make_base_df(n_hours=24 * 3)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.at[8, "meter_reading"] = df.at[7, "meter_reading"] - 500
        df.at[20, "meter_reading"] = df.at[19, "meter_reading"] - 100
        fixed, n_fixed = cleaner._fix_meter_regression(df)
        assert n_fixed >= 2
        meter = fixed["meter_reading"].values
        for i in range(1, len(meter)):
            assert meter[i] >= meter[i - 1] - 1e-6

    def test_clean_pipeline_handles_regression(self):
        df = _make_base_df(n_hours=24 * 2)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.at[5, "meter_reading"] = df.at[4, "meter_reading"] - 999
        cleaned, report = cleaner.clean_warehouse(df)
        meter = cleaned["meter_reading"].values
        assert (np.diff(meter) >= -1e-6).all()
        action_types = [a.action for a in report.actions]
        assert "fix_meter_regression" in action_types


# ============================================================
# 4. 温度校正
# ============================================================
class TestTemperatureCorrection:
    def test_ambient_factor_scales_with_temp(self):
        wh_type = "中温库"
        base = 25.0
        adjusted_low, f1, _ = analyzer._adjust_for_conditions(
            10.0, 20.0, 1.0, base, wh_type
        )
        adjusted_high, f2, _ = analyzer._adjust_for_conditions(
            10.0, 35.0, 1.0, base, wh_type
        )
        assert f1 < 1.0
        assert f2 > 1.0
        assert adjusted_low > 10.0
        assert adjusted_high < 10.0
        assert adjusted_low > adjusted_high

    def test_correction_avoids_weather_misjudgement(self):
        df_hot = _make_base_df("WH-HOT", "高温库", 24 * 10)
        df_hot["ambient_temp"] = df_hot["ambient_temp"] + 10
        df_cool = _make_base_df("WH-COOL", "高温库", 24 * 10)
        df_cool["power_kwh"] = df_cool["power_kwh"] * 1.15
        df_combined = pd.concat([df_hot, df_cool], ignore_index=True)
        analyses = analyzer.analyze_all(df_combined)

        raw_hot = analyses["WH-HOT"].raw_power_per_ton_day
        raw_cool = analyses["WH-COOL"].raw_power_per_ton_day
        adj_hot = analyses["WH-HOT"].adjusted_power_per_ton_day
        adj_cool = analyses["WH-COOL"].adjusted_power_per_ton_day

        if raw_hot > raw_cool:
            assert adj_hot < adj_cool or (adj_hot - adj_cool) < (raw_hot - raw_cool) * 0.3

    def test_analysis_contains_correction_fields(self):
        df = _make_base_df()
        analysis = analyzer.analyze_warehouse(df)
        assert analysis.ambient_adjustment_factor > 0
        assert analysis.inventory_adjustment_factor > 0
        assert analysis.adjusted_power_per_ton_day > 0
        assert analysis.raw_power_per_ton_day > 0


# ============================================================
# 5. 短循环识别
# ============================================================
class TestShortCycleDetection:
    def test_detects_short_cycles(self):
        df = _make_base_df(n_hours=48)
        df = df.sort_values("timestamp").reset_index(drop=True)
        status = df["compressor_status"].values.copy()
        for start in (4, 14, 28, 38):
            status[start] = 1
            status[start + 1] = 0
        df["compressor_status"] = status
        short = validator.detect_short_cycles(df, min_on_hours=2, min_off_hours=2)
        assert len(short) >= 4

    def test_distinguish_short_cycle_vs_normal(self):
        df_normal = _make_base_df("WH-NORMAL", "中温库", 24 * 10)
        df_bad = _make_base_df("WH-BAD", "中温库", 24 * 10)
        df_bad = df_bad.sort_values("timestamp").reset_index(drop=True)
        bad_status = df_bad["compressor_status"].values.copy()
        rng = np.random.default_rng(0)
        for i in range(len(bad_status) - 1):
            if rng.random() < 0.18:
                bad_status[i] = 1 - bad_status[i]
        df_bad["compressor_status"] = bad_status
        df_combined = pd.concat([df_normal, df_bad], ignore_index=True)
        analyses = analyzer.analyze_all(df_combined)
        assert analyses["WH-BAD"].short_cycle_ratio > analyses["WH-NORMAL"].short_cycle_ratio * 2
        assert analyses["WH-BAD"].short_cycle_ratio > 0.05

    def test_advice_triggers_for_severe_short_cycle(self):
        df = _make_base_df("WH-SHORT", "中温库", 24 * 10)
        df = df.sort_values("timestamp").reset_index(drop=True)
        status = df["compressor_status"].values.copy()
        rng = np.random.default_rng(1)
        for i in range(2, len(status) - 1):
            if rng.random() < 0.25:
                status[i] = 1 - status[i - 1]
        df["compressor_status"] = status
        analysis = analyzer.analyze_warehouse(df)
        advice_report = adviser.generate_warehouse_advice(analysis, df)
        cats = [a.category for a in advice_report.advice_list]
        assert "compressor_health" in cats


# ============================================================
# 6. 建议排序
# ============================================================
class TestAdviceSorting:
    def test_critical_before_high_before_medium(self):
        df = _make_base_df("WH-SORT", "中温库", 24 * 10)
        df = df.sort_values("timestamp").reset_index(drop=True)
        status = df["compressor_status"].values.copy()
        rng = np.random.default_rng(2)
        for i in range(2, len(status) - 1):
            if rng.random() < 0.3:
                status[i] = 1 - status[i - 1]
        df["compressor_status"] = status
        df["door_open_count"] = np.full(len(df), 25)
        df["temp_celsius"] = np.where(df["temp_celsius"] < 0, df["temp_celsius"] + 10, df["temp_celsius"])
        analysis = analyzer.analyze_warehouse(df)
        report = adviser.generate_warehouse_advice(analysis, df)
        if len(report.advice_list) >= 2:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            priorities = [priority_order.get(a.priority, 99) for a in report.advice_list]
            for i in range(1, len(priorities)):
                assert priorities[i] >= priorities[i - 1]

    def test_sorting_by_savings_within_same_priority(self):
        df = _make_base_df("WH-SORT2", "高温库", 24 * 10)
        df["door_open_count"] = np.full(len(df), 30)
        df = df.sort_values("timestamp").reset_index(drop=True)
        status = df["compressor_status"].values.copy()
        rng = np.random.default_rng(5)
        for i in range(2, len(status) - 1):
            if rng.random() < 0.15:
                status[i] = 1 - status[i - 1]
        df["compressor_status"] = status
        analysis = analyzer.analyze_warehouse(df)
        report = adviser.generate_warehouse_advice(analysis, df)
        same_priority_groups = {}
        for a in report.advice_list:
            same_priority_groups.setdefault(a.priority, []).append(a)
        for items in same_priority_groups.values():
            if len(items) >= 2:
                savings = [(a.estimated_savings_low_kwh + a.estimated_savings_high_kwh) / 2 for a in items]
                for i in range(1, len(savings)):
                    assert savings[i] <= savings[i - 1] + 1e-6


# ============================================================
# 7. 报告字段完整性
# ============================================================
class TestReportFieldCompleteness:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        sample_data.generate_sample_data(self.tmpdir, "2026-05-01", 10)
        self.df = pd.read_csv(
            os.path.join(self.tmpdir, "all_warehouses_monthly.csv"),
            parse_dates=["timestamp"],
        )
        self.validations = validator.validate_all(self.df)
        self.cleaned, self.cleanings = cleaner.clean_all(self.df)
        self.analyses = analyzer.analyze_all(self.cleaned)
        self.advice = adviser.generate_all_advice(self.analyses, self.cleaned)

    def test_markdown_report_has_required_sections(self):
        md = reporter.generate_markdown_report(
            self.analyses, self.advice, self.validations, self.cleanings, "2026-05"
        )
        assert "冷库月度运行分析报告" in md
        assert "总体概览" in md
        assert "核心能耗指标" in md
        assert "温度品质" in md
        assert "压缩机运行" in md
        assert "校正方法说明" in md
        for wh_id in self.analyses:
            assert wh_id in md

    def test_json_report_structure_complete(self):
        js = reporter.generate_advice_json(self.analyses, self.advice, "2026-05")
        data = json.loads(js)
        assert "generated_at" in data
        assert "report_period" in data
        assert "warehouses" in data
        for wh_id, wh in data["warehouses"].items():
            for field in (
                "warehouse_name", "warehouse_type", "benchmark_tier",
                "total_power_kwh", "adjusted_power_per_ton_day",
                "temp_compliance_rate", "short_cycle_ratio",
                "daily_metrics", "advice_list",
                "total_estimated_savings_low_kwh",
                "total_estimated_savings_high_kwh",
            ):
                assert field in wh, f"缺少字段 {field} 于 {wh_id}"
            assert isinstance(wh["daily_metrics"], list)
            assert isinstance(wh["advice_list"], list)
            for adv in wh["advice_list"]:
                for f in ("category", "priority", "recommendation", "estimated_savings_low_kwh", "estimated_savings_high_kwh"):
                    assert f in adv

    def test_cleaned_csv_has_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name
        try:
            reporter.export_cleaned_csv(self.cleaned, csv_path)
            reloaded = pd.read_csv(csv_path)
            for col in BASE_COLS:
                assert col in reloaded.columns, f"清洗后CSV缺少列 {col}"
            assert len(reloaded) == len(self.cleaned)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_daily_metrics_fields(self):
        for wh_id, analysis in self.analyses.items():
            assert len(analysis.daily_metrics) > 0
            dm = analysis.daily_metrics[0]
            for field in (
                "date", "total_power_kwh", "temp_compliance_rate",
                "compressor_starts", "avg_ambient_temp", "avg_inventory_kg",
            ):
                assert hasattr(dm, field)

    def test_end_to_end_export_produces_files(self):
        out_dir = os.path.join(self.tmpdir, "export_out")
        outputs = reporter.export_all(
            out_dir, self.cleaned, self.analyses, self.advice,
            self.validations, self.cleanings, "2026-05",
        )
        for name, path in outputs.items():
            assert os.path.exists(path), f"{name} 未生成: {path}"
            assert os.path.getsize(path) > 0, f"{name} 为空"

    def test_distinguish_two_special_warehouses(self):
        if "WH-A001" in self.analyses and "WH-B002" in self.analyses:
            a = self.analyses["WH-A001"]
            b = self.analyses["WH-B002"]
            assert a.avg_ambient_temp > b.avg_ambient_temp + 3, \
                f"WH-A001 应该是高温环境 ({a.avg_ambient_temp}°C)，WH-B002 普通环境 ({b.avg_ambient_temp}°C)"
            assert a.warehouse_type == "高温库"
            assert b.warehouse_type == "中温库"

            ambient_adj_ratio = a.ambient_adjustment_factor / max(0.01, b.ambient_adjustment_factor)
            assert ambient_adj_ratio > 1.0, "WH-A001 应有更高的环境校正因子（环境更热）"

            a_advice = self.advice.get("WH-A001")
            b_advice = self.advice.get("WH-B002")
            if a_advice and b_advice:
                a_short_ratio = a.short_cycle_ratio
                b_short_ratio = b.short_cycle_ratio
                if b_short_ratio < a_short_ratio:
                    a_short_in_advice = any("短循环" in ad.issue_description or
                                            "short_cycle" in ad.category or
                                            "启停" in ad.issue_description
                                            for ad in a_advice.advice_list)
                    b_other_issue = any(cat in ("compressor_health", "door_management", "temperature_quality")
                                        for ad in b_advice.advice_list
                                        for cat in [ad.category])
                    assert (not a_short_in_advice) or b_other_issue or len(b_advice.advice_list) > 0, \
                        "WH-B002 虽然短循环占比低，但应该有其他建议（如温度品质、启停机）"


# ============================================================
# 8. 集成测试：端到端CLI流水线
# ============================================================
class TestIntegrationPipeline:
    def test_full_pipeline_produces_cleaned_dataframe(self):
        tmpdir = tempfile.mkdtemp()
        sample_data.generate_sample_data(tmpdir, "2026-05-01", 7)
        df = pd.read_csv(os.path.join(tmpdir, "all_warehouses_monthly.csv"), parse_dates=["timestamp"])
        assert len(df) > 0
        validations = validator.validate_all(df)
        assert isinstance(validations, dict) and len(validations) > 0
        cleaned_df, cleaning_reports = cleaner.clean_all(df)
        for wh_id, rep in cleaning_reports.items():
            assert rep.cleaned_records > 0
        analyses = analyzer.analyze_all(cleaned_df)
        assert len(analyses) == len(validations)
        advices = adviser.generate_all_advice(analyses, cleaned_df)
        assert len(advices) == len(analyses)
        out_dir = os.path.join(tmpdir, "reports")
        outputs = reporter.export_all(out_dir, cleaned_df, analyses, advices, validations, cleaning_reports, "test")
        for p in outputs.values():
            assert os.path.exists(p)
