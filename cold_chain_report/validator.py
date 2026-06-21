"""数据校验模块 - 识别缺小时段、重复时间、明显错值、状态跳变、电表倒退"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    issue_type: str
    warehouse_id: str
    count: int
    details: List[str] = field(default_factory=list)
    severity: str = "warning"


@dataclass
class ValidationReport:
    warehouse_id: str
    total_records: int
    expected_records: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for issue in self.issues:
            summary[issue.issue_type] = summary.get(issue.issue_type, 0) + issue.count
        return summary


WAREHOUSE_TYPE_TEMP_RANGES = {
    "高温库": (0.0, 10.0),
    "中温库": (-25.0, -10.0),
    "低温库": (-45.0, -20.0),
    "恒温库": (-5.0, 8.0),
}

DEFAULT_TEMP_RANGE = (-60.0, 50.0)
DEFAULT_AMBIENT_RANGE = (-30.0, 60.0)
DEFAULT_POWER_RANGE = (0.0, 2000.0)
DEFAULT_INVENTORY_RANGE = (0.0, 1_000_000.0)


def _get_valid_temp_range(warehouse_type: str) -> Tuple[float, float]:
    return WAREHOUSE_TYPE_TEMP_RANGES.get(warehouse_type, DEFAULT_TEMP_RANGE)


def detect_missing_hours(df: pd.DataFrame) -> Tuple[List[pd.Timestamp], int]:
    if df.empty:
        return [], 0
    df_sorted = df.sort_values("timestamp")
    start = df_sorted["timestamp"].min().floor("h")
    end_raw = df_sorted["timestamp"].max().floor("h") + pd.Timedelta(hours=1)
    expected = pd.date_range(start=start, end=end_raw, freq="h", inclusive="left")
    actual_set = set(pd.to_datetime(df_sorted["timestamp"]).dt.floor("h"))
    missing = [ts for ts in expected if ts not in actual_set]
    expected_count = len(expected)
    return missing, expected_count


def detect_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0]
    dup_mask = df.duplicated(subset=["timestamp"], keep=False)
    return df[dup_mask].sort_values("timestamp")


def detect_outliers(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    results = {}
    if df.empty:
        return results

    warehouse_type = df["warehouse_type"].iloc[0] if "warehouse_type" in df.columns else ""
    temp_min, temp_max = _get_valid_temp_range(warehouse_type)

    temp_mask = (df["temp_celsius"] < temp_min) | (df["temp_celsius"] > temp_max)
    temp_mask |= df["temp_celsius"].isin([np.inf, -np.inf]) | df["temp_celsius"].isna()
    if temp_mask.any():
        results["temp_outliers"] = df[temp_mask]

    amb_mask = (df["ambient_temp"] < DEFAULT_AMBIENT_RANGE[0]) | (df["ambient_temp"] > DEFAULT_AMBIENT_RANGE[1])
    amb_mask |= df["ambient_temp"].isin([np.inf, -np.inf]) | df["ambient_temp"].isna()
    if amb_mask.any():
        results["ambient_outliers"] = df[amb_mask]

    pwr_mask = (df["power_kwh"] < DEFAULT_POWER_RANGE[0]) | (df["power_kwh"] > DEFAULT_POWER_RANGE[1])
    pwr_mask |= df["power_kwh"].isin([np.inf, -np.inf]) | df["power_kwh"].isna()
    if pwr_mask.any():
        results["power_outliers"] = df[pwr_mask]

    inv_mask = (df["inventory_kg"] < DEFAULT_INVENTORY_RANGE[0]) | (df["inventory_kg"] > DEFAULT_INVENTORY_RANGE[1])
    inv_mask |= df["inventory_kg"].isna()
    if inv_mask.any():
        results["inventory_outliers"] = df[inv_mask]

    comp_mask = ~df["compressor_status"].isin([0, 1])
    comp_mask |= df["compressor_status"].isna()
    if comp_mask.any():
        results["compressor_status_invalid"] = df[comp_mask]

    def_mask = ~df["defrost_status"].isin([0, 1])
    def_mask |= df["defrost_status"].isna()
    if def_mask.any():
        results["defrost_status_invalid"] = df[def_mask]

    door_mask = (df["door_open_count"] < 0) | df["door_open_count"].isna()
    if door_mask.any():
        results["door_count_invalid"] = df[door_mask]

    return results


def detect_state_jumps(df: pd.DataFrame, min_cycle_hours: int = 2) -> pd.DataFrame:
    if df.empty or len(df) < 3:
        return df.iloc[0:0]
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    status = df_sorted["compressor_status"].values
    jumps = []
    for i in range(1, len(status) - 1):
        if status[i] != status[i - 1] and status[i] != status[i + 1]:
            if status[i - 1] == status[i + 1]:
                jumps.append(i)
    if jumps:
        return df_sorted.iloc[jumps]
    return df.iloc[0:0]


def detect_short_cycles(df: pd.DataFrame, min_on_hours: int = 2, min_off_hours: int = 2) -> pd.DataFrame:
    if df.empty or len(df) < 4:
        return df.iloc[0:0]
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    status = df_sorted["compressor_status"].values
    timestamps = df_sorted["timestamp"].values

    short_indices = []
    run_start = 0
    current_status = status[0]

    for i in range(1, len(status)):
        if status[i] != current_status:
            duration_h = (timestamps[i] - timestamps[run_start]) / np.timedelta64(1, "h")
            duration_h = max(duration_h, (i - run_start))
            if current_status == 1 and duration_h < min_on_hours:
                for j in range(run_start, i):
                    short_indices.append(j)
            elif current_status == 0 and duration_h < min_off_hours:
                for j in range(run_start, i):
                    short_indices.append(j)
            run_start = i
            current_status = status[i]

    duration_h = (len(status) - run_start)
    if current_status == 1 and duration_h < min_on_hours:
        for j in range(run_start, len(status)):
            short_indices.append(j)
    elif current_status == 0 and duration_h < min_off_hours:
        for j in range(run_start, len(status)):
            short_indices.append(j)

    if short_indices:
        short_indices = sorted(set(short_indices))
        return df_sorted.iloc[short_indices]
    return df.iloc[0:0]


def detect_meter_regression(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df.iloc[0:0]
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    meter = df_sorted["meter_reading"].values
    regressions = []
    for i in range(1, len(meter)):
        if meter[i] < meter[i - 1] - 0.5:
            regressions.append(i)
    if regressions:
        return df_sorted.iloc[regressions]
    return df.iloc[0:0]


def validate_warehouse(df: pd.DataFrame) -> ValidationReport:
    warehouse_id = df["warehouse_id"].iloc[0] if not df.empty and "warehouse_id" in df.columns else "UNKNOWN"
    issues: List[ValidationIssue] = []

    missing_hours, expected_count = detect_missing_hours(df)
    if missing_hours:
        details = [f"首缺: {missing_hours[0]}"] + ([f"末缺: {missing_hours[-1]}"] if len(missing_hours) > 1 else [])
        issues.append(ValidationIssue(
            issue_type="missing_hours",
            warehouse_id=warehouse_id,
            count=len(missing_hours),
            details=details,
            severity="warning",
        ))

    dups = detect_duplicate_timestamps(df)
    if not dups.empty:
        issues.append(ValidationIssue(
            issue_type="duplicate_timestamps",
            warehouse_id=warehouse_id,
            count=len(dups),
            details=[f"重复时段: {dups['timestamp'].iloc[0]}"],
            severity="warning",
        ))

    outliers = detect_outliers(df)
    for key, out_df in outliers.items():
        severity = "error" if key in ("temp_outliers", "power_outliers") else "warning"
        issues.append(ValidationIssue(
            issue_type=key,
            warehouse_id=warehouse_id,
            count=len(out_df),
            details=[f"首条异常: {out_df['timestamp'].iloc[0]}"],
            severity=severity,
        ))

    jumps = detect_state_jumps(df)
    if not jumps.empty:
        issues.append(ValidationIssue(
            issue_type="compressor_state_jumps",
            warehouse_id=warehouse_id,
            count=len(jumps),
            details=[f"跳变点: {jumps['timestamp'].iloc[0]}"],
            severity="warning",
        ))

    short_cycles = detect_short_cycles(df)
    if not short_cycles.empty:
        issues.append(ValidationIssue(
            issue_type="compressor_short_cycles",
            warehouse_id=warehouse_id,
            count=len(short_cycles),
            details=[f"短循环占比: {len(short_cycles)/len(df)*100:.1f}%"],
            severity="error",
        ))

    meter_bad = detect_meter_regression(df)
    if not meter_bad.empty:
        issues.append(ValidationIssue(
            issue_type="meter_regression",
            warehouse_id=warehouse_id,
            count=len(meter_bad),
            details=[f"倒退点: {meter_bad['timestamp'].iloc[0]}"],
            severity="error",
        ))

    return ValidationReport(
        warehouse_id=warehouse_id,
        total_records=len(df),
        expected_records=expected_count,
        issues=issues,
    )


def validate_all(df: pd.DataFrame) -> Dict[str, ValidationReport]:
    reports = {}
    if df.empty:
        return reports
    for wh_id, grp in df.groupby("warehouse_id"):
        reports[wh_id] = validate_warehouse(grp)
    return reports
