"""数据清洗模块 - 处理缺小时段、重复时间、明显错值、状态跳变、电表倒退"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .validator import (
    DEFAULT_AMBIENT_RANGE,
    DEFAULT_INVENTORY_RANGE,
    DEFAULT_POWER_RANGE,
    detect_meter_regression,
    detect_outliers,
    detect_short_cycles,
    detect_state_jumps,
    _get_valid_temp_range,
)


@dataclass
class CleaningAction:
    action: str
    count: int
    description: str


@dataclass
class CleaningReport:
    warehouse_id: str
    original_records: int
    cleaned_records: int
    actions: List[CleaningAction] = field(default_factory=list)


def _fill_missing_hours(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"]).dt.floor("h")
    df_sorted = df_sorted.drop_duplicates(subset=["timestamp"], keep="first")

    start = df_sorted["timestamp"].min()
    end = df_sorted["timestamp"].max() + pd.Timedelta(hours=1)
    expected = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df_full = df_sorted.set_index("timestamp").reindex(expected)
    filled_count = df_full.isna().any(axis=1).sum()

    df_full["warehouse_id"] = df["warehouse_id"].iloc[0]
    if "warehouse_name" in df.columns:
        df_full["warehouse_name"] = df["warehouse_name"].iloc[0]
    if "warehouse_type" in df.columns:
        df_full["warehouse_type"] = df["warehouse_type"].iloc[0]

    numeric_cols = ["temp_celsius", "ambient_temp", "power_kwh", "inventory_kg", "meter_reading", "door_open_count"]
    for col in numeric_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].interpolate(method="time", limit_direction="both")
            df_full[col] = df_full[col].ffill().bfill()

    for col in ["compressor_status", "defrost_status"]:
        if col in df_full.columns:
            df_full[col] = df_full[col].ffill().bfill().astype(int)

    df_full["door_open_count"] = df_full["door_open_count"].fillna(0).round().astype(int)
    df_full = df_full.reset_index().rename(columns={"index": "timestamp"})
    return df_full, filled_count


def _deduplicate_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    before = len(df)
    agg_dict: Dict[str, str] = {}
    for col in df.columns:
        if col == "timestamp":
            continue
        if col in ("power_kwh", "door_open_count"):
            agg_dict[col] = "mean"
        elif col in ("compressor_status", "defrost_status"):
            agg_dict[col] = "max"
        elif col in ("temp_celsius", "ambient_temp", "inventory_kg", "meter_reading"):
            agg_dict[col] = "median"
        else:
            agg_dict[col] = "first"
    df_dedup = df.groupby("timestamp", as_index=False).agg(agg_dict)
    removed = before - len(df_dedup)
    return df_dedup, removed


def _fix_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df, {}
    warehouse_type = df["warehouse_type"].iloc[0] if "warehouse_type" in df.columns else ""
    temp_min, temp_max = _get_valid_temp_range(warehouse_type)
    fixed_counts = {}
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    fixes = {
        "temp_celsius": (temp_min, temp_max),
        "ambient_temp": DEFAULT_AMBIENT_RANGE,
        "power_kwh": DEFAULT_POWER_RANGE,
        "inventory_kg": DEFAULT_INVENTORY_RANGE,
    }

    for col, (lo, hi) in fixes.items():
        mask = (df[col] < lo) | (df[col] > hi) | df[col].isna() | np.isinf(df[col])
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, col] = np.nan
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].ffill().bfill()
            fixed_counts[col] = count

    for col in ["compressor_status", "defrost_status"]:
        mask = ~df[col].isin([0, 1]) | df[col].isna()
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, col] = np.nan
            df[col] = df[col].ffill().bfill().fillna(0).astype(int)
            fixed_counts[col] = count

    door_mask = (df["door_open_count"] < 0) | df["door_open_count"].isna()
    door_count = int(door_mask.sum())
    if door_count > 0:
        df.loc[door_mask, "door_open_count"] = 0
        df["door_open_count"] = df["door_open_count"].round().astype(int)
        fixed_counts["door_open_count"] = door_count

    return df, fixed_counts


def _fix_state_jumps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or len(df) < 3:
        return df, 0
    df = df.sort_values("timestamp").reset_index(drop=True)
    status = df["compressor_status"].values.copy()
    fixed = 0
    for i in range(1, len(status) - 1):
        if status[i] != status[i - 1] and status[i] != status[i + 1]:
            if status[i - 1] == status[i + 1]:
                status[i] = status[i - 1]
                fixed += 1
    df["compressor_status"] = status.astype(int)
    return df, fixed


def _fix_meter_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or len(df) < 2:
        return df, 0
    df = df.sort_values("timestamp").reset_index(drop=True)
    meter = df["meter_reading"].values.copy()
    power = df["power_kwh"].values.copy()
    fixed = 0

    for i in range(1, len(meter)):
        if meter[i] < meter[i - 1]:
            expected = meter[i - 1] + max(0.0, power[i])
            meter[i] = expected
            fixed += 1
    for i in range(1, len(meter)):
        if meter[i] < meter[i - 1]:
            meter[i] = meter[i - 1] + 0.1
            fixed += 1

    df["meter_reading"] = meter
    return df, fixed


def clean_warehouse(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    warehouse_id = df["warehouse_id"].iloc[0] if not df.empty and "warehouse_id" in df.columns else "UNKNOWN"
    actions: List[CleaningAction] = []
    original_count = len(df)

    df_dedup, dedup_removed = _deduplicate_timestamps(df)
    if dedup_removed > 0:
        actions.append(CleaningAction(
            action="deduplicate_timestamps",
            count=dedup_removed,
            description=f"去除重复时间戳记录 {dedup_removed} 条",
        ))

    df_filled, filled_count = _fill_missing_hours(df_dedup)
    if filled_count > 0:
        actions.append(CleaningAction(
            action="fill_missing_hours",
            count=filled_count,
            description=f"线性插值补全缺失小时段 {filled_count} 个",
        ))

    df_outliers_fixed, outlier_counts = _fix_outliers(df_filled)
    for col, cnt in outlier_counts.items():
        actions.append(CleaningAction(
            action=f"fix_outlier_{col}",
            count=cnt,
            description=f"修复 {col} 异常值 {cnt} 个",
        ))

    df_jumps_fixed, jumps_fixed = _fix_state_jumps(df_outliers_fixed)
    if jumps_fixed > 0:
        actions.append(CleaningAction(
            action="fix_state_jumps",
            count=jumps_fixed,
            description=f"修复压缩机状态跳变 {jumps_fixed} 处",
        ))

    df_meter_fixed, meter_fixed = _fix_meter_regression(df_jumps_fixed)
    if meter_fixed > 0:
        actions.append(CleaningAction(
            action="fix_meter_regression",
            count=meter_fixed,
            description=f"修复电表读数倒退 {meter_fixed} 处",
        ))

    report = CleaningReport(
        warehouse_id=warehouse_id,
        original_records=original_count,
        cleaned_records=len(df_meter_fixed),
        actions=actions,
    )
    return df_meter_fixed, report


def clean_all(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, CleaningReport]]:
    if df.empty:
        return df, {}
    cleaned_frames = []
    reports = {}
    for wh_id, grp in df.groupby("warehouse_id", sort=False):
        cleaned_grp, report = clean_warehouse(grp)
        cleaned_frames.append(cleaned_grp)
        reports[wh_id] = report
    result = pd.concat(cleaned_frames, ignore_index=True)
    return result, reports
