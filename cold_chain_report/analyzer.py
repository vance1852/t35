"""分析模块 - 日均能耗、温度达标率、启停次数、环境温度和库存量校正、对标"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .validator import WAREHOUSE_TYPE_TEMP_RANGES, detect_short_cycles


@dataclass
class DailyMetrics:
    date: str
    total_power_kwh: float
    avg_temp_celsius: float
    temp_compliance_rate: float
    compressor_starts: int
    compressor_run_hours: float
    avg_ambient_temp: float
    avg_inventory_kg: float
    peak_power_kw: float
    door_open_total: int
    defrost_count: int


@dataclass
class WarehouseAnalysis:
    warehouse_id: str
    warehouse_name: str
    warehouse_type: str
    target_temp: float
    temp_tolerance: float

    total_power_kwh: float = 0.0
    avg_daily_power_kwh: float = 0.0
    total_days: int = 0

    temp_compliance_rate: float = 0.0
    temp_overshoot_degree_hours: float = 0.0
    avg_temp: float = 0.0
    min_temp: float = 0.0
    max_temp: float = 0.0

    compressor_total_starts: int = 0
    compressor_avg_starts_per_day: float = 0.0
    compressor_run_ratio: float = 0.0
    short_cycle_count: int = 0
    short_cycle_ratio: float = 0.0

    avg_ambient_temp: float = 0.0
    avg_inventory_kg: float = 0.0

    raw_power_per_ton_day: float = 0.0
    adjusted_power_per_ton_day: float = 0.0
    ambient_adjustment_factor: float = 0.0
    inventory_adjustment_factor: float = 0.0

    total_door_opens: int = 0
    avg_door_opens_per_day: float = 0.0
    avg_defrosts_per_day: float = 0.0
    total_defrost_hours: int = 0

    daily_metrics: List[DailyMetrics] = field(default_factory=list)
    benchmark_ranking: Optional[int] = None
    benchmark_tier: str = "normal"


WAREHOUSE_TARGET_CONFIG = {
    "高温库": {"target_temp": 4.0, "tolerance": 1.5},
    "中温库": {"target_temp": -18.0, "tolerance": 2.0},
    "低温库": {"target_temp": -30.0, "tolerance": 2.5},
    "恒温库": {"target_temp": 0.0, "tolerance": 1.0},
}

BASELINE_AMBIENT_BY_TYPE = {
    "高温库": 25.0,
    "中温库": 25.0,
    "低温库": 25.0,
    "恒温库": 25.0,
}

REFERENCE_POWER_PER_TON_DAY = {
    "高温库": 6.0,
    "中温库": 12.0,
    "低温库": 22.0,
    "恒温库": 8.0,
}


def _get_target_config(warehouse_type: str) -> Tuple[float, float]:
    cfg = WAREHOUSE_TARGET_CONFIG.get(warehouse_type, {"target_temp": 0.0, "tolerance": 2.0})
    return cfg["target_temp"], cfg["tolerance"]


def _calc_compliance(temp_series: pd.Series, target: float, tolerance: float) -> Tuple[float, float]:
    upper = target + tolerance
    lower = target - tolerance
    within = (temp_series >= lower) & (temp_series <= upper)
    rate = float(within.mean()) if len(temp_series) > 0 else 0.0

    overshoot_high = np.maximum(0.0, temp_series - upper)
    overshoot_low = np.maximum(0.0, lower - temp_series)
    degree_hours = float((overshoot_high + overshoot_low).sum())
    return rate, degree_hours


def _calc_compressor_starts(status_series: pd.Series) -> int:
    if len(status_series) < 2:
        return 0
    arr = status_series.values.astype(int)
    starts = int(((arr[1:] == 1) & (arr[:-1] == 0)).sum())
    return starts


def _calc_daily_metrics(df: pd.DataFrame, target: float, tolerance: float) -> List[DailyMetrics]:
    result = []
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    for date_val, grp in df.groupby("date"):
        grp = grp.sort_values("timestamp")
        compliance, _ = _calc_compliance(grp["temp_celsius"], target, tolerance)
        starts = _calc_compressor_starts(grp["compressor_status"])
        result.append(DailyMetrics(
            date=str(date_val),
            total_power_kwh=float(grp["power_kwh"].sum()),
            avg_temp_celsius=float(grp["temp_celsius"].mean()),
            temp_compliance_rate=compliance,
            compressor_starts=starts,
            compressor_run_hours=float(grp["compressor_status"].sum()),
            avg_ambient_temp=float(grp["ambient_temp"].mean()),
            avg_inventory_kg=float(grp["inventory_kg"].mean()),
            peak_power_kw=float(grp["power_kwh"].max()),
            door_open_total=int(grp["door_open_count"].sum()),
            defrost_count=int(grp["defrost_status"].sum()),
        ))
    return result


def _adjust_for_conditions(
    raw_per_ton_day: float,
    actual_ambient: float,
    actual_inventory_ratio: float,
    baseline_ambient: float,
    warehouse_type: str,
) -> Tuple[float, float, float]:
    ambient_delta = actual_ambient - baseline_ambient
    ambient_factor = 1.0 + ambient_delta * 0.015

    inventory_factor = 1.0 / max(0.5, actual_inventory_ratio)

    adjusted = raw_per_ton_day / ambient_factor / inventory_factor

    if warehouse_type in ("低温库", "中温库"):
        adjusted = raw_per_ton_day / (ambient_factor ** 1.2) / inventory_factor

    return adjusted, ambient_factor, inventory_factor


def analyze_warehouse(df: pd.DataFrame, raw_df_for_short_cycle: pd.DataFrame = None) -> WarehouseAnalysis:
    if df.empty:
        return WarehouseAnalysis(warehouse_id="UNKNOWN", warehouse_name="", warehouse_type="")

    warehouse_id = df["warehouse_id"].iloc[0]
    warehouse_name = df["warehouse_name"].iloc[0] if "warehouse_name" in df.columns else warehouse_id
    warehouse_type = df["warehouse_type"].iloc[0] if "warehouse_type" in df.columns else "恒温库"
    target_temp, temp_tolerance = _get_target_config(warehouse_type)

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    short_cycle_source = raw_df_for_short_cycle
    if short_cycle_source is None or short_cycle_source.empty:
        short_cycle_source = df_sorted
    else:
        short_cycle_source = short_cycle_source.sort_values("timestamp").reset_index(drop=True)

    total_power = float(df_sorted["power_kwh"].sum())
    total_days = df_sorted["timestamp"].dt.normalize().nunique()
    avg_daily = total_power / total_days if total_days > 0 else 0.0

    compliance, overshoot = _calc_compliance(df_sorted["temp_celsius"], target_temp, temp_tolerance)

    total_starts = _calc_compressor_starts(df_sorted["compressor_status"])
    avg_starts = total_starts / total_days if total_days > 0 else 0.0
    run_ratio = float(df_sorted["compressor_status"].mean()) if len(df_sorted) > 0 else 0.0

    short_df = detect_short_cycles(short_cycle_source)
    short_count = len(short_df)
    short_ratio = short_count / len(short_cycle_source) if len(short_cycle_source) > 0 else 0.0

    avg_ambient = float(df_sorted["ambient_temp"].mean())
    avg_inventory = float(df_sorted["inventory_kg"].mean())

    inv_range = WAREHOUSE_TYPE_TEMP_RANGES
    _ = inv_range
    max_inv_capacity = avg_inventory / 0.7 if avg_inventory > 0 else 1.0
    inventory_ratio = min(1.0, avg_inventory / max_inv_capacity)

    tons = avg_inventory / 1000.0 if avg_inventory > 0 else 0.1
    power_per_ton_day = (total_power / total_days) / tons if (total_days > 0 and tons > 0) else 0.0

    baseline_amb = BASELINE_AMBIENT_BY_TYPE.get(warehouse_type, 25.0)
    adjusted_power, amb_factor, inv_factor = _adjust_for_conditions(
        power_per_ton_day, avg_ambient, inventory_ratio, baseline_amb, warehouse_type
    )

    total_door = int(df_sorted["door_open_count"].sum())
    avg_door = total_door / total_days if total_days > 0 else 0.0

    total_defrost_hrs = int(df_sorted["defrost_status"].sum())
    avg_defrost_per_day = total_defrost_hrs / total_days if total_days > 0 else 0.0

    daily_metrics = _calc_daily_metrics(df_sorted, target_temp, temp_tolerance)

    return WarehouseAnalysis(
        warehouse_id=warehouse_id,
        warehouse_name=warehouse_name,
        warehouse_type=warehouse_type,
        target_temp=target_temp,
        temp_tolerance=temp_tolerance,
        total_power_kwh=round(total_power, 2),
        avg_daily_power_kwh=round(avg_daily, 2),
        total_days=total_days,
        temp_compliance_rate=round(compliance, 4),
        temp_overshoot_degree_hours=round(overshoot, 2),
        avg_temp=round(float(df_sorted["temp_celsius"].mean()), 2),
        min_temp=round(float(df_sorted["temp_celsius"].min()), 2),
        max_temp=round(float(df_sorted["temp_celsius"].max()), 2),
        compressor_total_starts=total_starts,
        compressor_avg_starts_per_day=round(avg_starts, 2),
        compressor_run_ratio=round(run_ratio, 4),
        short_cycle_count=short_count,
        short_cycle_ratio=round(short_ratio, 4),
        avg_ambient_temp=round(avg_ambient, 2),
        avg_inventory_kg=round(avg_inventory, 0),
        raw_power_per_ton_day=round(power_per_ton_day, 3),
        adjusted_power_per_ton_day=round(adjusted_power, 3),
        ambient_adjustment_factor=round(amb_factor, 3),
        inventory_adjustment_factor=round(inv_factor, 3),
        total_door_opens=total_door,
        avg_door_opens_per_day=round(avg_door, 1),
        avg_defrosts_per_day=round(avg_defrost_per_day, 2),
        total_defrost_hours=total_defrost_hrs,
        daily_metrics=daily_metrics,
    )


def analyze_all(df: pd.DataFrame, raw_df: pd.DataFrame = None) -> Dict[str, WarehouseAnalysis]:
    analyses = {}
    if df.empty:
        return analyses
    raw_groups = {}
    if raw_df is not None and not raw_df.empty:
        for wh_id, grp in raw_df.groupby("warehouse_id"):
            raw_groups[wh_id] = grp
    for wh_id, grp in df.groupby("warehouse_id"):
        raw_grp = raw_groups.get(wh_id)
        analyses[wh_id] = analyze_warehouse(grp, raw_grp)
    return _assign_benchmark_tiers(analyses)


def _assign_benchmark_tiers(analyses: Dict[str, WarehouseAnalysis]) -> Dict[str, WarehouseAnalysis]:
    type_groups: Dict[str, List[WarehouseAnalysis]] = {}
    for a in analyses.values():
        type_groups.setdefault(a.warehouse_type, []).append(a)

    for wh_type, items in type_groups.items():
        items_sorted = sorted(items, key=lambda x: x.adjusted_power_per_ton_day if x.adjusted_power_per_ton_day > 0 else 9999)
        for rank, item in enumerate(items_sorted, 1):
            item.benchmark_ranking = rank
            n = len(items_sorted)
            if rank <= max(1, n // 3):
                item.benchmark_tier = "优秀"
            elif rank <= max(1, 2 * n // 3):
                item.benchmark_tier = "良好"
            else:
                item.benchmark_tier = "待改进"
    return analyses
