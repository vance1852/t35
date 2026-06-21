"""节能建议模块 - 识别问题并生成带节电量估算的建议"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyzer import REFERENCE_POWER_PER_TON_DAY, WarehouseAnalysis


@dataclass
class EnergySavingAdvice:
    warehouse_id: str
    warehouse_name: str
    category: str
    priority: str
    issue_description: str
    recommendation: str
    evidence: str
    estimated_savings_low_kwh: float = 0.0
    estimated_savings_high_kwh: float = 0.0
    confidence: float = 0.0
    affected_metric: str = ""


@dataclass
class WarehouseAdviceReport:
    warehouse_id: str
    warehouse_name: str
    advice_list: List[EnergySavingAdvice] = field(default_factory=list)
    total_estimated_savings_low_kwh: float = 0.0
    total_estimated_savings_high_kwh: float = 0.0

    def sort_advice(self):
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.advice_list.sort(
            key=lambda a: (
                priority_order.get(a.priority, 99),
                -(a.estimated_savings_high_kwh + a.estimated_savings_low_kwh) / 2,
            )
        )


DOOR_OPENS_THRESHOLDS = {
    "高温库": 12,
    "中温库": 8,
    "低温库": 5,
    "恒温库": 10,
}

DEFROSTS_PER_DAY_NORMAL = {
    "高温库": 3,
    "中温库": 3,
    "低温库": 2,
    "恒温库": 3,
}

COMPRESSOR_STARTS_THRESHOLD = {
    "高温库": 8,
    "中温库": 10,
    "低温库": 12,
    "恒温库": 8,
}

SHORT_CYCLE_RATIO_THRESHOLD = 0.08
TEMP_COMPLIANCE_THRESHOLD = 0.95


def _advice_door_open_frequency(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    threshold = DOOR_OPENS_THRESHOLDS.get(analysis.warehouse_type, 8)
    if analysis.avg_door_opens_per_day <= threshold:
        return None

    excess = analysis.avg_door_opens_per_day - threshold
    excess_ratio = excess / max(1, threshold)

    avg_power = analysis.avg_daily_power_kwh
    savings_factor = min(0.15, 0.015 + excess_ratio * 0.04)
    low = avg_power * analysis.total_days * savings_factor * 0.6
    high = avg_power * analysis.total_days * savings_factor * 1.2

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="door_management",
        priority="high" if excess_ratio > 0.5 else "medium",
        issue_description=f"日均开门次数 {analysis.avg_door_opens_per_day:.1f} 次，超过{analysis.warehouse_type}参考阈值 {threshold} 次",
        recommendation="优化作业流程，减少不必要的开门；安装门帘或快速门；考虑分区作业",
        evidence=f"本月共开门 {analysis.total_door_opens} 次，日均超出基准 {excess:.1f} 次",
        estimated_savings_low_kwh=round(low, 0),
        estimated_savings_high_kwh=round(high, 0),
        confidence=0.75,
        affected_metric="door_open_count",
    )


def _advice_defrost_cycle(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    normal = DEFROSTS_PER_DAY_NORMAL.get(analysis.warehouse_type, 3)
    actual = analysis.avg_defrosts_per_day

    if abs(actual - normal) <= 1:
        return None

    excess_ratio = (actual - normal) / max(1, normal) if actual > normal else 0
    insufficient = actual < normal - 1

    avg_power = analysis.avg_daily_power_kwh

    if excess_ratio > 0.3:
        savings_low = avg_power * analysis.total_days * 0.05
        savings_high = avg_power * analysis.total_days * 0.12
        return EnergySavingAdvice(
            warehouse_id=analysis.warehouse_id,
            warehouse_name=analysis.warehouse_name,
            category="defrost_optimization",
            priority="medium",
            issue_description=f"日均化霜 {actual:.2f} 次，超过推荐值 {normal} 次，化霜周期过短",
            recommendation="根据结霜情况适当延长化霜间隔；检查蒸发器结霜传感器是否误报；优化化霜启动条件",
            evidence=f"本月共化霜 {analysis.total_defrost_hours} 小时，频率高于同类型冷库基准",
            estimated_savings_low_kwh=round(savings_low, 0),
            estimated_savings_high_kwh=round(savings_high, 0),
            confidence=0.7,
            affected_metric="defrost_status",
        )
    elif insufficient and analysis.temp_compliance_rate < TEMP_COMPLIANCE_THRESHOLD:
        return EnergySavingAdvice(
            warehouse_id=analysis.warehouse_id,
            warehouse_name=analysis.warehouse_name,
            category="defrost_optimization",
            priority="high",
            issue_description=f"日均化霜 {actual:.2f} 次，低于推荐值 {normal} 次，且温度达标率仅 {analysis.temp_compliance_rate*100:.1f}%",
            recommendation="增加化霜频率；检查蒸发器是否严重结霜导致换热效率下降",
            evidence=f"化霜不足与温度不达标同时存在，高度怀疑蒸发器结霜问题",
            estimated_savings_low_kwh=0,
            estimated_savings_high_kwh=round(avg_power * analysis.total_days * 0.08, 0),
            confidence=0.65,
            affected_metric="defrost_status",
        )
    return None


def _advice_compressor_short_cycle(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    if analysis.short_cycle_ratio < SHORT_CYCLE_RATIO_THRESHOLD:
        return None

    short_ratio = analysis.short_cycle_ratio
    avg_power = analysis.avg_daily_power_kwh

    extra_loss_factor = min(0.25, short_ratio * 1.2)
    savings_low = avg_power * analysis.total_days * extra_loss_factor * 0.5
    savings_high = avg_power * analysis.total_days * extra_loss_factor * 1.0

    priority = "critical" if short_ratio > 0.2 else "high"

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="compressor_health",
        priority=priority,
        issue_description=f"压缩机短循环严重，短循环时段占比 {short_ratio*100:.1f}%，启停 {analysis.compressor_total_starts} 次",
        recommendation="检查温控器设定和滞环区间；检查制冷剂充注量；排查膨胀阀故障；考虑加装变频驱动",
        evidence=f"短循环占比 {short_ratio*100:.1f}% 远超正常水平（<8%），压缩机平均每小时启停约 {analysis.compressor_avg_starts_per_day/24:.2f} 次",
        estimated_savings_low_kwh=round(savings_low, 0),
        estimated_savings_high_kwh=round(savings_high, 0),
        confidence=0.85,
        affected_metric="compressor_status",
    )


def _advice_night_temperature(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    if df.empty:
        return None

    df_night = df[(df["timestamp"].dt.hour >= 22) | (df["timestamp"].dt.hour < 6)].copy()
    df_day = df[(df["timestamp"].dt.hour >= 8) & (df["timestamp"].dt.hour < 18)].copy()

    if len(df_night) < 10 or len(df_day) < 10:
        return None

    night_temp = float(df_night["temp_celsius"].mean())
    day_temp = float(df_day["temp_celsius"].mean())
    target = analysis.target_temp
    tolerance = analysis.temp_tolerance

    day_margin = abs(day_temp - target)
    night_margin = abs(night_temp - target)

    if day_margin > tolerance * 0.8 or night_margin > tolerance * 1.5:
        return None

    lower_night = night_temp < target - tolerance * 0.3
    diff_too_big = day_temp - night_temp > tolerance * 0.8

    if not (lower_night and diff_too_big):
        return None

    potential_raise = min(abs(night_temp - (target - tolerance * 0.2)), 2.5)
    if potential_raise < 0.5:
        return None

    ambient_delta = float(df["ambient_temp"].mean()) - target
    heat_gain_factor = max(0.3, ambient_delta / 25.0)

    night_ratio = len(df_night) / len(df)
    avg_power = analysis.avg_daily_power_kwh
    savings_per_degree = avg_power * night_ratio * min(0.08, 0.02 + heat_gain_factor * 0.03)

    savings_low = savings_per_degree * potential_raise * analysis.total_days * 0.5
    savings_high = savings_per_degree * potential_raise * analysis.total_days * 1.0

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="temperature_setpoint",
        priority="medium",
        issue_description=f"夜间温度 {night_temp:.2f}°C 较日间 {day_temp:.2f}°C 偏低 {day_temp-night_temp:.2f}°C，有提升空间约 {potential_raise:.1f}°C",
        recommendation=f"适当调高夜间温度设定 {potential_raise:.1f}°C，保持在目标温度 {target}°C 下限附近即可，避免过冷运行",
        evidence=f"日夜间温差达 {day_temp-night_temp:.2f}°C，夜间温度低于目标 {target-night_temp:.2f}°C，存在过冷浪费",
        estimated_savings_low_kwh=round(savings_low, 0),
        estimated_savings_high_kwh=round(savings_high, 0),
        confidence=0.7,
        affected_metric="temp_celsius",
    )


def _advice_start_frequency(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    threshold = COMPRESSOR_STARTS_THRESHOLD.get(analysis.warehouse_type, 8)
    if analysis.compressor_avg_starts_per_day <= threshold * 1.5:
        return None

    excess = analysis.compressor_avg_starts_per_day - threshold
    excess_ratio = excess / threshold
    if excess_ratio < 0.5:
        return None

    avg_power = analysis.avg_daily_power_kwh
    savings_factor = min(0.1, 0.01 + excess_ratio * 0.025)
    low = avg_power * analysis.total_days * savings_factor * 0.5
    high = avg_power * analysis.total_days * savings_factor * 0.9

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="compressor_health",
        priority="medium" if excess_ratio < 1.0 else "high",
        issue_description=f"压缩机日均启停 {analysis.compressor_avg_starts_per_day:.1f} 次，参考阈值 {threshold} 次",
        recommendation="扩宽温度控制滞环；检查冷媒系统平衡；评估蓄冷改造",
        evidence=f"月内累计启停 {analysis.compressor_total_starts} 次，超基准 {excess_ratio*100:.0f}%",
        estimated_savings_low_kwh=round(low, 0),
        estimated_savings_high_kwh=round(high, 0),
        confidence=0.6,
        affected_metric="compressor_status",
    )


def _advice_benchmark_gap(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    ref = REFERENCE_POWER_PER_TON_DAY.get(analysis.warehouse_type)
    if not ref or analysis.adjusted_power_per_ton_day <= 0:
        return None

    gap = analysis.adjusted_power_per_ton_day - ref
    if gap <= ref * 0.1:
        return None

    gap_ratio = gap / ref
    total_power = analysis.total_power_kwh
    savings_potential = total_power * min(0.25, gap_ratio * 0.5)

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="overall_efficiency",
        priority="medium" if gap_ratio < 0.3 else "high",
        issue_description=f"校正后吨日耗电 {analysis.adjusted_power_per_ton_day:.2f} kWh/吨·日，行业基准 {ref:.1f} kWh/吨·日，差距 {gap_ratio*100:.1f}%",
        recommendation="综合评估设备健康、围护结构和管理流程，制定系统节能改造方案",
        evidence=f"校正能耗高于同类型冷库 {gap_ratio*100:.0f}%，对标等级为「{analysis.benchmark_tier}」",
        estimated_savings_low_kwh=round(savings_potential * 0.3, 0),
        estimated_savings_high_kwh=round(savings_potential * 0.8, 0),
        confidence=0.55,
        affected_metric="adjusted_power_per_ton_day",
    )


def _advice_temp_compliance(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> Optional[EnergySavingAdvice]:
    if analysis.temp_compliance_rate >= TEMP_COMPLIANCE_THRESHOLD:
        return None

    return EnergySavingAdvice(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
        category="temperature_quality",
        priority="critical",
        issue_description=f"温度达标率 {analysis.temp_compliance_rate*100:.1f}% 低于 95% 要求，累计温超标 {analysis.temp_overshoot_degree_hours:.1f} °C·h",
        recommendation="优先排查温度不达标的根因：设备故障、负荷过大、化霜异常或围护结构漏冷",
        evidence=f"温超标度数 {analysis.temp_overshoot_degree_hours:.1f} °C·h，平均温度 {analysis.avg_temp:.2f}°C，目标 {analysis.target_temp}°C",
        estimated_savings_low_kwh=0,
        estimated_savings_high_kwh=0,
        confidence=0.9,
        affected_metric="temp_compliance_rate",
    )


def generate_warehouse_advice(
    analysis: WarehouseAnalysis, df: pd.DataFrame
) -> WarehouseAdviceReport:
    generators = [
        _advice_temp_compliance,
        _advice_compressor_short_cycle,
        _advice_door_open_frequency,
        _advice_start_frequency,
        _advice_defrost_cycle,
        _advice_night_temperature,
        _advice_benchmark_gap,
    ]

    report = WarehouseAdviceReport(
        warehouse_id=analysis.warehouse_id,
        warehouse_name=analysis.warehouse_name,
    )

    for gen in generators:
        advice = gen(analysis, df)
        if advice is not None:
            report.advice_list.append(advice)

    report.sort_advice()

    report.total_estimated_savings_low_kwh = sum(a.estimated_savings_low_kwh for a in report.advice_list)
    report.total_estimated_savings_high_kwh = sum(a.estimated_savings_high_kwh for a in report.advice_list)

    return report


def generate_all_advice(
    analyses: Dict[str, WarehouseAnalysis], df: pd.DataFrame
) -> Dict[str, WarehouseAdviceReport]:
    reports = {}
    if df.empty:
        return reports
    for wh_id, grp in df.groupby("warehouse_id"):
        if wh_id in analyses:
            reports[wh_id] = generate_warehouse_advice(analyses[wh_id], grp)
    return reports
