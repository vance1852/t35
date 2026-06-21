"""报告模块 - 导出 Markdown 报告、清洗后 CSV 和建议 JSON"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .adviser import EnergySavingAdvice, WarehouseAdviceReport
from .analyzer import WarehouseAnalysis, DailyMetrics
from .cleaner import CleaningReport
from .validator import ValidationReport


def _severity_to_text(severity: str) -> str:
    return {"critical": "严重", "high": "高", "medium": "中", "low": "低"}.get(severity, severity)


def _priority_to_text(priority: str) -> str:
    return {"critical": "⚠️ 紧急", "high": "🔴 高优", "medium": "🟡 中优", "low": "🟢 低优"}.get(priority, priority)


def _category_to_text(category: str) -> str:
    mapping = {
        "door_management": "🚪 开门管理",
        "defrost_optimization": "❄️ 化霜优化",
        "compressor_health": "⚙️ 压缩机健康",
        "temperature_setpoint": "🌡️ 温度设定",
        "overall_efficiency": "📊 综合能效",
        "temperature_quality": "🔬 温度品质",
    }
    return mapping.get(category, category)


def _write_warehouse_section_markdown(
    analysis: WarehouseAnalysis,
    advice_report: WarehouseAdviceReport,
    validation: ValidationReport = None,
    cleaning: CleaningReport = None,
) -> str:
    lines: List[str] = []

    rank_text = f"（同类型第 {analysis.benchmark_ranking} 名）" if analysis.benchmark_ranking else ""
    lines.append(f"## {analysis.warehouse_name} [{analysis.warehouse_id}]")
    lines.append("")
    lines.append(f"- **库型**: {analysis.warehouse_type}")
    lines.append(f"- **目标温度**: {analysis.target_temp}°C（±{analysis.temp_tolerance}°C）")
    lines.append(f"- **对标等级**: **{analysis.benchmark_tier}** {rank_text}")
    lines.append(f"- **统计周期**: {analysis.total_days} 天")
    lines.append("")

    lines.append("### 📈 核心能耗指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 备注 |")
    lines.append("|---|---|---|")
    lines.append(f"| 总耗电量 | **{analysis.total_power_kwh:,.0f} kWh** | |")
    lines.append(f"| 日均耗电量 | {analysis.avg_daily_power_kwh:,.0f} kWh/天 | |")
    lines.append(f"| 原始吨日耗电 | {analysis.raw_power_per_ton_day:.2f} kWh/吨·天 | 未校正 |")
    lines.append(f"| **校正后吨日耗电** | **{analysis.adjusted_power_per_ton_day:.2f} kWh/吨·天** | 按温度/库存校正 |")
    lines.append(f"| 环境温度校正因子 | ×{analysis.ambient_adjustment_factor:.3f} | 均值{analysis.avg_ambient_temp:.1f}°C |")
    lines.append(f"| 库存因子校正 | ×{analysis.inventory_adjustment_factor:.3f} | 平均库存{analysis.avg_inventory_kg:,.0f}kg |")
    lines.append("")

    lines.append("### 🌡️ 温度品质")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 温度达标率 | **{analysis.temp_compliance_rate*100:.1f}%** |")
    lines.append(f"| 平均库温 | {analysis.avg_temp:.2f}°C |")
    lines.append(f"| 最低/最高 | {analysis.min_temp:.2f}°C / {analysis.max_temp:.2f}°C |")
    lines.append(f"| 累计温超标 | {analysis.temp_overshoot_degree_hours:.1f} °C·h |")
    lines.append("")

    lines.append("### ⚙️ 压缩机运行")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 总启停机次数 | {analysis.compressor_total_starts:,} 次 |")
    lines.append(f"| 日均启停 | {analysis.compressor_avg_starts_per_day:.1f} 次/天 |")
    lines.append(f"| 运行时长占比 | {analysis.compressor_run_ratio*100:.1f}% |")
    lines.append(f"| **短循环占比** | **{analysis.short_cycle_ratio*100:.1f}%** ({analysis.short_cycle_count} 小时) |")
    lines.append("")

    lines.append("### 🚪 作业与化霜")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 月总开门次数 | {analysis.total_door_opens:,} 次 |")
    lines.append(f"| 日均开门 | {analysis.avg_door_opens_per_day:.1f} 次/天 |")
    lines.append(f"| 日均化霜次数 | {analysis.avg_defrosts_per_day:.2f} 次/天 |")
    lines.append(f"| 月化霜时长 | {analysis.total_defrost_hours} 小时 |")
    lines.append("")

    if validation and validation.issues:
        lines.append("### 🔍 数据质量校验")
        lines.append("")
        lines.append(f"原始记录 {validation.total_records} 条，预期 {validation.expected_records} 条")
        lines.append("")
        lines.append("| 问题类型 | 数量 | 严重度 | 首条详情 |")
        lines.append("|---|---|---|---|")
        for issue in validation.issues:
            detail = issue.details[0] if issue.details else "-"
            sev_text = _severity_to_text(issue.severity)
            lines.append(f"| {issue.issue_type} | {issue.count} | {sev_text} | {detail} |")
        lines.append("")

    if cleaning and cleaning.actions:
        lines.append("### 🧹 数据清洗动作")
        lines.append("")
        lines.append(f"清洗前 {cleaning.original_records} 条 → 清洗后 {cleaning.cleaned_records} 条")
        lines.append("")
        lines.append("| 动作 | 数量 | 说明 |")
        lines.append("|---|---|---|")
        for act in cleaning.actions:
            lines.append(f"| {act.action} | {act.count} | {act.description} |")
        lines.append("")

    if advice_report.advice_list:
        lines.append("### 💡 节能改进建议")
        lines.append("")
        lines.append(f"**预估月度节电量范围**: {advice_report.total_estimated_savings_low_kwh:,.0f} ~ {advice_report.total_estimated_savings_high_kwh:,.0f} kWh")
        lines.append("")
        for idx, adv in enumerate(advice_report.advice_list, 1):
            priority = _priority_to_text(adv.priority)
            category = _category_to_text(adv.category)
            lines.append(f"#### {idx}. {priority} - {category}")
            lines.append("")
            lines.append(f"**问题**: {adv.issue_description}")
            lines.append("")
            lines.append(f"**建议**: {adv.recommendation}")
            lines.append("")
            lines.append(f"**依据**: {adv.evidence}")
            lines.append("")
            if adv.estimated_savings_high_kwh > 0 or adv.estimated_savings_low_kwh > 0:
                lines.append(f"**节电量预估**: {adv.estimated_savings_low_kwh:,.0f} ~ {adv.estimated_savings_high_kwh:,.0f} kWh/月（置信度 {adv.confidence*100:.0f}%）")
                lines.append("")
    else:
        lines.append("### ✅ 节能评估")
        lines.append("")
        lines.append("运行状况良好，暂未发现显著节能机会点。")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_report(
    analyses: Dict[str, WarehouseAnalysis],
    advice_reports: Dict[str, WarehouseAdviceReport],
    validations: Dict[str, ValidationReport] = None,
    cleanings: Dict[str, CleaningReport] = None,
    report_period: str = "",
) -> str:
    validations = validations or {}
    cleanings = cleanings or {}

    total_power = sum(a.total_power_kwh for a in analyses.values())
    total_savings_low = sum(r.total_estimated_savings_low_kwh for r in advice_reports.values())
    total_savings_high = sum(r.total_estimated_savings_high_kwh for r in advice_reports.values())
    n_warehouses = len(analyses)

    lines: List[str] = []
    lines.append("# ❄️ 冷库月度运行分析报告")
    lines.append("")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **报告周期**: {report_period or '本月'}")
    lines.append(f"- **冷库数量**: {n_warehouses} 座")
    lines.append("")

    lines.append("## 📊 总体概览")
    lines.append("")
    lines.append(f"- **全库总耗电量**: **{total_power:,.0f} kWh**")
    lines.append(f"- **全库预估节电量范围**: **{total_savings_low:,.0f} ~ {total_savings_high:,.0f} kWh**")
    lines.append(f"- **建议总数**: {sum(len(r.advice_list) for r in advice_reports.values())} 条")
    lines.append("")

    lines.append("| 冷库名称 | 库型 | 对标等级 | 日均耗电(kWh) | 校正吨日耗电 | 温度达标率 | 短循环占比 | 建议数 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for a in sorted(analyses.values(), key=lambda x: (
        0 if x.benchmark_tier == "优秀" else 1 if x.benchmark_tier == "良好" else 2,
        x.adjusted_power_per_ton_day,
    )):
        lines.append(
            f"| {a.warehouse_name} [{a.warehouse_id}] | {a.warehouse_type} | {a.benchmark_tier} | "
            f"{a.avg_daily_power_kwh:,.0f} | {a.adjusted_power_per_ton_day:.2f} | "
            f"{a.temp_compliance_rate*100:.1f}% | {a.short_cycle_ratio*100:.1f}% | "
            f"{len(advice_reports.get(a.warehouse_id, WarehouseAdviceReport('', '')).advice_list)} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")

    sorted_ids = sorted(
        analyses.keys(),
        key=lambda wh: (
            0 if analyses[wh].benchmark_tier == "待改进" else 1 if analyses[wh].benchmark_tier == "良好" else 2,
            -analyses[wh].adjusted_power_per_ton_day,
        ),
    )
    for wh_id in sorted_ids:
        lines.append(_write_warehouse_section_markdown(
            analyses[wh_id],
            advice_reports.get(wh_id, WarehouseAdviceReport(wh_id, analyses[wh_id].warehouse_name)),
            validations.get(wh_id),
            cleanings.get(wh_id),
        ))
        lines.append("---")
        lines.append("")

    lines.append("## 📚 附录：校正方法说明")
    lines.append("")
    lines.append("### 环境温度校正")
    lines.append("- 以 25°C 为基准环境温度，每高出 1°C 能耗基准上调约 1.5%")
    lines.append("- 低温/中温库的温度敏感度系数提高为 1.8%/°C")
    lines.append("")
    lines.append("### 库存量校正")
    lines.append("- 以 70% 满载率为基准，库存率越低则单位吨日耗电越高")
    lines.append("- 校正后可避免将「未满载」误判为「设备差」")
    lines.append("")
    lines.append("### 对标分级")
    lines.append("- 同库型内部按校正后吨日耗电排序，前三分之一为「优秀」，后三分之一为「待改进」")
    lines.append("")

    return "\n".join(lines)


def _advice_to_dict(adv: EnergySavingAdvice) -> dict:
    return {
        "category": adv.category,
        "priority": adv.priority,
        "issue_description": adv.issue_description,
        "recommendation": adv.recommendation,
        "evidence": adv.evidence,
        "estimated_savings_low_kwh": adv.estimated_savings_low_kwh,
        "estimated_savings_high_kwh": adv.estimated_savings_high_kwh,
        "confidence": adv.confidence,
        "affected_metric": adv.affected_metric,
    }


def _daily_to_list(daily: List[DailyMetrics]) -> List[dict]:
    return [asdict(d) for d in daily]


def generate_advice_json(
    analyses: Dict[str, WarehouseAnalysis],
    advice_reports: Dict[str, WarehouseAdviceReport],
    report_period: str = "",
) -> str:
    root = {
        "generated_at": datetime.now().isoformat(),
        "report_period": report_period,
        "warehouses": {},
    }
    for wh_id, analysis in analyses.items():
        advice = advice_reports.get(wh_id, WarehouseAdviceReport(wh_id, analysis.warehouse_name))
        root["warehouses"][wh_id] = {
            "warehouse_name": analysis.warehouse_name,
            "warehouse_type": analysis.warehouse_type,
            "benchmark_tier": analysis.benchmark_tier,
            "benchmark_ranking": analysis.benchmark_ranking,
            "total_power_kwh": analysis.total_power_kwh,
            "avg_daily_power_kwh": analysis.avg_daily_power_kwh,
            "adjusted_power_per_ton_day": analysis.adjusted_power_per_ton_day,
            "temp_compliance_rate": analysis.temp_compliance_rate,
            "short_cycle_ratio": analysis.short_cycle_ratio,
            "daily_metrics": _daily_to_list(analysis.daily_metrics),
            "advice_count": len(advice.advice_list),
            "total_estimated_savings_low_kwh": advice.total_estimated_savings_low_kwh,
            "total_estimated_savings_high_kwh": advice.total_estimated_savings_high_kwh,
            "advice_list": [_advice_to_dict(a) for a in advice.advice_list],
        }
    return json.dumps(root, ensure_ascii=False, indent=2)


def export_cleaned_csv(cleaned_df: pd.DataFrame, output_path: str) -> str:
    cleaned_df.to_csv(output_path, index=False)
    return output_path


def export_all(
    output_dir: str,
    cleaned_df: pd.DataFrame,
    analyses: Dict[str, WarehouseAnalysis],
    advice_reports: Dict[str, WarehouseAdviceReport],
    validations: Dict[str, ValidationReport] = None,
    cleanings: Dict[str, CleaningReport] = None,
    report_period: str = "",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}

    md_text = generate_markdown_report(analyses, advice_reports, validations, cleanings, report_period)
    md_path = os.path.join(output_dir, "monthly_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    outputs["markdown"] = md_path

    json_text = generate_advice_json(analyses, advice_reports, report_period)
    json_path = os.path.join(output_dir, "saving_advice.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_text)
    outputs["advice_json"] = json_path

    csv_path = os.path.join(output_dir, "cleaned_data.csv")
    export_cleaned_csv(cleaned_df, csv_path)
    outputs["cleaned_csv"] = csv_path

    return outputs
