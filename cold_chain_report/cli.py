"""命令行入口 - click CLI"""

import os
import sys

import click
import pandas as pd

from .sample_data import generate_sample_data, load_sample_data
from .validator import validate_all, ValidationReport
from .cleaner import clean_all, CleaningReport
from .analyzer import analyze_all, WarehouseAnalysis
from .adviser import generate_all_advice, WarehouseAdviceReport
from .reporter import export_all, generate_markdown_report, generate_advice_json


@click.group(help="冷库月报离线分析工具")
@click.version_option("1.0.0")
def cli():
    pass


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise click.FileError(path, "文件不存在")
    return pd.read_csv(path, parse_dates=["timestamp"])


def _print_validation_table(reports: dict):
    click.echo("\n======== 数据校验结果 ========")
    for wh_id, rep in sorted(reports.items()):
        click.echo(f"\n仓库 {wh_id}: {rep.total_records} 条 / 预期 {rep.expected_records} 条")
        if rep.issues:
            for issue in rep.issues:
                click.echo(f"  [{issue.severity.upper():7s}] {issue.issue_type:30s} × {issue.count}")
        else:
            click.echo("  [OK] 无异常")


def _print_cleaning_table(reports: dict):
    click.echo("\n======== 数据清洗动作 ========")
    for wh_id, rep in sorted(reports.items()):
        click.echo(f"\n仓库 {wh_id}: {rep.original_records} → {rep.cleaned_records} 条")
        for act in rep.actions:
            click.echo(f"  · {act.action}: {act.description}")
        if not rep.actions:
            click.echo("  [OK] 无需清洗")


def _print_analysis_table(analyses: dict):
    click.echo("\n======== 分析与对标 ========")
    click.echo(
        f"{'仓库':<25s} {'库型':<6s} {'对标':<4s} {'日均(kWh)':>10s} "
        f"{'校正吨日':>9s} {'达标率':>8s} {'短循%':>7s} {'启停/天':>7s}"
    )
    items = sorted(
        analyses.values(),
        key=lambda a: (
            0 if a.benchmark_tier == "优秀" else 1 if a.benchmark_tier == "良好" else 2,
            a.adjusted_power_per_ton_day,
        ),
    )
    for a in items:
        click.echo(
            f"{a.warehouse_name[:23]:<25s} {a.warehouse_type:<6s} {a.benchmark_tier:<4s} "
            f"{a.avg_daily_power_kwh:>10,.0f} {a.adjusted_power_per_ton_day:>9.2f} "
            f"{a.temp_compliance_rate*100:>7.1f}% {a.short_cycle_ratio*100:>6.1f}% "
            f"{a.compressor_avg_starts_per_day:>7.1f}"
        )


def _print_advice_table(advice_reports: dict):
    click.echo("\n======== 节能建议 ========")
    total_low = 0.0
    total_high = 0.0
    for wh_id, rep in sorted(advice_reports.items()):
        click.echo(f"\n仓库 {wh_id}: {len(rep.advice_list)} 条建议，月节电 {rep.total_estimated_savings_low_kwh:,.0f}~{rep.total_estimated_savings_high_kwh:,.0f} kWh")
        total_low += rep.total_estimated_savings_low_kwh
        total_high += rep.total_estimated_savings_high_kwh
        for idx, adv in enumerate(rep.advice_list, 1):
            click.echo(f"  {idx}. [{adv.priority.upper():6s}] {adv.issue_description[:50]}")
            if adv.estimated_savings_high_kwh > 0:
                click.echo(f"      → 节电 {adv.estimated_savings_low_kwh:,.0f}~{adv.estimated_savings_high_kwh:,.0f} kWh: {adv.recommendation[:50]}")
    click.echo(f"\n全库合计节电潜力: {total_low:,.0f} ~ {total_high:,.0f} kWh/月")


@cli.command("generate", help="生成多座冷库一整月的小时级样例CSV")
@click.option("-o", "--output-dir", default="./sample_data", show_default=True, help="输出目录")
@click.option("--start-date", default="2026-05-01", show_default=True, help="起始日期 YYYY-MM-DD")
@click.option("--days", default=30, show_default=True, type=int, help="天数")
def cmd_generate(output_dir: str, start_date: str, days: int):
    merged = generate_sample_data(output_dir, start_date=start_date, days=days)
    click.echo(f"\n[OK] 样例数据已生成，合并文件: {merged}")


@cli.command("validate", help="校验原始数据质量，识别缺口/重复/错值/跳变/电表倒退")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
def cmd_validate(input_csv: str):
    df = _read_csv(input_csv)
    reports = validate_all(df)
    _print_validation_table(reports)
    total_issues = sum(len(r.issues) for r in reports.values())
    if total_issues == 0:
        click.echo("\n[OK] 全部仓库数据质量良好，无需清洗。")
    else:
        click.echo(f"\n[WARN] 共发现 {total_issues} 类问题，建议运行 `clean` 命令。")


@cli.command("clean", help="执行数据清洗并输出清洗后CSV")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-csv", default=None, help="清洗后CSV输出路径")
def cmd_clean(input_csv: str, output_csv: str):
    df = _read_csv(input_csv)
    cleaned_df, cleaning_reports = clean_all(df)
    _print_cleaning_table(cleaning_reports)

    out = output_csv or os.path.join(os.path.dirname(input_csv), "cleaned_data.csv")
    cleaned_df.to_csv(out, index=False)
    click.echo(f"\n[OK] 清洗后数据已输出: {out} (共 {len(cleaned_df)} 条)")


@cli.command("analyze", help="运行能耗/温度/启停机分析（含环境温度+库存校正）并对标")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
def cmd_analyze(input_csv: str):
    df = _read_csv(input_csv)
    cleaned_df, _ = clean_all(df)
    analyses = analyze_all(cleaned_df, raw_df=df)
    _print_analysis_table(analyses)

    advice_reports = generate_all_advice(analyses, cleaned_df)
    _print_advice_table(advice_reports)


@cli.command("benchmark", help="输出对标排名表（仅校正后吨日耗电排序）")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
def cmd_benchmark(input_csv: str):
    df = _read_csv(input_csv)
    cleaned_df, _ = clean_all(df)
    analyses = analyze_all(cleaned_df, raw_df=df)

    type_groups: dict = {}
    for a in analyses.values():
        type_groups.setdefault(a.warehouse_type, []).append(a)

    click.echo("\n======== 同库型对标排名 ========")
    for wh_type, items in type_groups.items():
        click.echo(f"\n[{wh_type}]")
        items_sorted = sorted(items, key=lambda x: x.adjusted_power_per_ton_day)
        click.echo(f"  排名  仓库                          校正吨日  环境校正  库存校正  温度达标")
        for rank, a in enumerate(items_sorted, 1):
            marker = "[1]" if rank == 1 else ("[!]" if rank == len(items_sorted) and rank > 1 else "   ")
            click.echo(
                f"  {marker}{rank:>2d}. {a.warehouse_name[:26]:<27s} {a.adjusted_power_per_ton_day:>8.2f} "
                f"x{a.ambient_adjustment_factor:>6.3f} x{a.inventory_adjustment_factor:>6.3f} "
                f"{a.temp_compliance_rate*100:>6.1f}%"
            )

    wh_a = analyses.get("WH-A001")
    wh_b = analyses.get("WH-B002")
    if wh_a and wh_b:
        click.echo("\n关键对比验证（易误判库的区分度）:")
        click.echo(f"  WH-A001 高温(环境热但正常):  校正吨日={wh_a.adjusted_power_per_ton_day:.2f}, 短循占比={wh_a.short_cycle_ratio*100:.1f}%, 等级={wh_a.benchmark_tier}")
        click.echo(f"  WH-B002 中温(环境普通但短循):  校正吨日={wh_b.adjusted_power_per_ton_day:.2f}, 短循占比={wh_b.short_cycle_ratio*100:.1f}%, 等级={wh_b.benchmark_tier}")
        if wh_a.short_cycle_ratio < wh_b.short_cycle_ratio * 0.5:
            click.echo("  [OK] 正确区分：WH-B002 短循环明显更严重")


@cli.command("report", help="一键完成：校验→清洗→分析→对标→导出完整报告")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output-dir", default="./report_output", show_default=True, help="报告输出目录")
@click.option("--period", default="", help="报告周期显示文本")
def cmd_report(input_csv: str, output_dir: str, period: str):
    df = _read_csv(input_csv)
    validations = validate_all(df)
    cleaned_df, cleanings = clean_all(df)
    analyses = analyze_all(cleaned_df, raw_df=df)
    advice_reports = generate_all_advice(analyses, cleaned_df)

    _print_validation_table(validations)
    _print_cleaning_table(cleanings)
    _print_analysis_table(analyses)
    _print_advice_table(advice_reports)

    outputs = export_all(
        output_dir=output_dir,
        cleaned_df=cleaned_df,
        analyses=analyses,
        advice_reports=advice_reports,
        validations=validations,
        cleanings=cleanings,
        report_period=period,
    )
    click.echo("\n======== 导出文件 ========")
    for name, path in outputs.items():
        click.echo(f"  [OK] {name}: {path}")
    click.echo(f"\n[OK] 报告已全部生成至目录: {output_dir}")


def main():
    cli(prog_name="cold-chain-report")


if __name__ == "__main__":
    main()
