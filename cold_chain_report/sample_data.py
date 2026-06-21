"""数据生成模块 - 生成多座冷库小时级样例数据"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

COLUMNS = [
    "warehouse_id", "warehouse_name", "warehouse_type",
    "timestamp", "power_kwh", "temp_celsius", "ambient_temp",
    "compressor_status", "door_open_count", "defrost_status",
    "inventory_kg", "meter_reading"
]

WAREHOUSE_CONFIGS = [
    {
        "warehouse_id": "WH-A001",
        "warehouse_name": "高温蔬果冷库（新设备）",
        "warehouse_type": "高温库",
        "target_temp": 4.0,
        "temp_tolerance": 1.5,
        "base_power": 40.0,
        "cooling_efficiency": 0.9,
        "ambient_base": 32.0,
        "ambient_amp": 8.0,
        "normal_door_count_range": (1, 4),
        "defrost_hours": [6, 14, 22],
        "inventory_range": (15000, 25000),
        "anomaly_type": "HIGH_AMBIENT_NORMAL_OP",
    },
    {
        "warehouse_id": "WH-A002",
        "warehouse_name": "高温果蔬冷库（普通设备）",
        "warehouse_type": "高温库",
        "target_temp": 4.0,
        "temp_tolerance": 1.5,
        "base_power": 45.0,
        "cooling_efficiency": 0.7,
        "ambient_base": 29.0,
        "ambient_amp": 7.0,
        "normal_door_count_range": (1, 5),
        "defrost_hours": [6, 14, 22],
        "inventory_range": (12000, 22000),
        "anomaly_type": "NORMAL",
    },
    {
        "warehouse_id": "WH-A003",
        "warehouse_name": "高温鲜品冷库（老旧设备）",
        "warehouse_type": "高温库",
        "target_temp": 4.0,
        "temp_tolerance": 1.5,
        "base_power": 55.0,
        "cooling_efficiency": 0.55,
        "ambient_base": 27.0,
        "ambient_amp": 6.0,
        "normal_door_count_range": (2, 6),
        "defrost_hours": [3, 9, 15, 21],
        "inventory_range": (10000, 20000),
        "anomaly_type": "INEFFICIENT_EQUIPMENT",
    },
    {
        "warehouse_id": "WH-B001",
        "warehouse_name": "中温肉品冷库（运行良好）",
        "warehouse_type": "中温库",
        "target_temp": -18.0,
        "temp_tolerance": 2.0,
        "base_power": 70.0,
        "cooling_efficiency": 0.8,
        "ambient_base": 26.0,
        "ambient_amp": 6.0,
        "normal_door_count_range": (0, 3),
        "defrost_hours": [4, 12, 20],
        "inventory_range": (20000, 35000),
        "anomaly_type": "NORMAL",
    },
    {
        "warehouse_id": "WH-B002",
        "warehouse_name": "中温肉类冷库（短循环严重）",
        "warehouse_type": "中温库",
        "target_temp": -18.0,
        "temp_tolerance": 2.0,
        "base_power": 80.0,
        "cooling_efficiency": 0.75,
        "ambient_base": 26.0,
        "ambient_amp": 6.0,
        "normal_door_count_range": (0, 3),
        "defrost_hours": [4, 12, 20],
        "inventory_range": (20000, 35000),
        "anomaly_type": "SHORT_CYCLE_COMPRESSOR",
    },
    {
        "warehouse_id": "WH-C003",
        "warehouse_name": "低温速冻冷库",
        "warehouse_type": "低温库",
        "target_temp": -30.0,
        "temp_tolerance": 2.5,
        "base_power": 150.0,
        "cooling_efficiency": 0.85,
        "ambient_base": 28.0,
        "ambient_amp": 7.0,
        "normal_door_count_range": (0, 2),
        "defrost_hours": [8, 20],
        "inventory_range": (10000, 18000),
        "anomaly_type": "NORMAL",
    },
    {
        "warehouse_id": "WH-D004",
        "warehouse_name": "恒温保鲜冷库",
        "warehouse_type": "恒温库",
        "target_temp": 0.0,
        "temp_tolerance": 1.0,
        "base_power": 45.0,
        "cooling_efficiency": 0.78,
        "ambient_base": 25.0,
        "ambient_amp": 5.0,
        "normal_door_count_range": (1, 5),
        "defrost_hours": [2, 10, 18],
        "inventory_range": (8000, 15000),
        "anomaly_type": "NORMAL",
    },
]


def _generate_hourly_timestamps(start_date: datetime, days: int = 30) -> pd.DatetimeIndex:
    end = start_date + timedelta(days=days)
    return pd.date_range(start=start_date, end=end, freq="h", inclusive="left")


def _ambient_temp_at(hour: int, base: float, amp: float) -> float:
    phase = (hour - 14) / 24 * 2 * np.pi
    daily = -amp * np.cos(phase)
    noise = np.random.normal(0, 0.8)
    return round(base + daily + noise, 1)


def _generate_warehouse_data(cfg: dict, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    compressor_on = True
    meter_reading = 100000.0 + random.uniform(0, 5000)
    inventory_level = random.uniform(*cfg["inventory_range"])
    short_cycle_countdown = 0

    cooling_eff = cfg.get("cooling_efficiency", 0.7)
    if cfg.get("anomaly_type") == "INEFFICIENT_EQUIPMENT":
        cooling_eff *= 0.7

    current_temp = cfg["target_temp"] + 0.5

    for ts in timestamps:
        hour = ts.hour
        ambient = _ambient_temp_at(hour, cfg["ambient_base"], cfg["ambient_amp"])
        inventory_level += np.random.normal(0, 120)
        inventory_level = float(np.clip(inventory_level, *cfg["inventory_range"]))

        temp_diff = max(0.1, ambient - cfg["target_temp"])
        heat_gain_per_hour = temp_diff * 0.018
        cooling_per_hour = cfg["base_power"] * 0.075 * cooling_eff

        target_temp = cfg["target_temp"]
        if hour >= 22 or hour < 6:
            target_temp += 1.5

        hysteresis_high = target_temp + cfg["temp_tolerance"] * 0.6
        hysteresis_low = target_temp - cfg["temp_tolerance"] * 0.4

        if cfg["anomaly_type"] == "SHORT_CYCLE_COMPRESSOR":
            if short_cycle_countdown > 0:
                short_cycle_countdown -= 1
                if random.random() < 0.7:
                    compressor_on = not compressor_on
            elif random.random() < 0.2:
                short_cycle_countdown = random.randint(2, 6)
                compressor_on = not compressor_on
            else:
                if not compressor_on and current_temp > hysteresis_high:
                    compressor_on = True
                elif compressor_on and current_temp < hysteresis_low:
                    compressor_on = False
        else:
            if not compressor_on and current_temp > hysteresis_high:
                compressor_on = True
            elif compressor_on and current_temp < hysteresis_low:
                compressor_on = False

        door_count = random.randint(*cfg["normal_door_count_range"])
        if 8 <= hour <= 18:
            door_count = int(door_count * 1.5)
        if random.random() < 0.02:
            door_count += random.randint(3, 8)

        defrost = 1 if hour in cfg["defrost_hours"] else 0
        if random.random() < 0.005:
            defrost = 1

        temp_delta = 0.0
        if compressor_on:
            temp_delta -= cooling_per_hour
        temp_delta += heat_gain_per_hour
        if defrost:
            temp_delta += 0.4
        temp_delta += door_count * 0.02
        temp_delta += np.random.normal(0, 0.04)

        current_temp += temp_delta
        current_temp = float(np.clip(
            current_temp,
            cfg["target_temp"] - cfg["temp_tolerance"] * 4,
            cfg["target_temp"] + cfg["temp_tolerance"] * 5
        ))

        power = 0.0
        if compressor_on:
            compressor_power = cfg["base_power"] * (1.0 + heat_gain_per_hour * 0.8)
            compressor_power *= random.uniform(0.92, 1.08)
            power += compressor_power
        if defrost:
            power += cfg["base_power"] * 0.25 * random.uniform(0.9, 1.1)
        power += door_count * cfg["base_power"] * 0.006
        power = round(max(0.0, power), 2)
        meter_reading += power + random.uniform(-0.05, 0.05)

        rows.append({
            "warehouse_id": cfg["warehouse_id"],
            "warehouse_name": cfg["warehouse_name"],
            "warehouse_type": cfg["warehouse_type"],
            "timestamp": ts,
            "power_kwh": power,
            "temp_celsius": round(current_temp, 2),
            "ambient_temp": ambient,
            "compressor_status": 1 if compressor_on else 0,
            "door_open_count": door_count,
            "defrost_status": defrost,
            "inventory_kg": round(inventory_level, 0),
            "meter_reading": round(meter_reading, 2),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def _inject_data_quality_issues(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    rng = random.Random(hash(cfg["warehouse_id"]) & 0xFFFFFFFF)

    missing_count = max(2, int(n * 0.015))
    missing_indices = sorted(rng.sample(range(4, n - 4), missing_count))
    df = df.drop(index=missing_indices).reset_index(drop=True)

    dup_count = rng.randint(3, 8)
    dup_indices = rng.sample(range(len(df) - 1), dup_count)
    dup_rows = []
    for idx in dup_indices:
        row = df.iloc[idx].copy()
        row["power_kwh"] = round(row["power_kwh"] * rng.uniform(0.95, 1.05), 2)
        dup_rows.append(row)
    df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)

    wrong_idx = rng.sample(range(len(df)), rng.randint(2, 5))
    for idx in wrong_idx:
        issue_type = rng.choice(["temp", "power", "ambient", "inventory", "meter"])
        if issue_type == "temp":
            df.at[idx, "temp_celsius"] = rng.choice([999.0, -999.0, 100.0, -100.0])
        elif issue_type == "power":
            df.at[idx, "power_kwh"] = rng.choice([-50.0, 9999.0, -1.0])
        elif issue_type == "ambient":
            df.at[idx, "ambient_temp"] = rng.choice([-50.0, 200.0])
        elif issue_type == "inventory":
            df.at[idx, "inventory_kg"] = rng.choice([-100.0, 999999.0])
        elif issue_type == "meter":
            df.at[idx, "meter_reading"] = df.at[idx, "meter_reading"] - rng.uniform(200, 800)

    if cfg["anomaly_type"] != "SHORT_CYCLE_COMPRESSOR":
        jump_idx = rng.sample(range(1, len(df) - 1), rng.randint(1, 3))
        for idx in jump_idx:
            if df.at[idx, "compressor_status"] == 0:
                df.at[idx, "compressor_status"] = 1
                if idx + 1 < len(df):
                    df.at[idx + 1, "compressor_status"] = 0

    meter_bad_idx = rng.sample(range(20, len(df)), rng.randint(1, 3))
    for idx in meter_bad_idx:
        if idx > 0:
            prev = df.at[idx - 1, "meter_reading"]
            df.at[idx, "meter_reading"] = prev - rng.uniform(50, 300)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_sample_data(output_dir: str, start_date: str = "2026-05-01", days: int = 30) -> str:
    os.makedirs(output_dir, exist_ok=True)
    start = pd.Timestamp(start_date)
    timestamps = _generate_hourly_timestamps(start.to_pydatetime(), days)

    all_frames = []
    csv_paths = []
    for cfg in WAREHOUSE_CONFIGS:
        clean_df = _generate_warehouse_data(cfg, timestamps)
        df = _inject_data_quality_issues(clean_df, cfg)
        csv_path = os.path.join(output_dir, f"{cfg['warehouse_id']}_hourly.csv")
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        all_frames.append(df)

    merged_path = os.path.join(output_dir, "all_warehouses_monthly.csv")
    pd.concat(all_frames, ignore_index=True).to_csv(merged_path, index=False)

    print(f"已生成 {len(WAREHOUSE_CONFIGS)} 座冷库的样例数据：")
    for p in csv_paths:
        print(f"  - {p}")
    print(f"合并文件: {merged_path}")
    return merged_path


def load_sample_data(data_dir: str) -> pd.DataFrame:
    merged_path = os.path.join(data_dir, "all_warehouses_monthly.csv")
    if os.path.exists(merged_path):
        return pd.read_csv(merged_path, parse_dates=["timestamp"])
    frames = []
    for fname in os.listdir(data_dir):
        if fname.endswith("_hourly.csv"):
            fpath = os.path.join(data_dir, fname)
            frames.append(pd.read_csv(fpath, parse_dates=["timestamp"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
