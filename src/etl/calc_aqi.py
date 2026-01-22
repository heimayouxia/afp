import os
import pandas as pd
from typing import Callable, Optional


def add_aqi_column(
    in_csv: str,
    out_csv: str,
    func: Callable[[str, str, str, float], Optional[float]],
    *,
    dtype: Optional[dict] = None,
) -> None:
    """
    为 CSV 增加一列 'aqi' 并保存为新文件。
    若 func 返回 None，则该行被丢弃。

    参数
    ----
    in_csv  : 输入文件路径
    out_csv : 输出文件路径
    func    : 计算 AQI 的函数，签名
              func(parameter, period, unit, value) -> float | None
    dtype   : 可选，手动指定列类型
    """
    # 1. 读入
    cols = [
        "value",
        "parameter.name",
        "period.datetimeFrom.utc",
        "latitude",
        "longitude",
        "period.interval",
        "parameter.units",
    ]
    df = pd.read_csv(in_csv, usecols=cols, dtype=dtype)

    # 2. 清洗 value 列（可重复利用之前逻辑）
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .replace({"": pd.NA, "N/A": pd.NA, "NULL": pd.NA})
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 3. 计算 AQI，允许 None
    def _calc(row):
        return func(
            row["parameter.name"],
            row["period.interval"],
            row["parameter.units"],
            row["value"],
        )

    df["aqi"] = df.apply(_calc, axis=1)

    # 4. 丢弃 None 行
    df = df[df["aqi"].notna()]

    # 5. 写出
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"已生成 {out_csv}，共 {len(df)} 行（None 行已剔除）。")


def convert_to_aqi(parameter, period, unit, value):
    aqi = None
    if parameter == "o3" and period == "08:00:00":
        o3_val = int(value * 1000) / 1000.0
        if o3_val >= 0.0 and o3_val <= 0.054:
            aqi = (50 - 0) / (0.054 - 0.0) * (o3_val - 0.0) + 0
        elif o3_val >= 0.055 and o3_val <= 0.070:
            aqi = (100 - 51) / (0.070 - 0.055) * (o3_val - 0.055) + 51
        elif o3_val >= 0.071 and o3_val <= 0.085:
            aqi = (150 - 101) / (0.085 - 0.071) * (o3_val - 0.071) + 101
        elif o3_val >= 0.086 and o3_val <= 0.105:
            aqi = (200 - 151) / (0.105 - 0.086) * (o3_val - 0.086) + 151
        elif o3_val >= 0.106 and o3_val < 0.200:
            aqi = (300 - 201) / (0.200 - 0.106) * (o3_val - 0.106) + 201
        else:
            aqi = 300
    elif parameter == "o3" and period == "01:00:00":
        o3_val = int(value * 1000) / 1000.0
        if o3_val >= 0.125 and o3_val <= 0.164:
            aqi = (150 - 101) / (0.164 - 0.125) * (o3_val - 0.125) + 101
        elif o3_val >= 0.165 and o3_val <= 0.204:
            aqi = (200 - 151) / (0.204 - 0.165) * (o3_val - 0.165) + 151
        elif o3_val >= 0.205 and o3_val <= 0.404:
            aqi = (300 - 201) / (0.404 - 0.205) * (o3_val - 0.205) + 201
        elif o3_val >= 0.405 and o3_val <= 0.604:
            aqi = (500 - 301) / (0.604 - 0.405) * (o3_val - 0.405) + 301
        elif o3_val >= 0.605:
            aqi = (500 - 301) / (0.604 - 0.405) * (o3_val - 0.605) + 500
        else:
            aqi = None
    elif parameter == "pm25" and period == "24:00:00":
        pm25_val = int(value * 10) / 10.0
        if pm25_val >= 0.0 and pm25_val <= 9.0:
            aqi = (50 - 0) / (9.0 - 0.0) * (pm25_val - 0.0) + 0
        elif pm25_val >= 9.1 and pm25_val <= 35.4:
            aqi = (100 - 51) / (35.4 - 9.1) * (pm25_val - 9.1) + 51
        elif pm25_val >= 35.5 and pm25_val <= 55.4:
            aqi = (150 - 101) / (55.4 - 35.5) * (pm25_val - 35.5) + 101
        elif pm25_val >= 55.5 and pm25_val <= 125.4:
            aqi = (200 - 151) / (125.4 - 55.5) * (pm25_val - 55.5) + 151
        elif pm25_val >= 125.5 and pm25_val <= 225.4:
            aqi = (300 - 201) / (225.4 - 125.5) * (pm25_val - 125.5) + 201
        elif pm25_val >= 225.5 and pm25_val <= 325.4:
            aqi = (500 - 301) / (325.4 - 225.5) * (pm25_val - 225.5) + 301
        elif pm25_val >= 325.5:
            aqi = (500 - 301) / (325.4 - 225.5) * (pm25_val - 325.5) + 500
        else:
            aqi = None
    elif parameter == "pm10" and period == "24:00:00":
        pm10_val = int(value)
        if pm10_val >= 0 and pm10_val <= 54:
            aqi = (50 - 0) / (54.0 - 0.0) * (pm10_val - 0.0) + 0
        elif pm10_val >= 55 and pm10_val <= 154:
            aqi = (100 - 51) / (154.0 - 55.0) * (pm10_val - 55.0) + 51
        elif pm10_val >= 155 and pm10_val <= 254:
            aqi = (150 - 101) / (254.0 - 155.0) * (pm10_val - 155.0) + 101
        elif pm10_val >= 255 and pm10_val <= 354:
            aqi = (200 - 151) / (354.0 - 255.0) * (pm10_val - 255.0) + 151
        elif pm10_val >= 355 and pm10_val <= 424:
            aqi = (300 - 201) / (424.0 - 355.0) * (pm10_val - 355.0) + 201
        elif pm10_val >= 425 and pm10_val <= 604:
            aqi = (500 - 301) / (604.0 - 425.0) * (pm10_val - 425.0) + 301
        elif pm10_val >= 605:
            aqi = (500 - 301) / (604.0 - 425.0) * (pm10_val - 605) + 500
        else:
            aqi = None
    elif parameter == "co" and period == "08:00:00":
        co_val = int(value * 10) / 10.0
        if co_val >= 0.0 and co_val <= 4.4:
            aqi = (50 - 0) / (4.4 - 0.0) * (co_val - 0.0) + 0
        elif co_val >= 4.5 and co_val <= 9.4:
            aqi = (100 - 51) / (9.4 - 4.5) * (co_val - 4.5) + 51
        elif co_val >= 9.5 and co_val <= 12.4:
            aqi = (150 - 101) / (12.4 - 9.5) * (co_val - 9.5) + 101
        elif co_val >= 12.5 and co_val <= 15.4:
            aqi = (200 - 151) / (15.4 - 12.5) * (co_val - 12.5) + 151
        elif co_val >= 15.5 and co_val <= 30.4:
            aqi = (300 - 201) / (30.4 - 15.5) * (co_val - 15.5) + 201
        elif co_val >= 30.5 and co_val <= 50.4:
            aqi = (500 - 301) / (50.4 - 30.5) * (co_val - 30.5) + 301
        elif co_val >= 50.5:
            aqi = (500 - 301) / (50.4 - 30.5) * (co_val - 50.5) + 500
        else:
            aqi = None
    elif parameter == "so2" and period == "01:00:00":
        so2_val = int(value) if unit == "ppb" else int(value * 1000)
        if so2_val >= 0 and so2_val <= 35:
            aqi = (50 - 0) / (35.0 - 0.0) * (so2_val - 0.0) + 0
        elif so2_val >= 36 and so2_val <= 75:
            aqi = (100 - 51) / (75.0 - 36.0) * (so2_val - 36.0) + 51
        elif so2_val >= 76 and so2_val <= 185:
            aqi = (150 - 101) / (185.0 - 76.0) * (so2_val - 76.0) + 101
        elif so2_val >= 186 and so2_val <= 304:
            aqi = (200 - 151) / (304.0 - 186.0) * (so2_val - 186.0) + 151
        else:
            aqi = 200
    elif parameter == "no2" and period == "01:00:00":
        no2_val = int(value) if unit == "ppb" else int(value * 1000)
        if no2_val >= 0 and no2_val <= 53:
            aqi = (50 - 0) / (53.0 - 0.0) * (no2_val - 0.0) + 0
        elif no2_val >= 54 and no2_val <= 100:
            aqi = (100 - 51) / (100.0 - 54.0) * (no2_val - 54.0) + 51
        elif no2_val >= 101 and no2_val <= 360:
            aqi = (150 - 101) / (360.0 - 101.0) * (no2_val - 101.0) + 101
        elif no2_val >= 361 and no2_val <= 649:
            aqi = (200 - 151) / (649.0 - 361.0) * (no2_val - 361.0) + 151
        elif no2_val >= 650 and no2_val <= 1249:
            aqi = (300 - 201) / (1249.0 - 650.0) * (no2_val - 650.0) + 201
        elif no2_val >= 1250 and no2_val <= 2049:
            aqi = (500 - 301) / (2049.0 - 1250.0) * (no2_val - 1250.0) + 301
        elif no2_val >= 2050:
            aqi = (500 - 301) / (2049.0 - 1250.0) * (no2_val - 2050.0) + 500
        else:
            aqi = None
    else:
        aqi = None

    return round(aqi) if aqi is not None else None


if __name__ == "__main__":
    filtered_path = os.path.join(
        os.path.dirname(__file__),
        "../../",
        "data/processed/US_20250101_20260118_sensor_filtered.csv",
    )
    aqi_added_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/processed/US_sensor_with_aqi.csv"
    )
    add_aqi_column(filtered_path, aqi_added_path, convert_to_aqi)
