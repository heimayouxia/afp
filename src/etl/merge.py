import os
import pandas as pd
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm
from pathlib import Path

EARTH_RADIUS_KM = 6371.0088


def add_nearby_max_aqi(
    csv_a: str, csv_b: str, out_csv: str, dist_km: float = 50
) -> None:
    """
    同日期 + 50 km 内最大 AQI，无匹配则丢弃该行。
    """
    # 1. 读数据
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    df_a.rename(columns=str.upper, inplace=True)
    df_b.rename(columns=str.upper, inplace=True)

    # 2. 统一日期键（只保留年月日）
    df_a["DATE"] = pd.to_datetime(df_a["DATE"], errors="coerce")
    df_b["PERIOD.DATETIMEFROM.UTC"] = pd.to_datetime(
        df_b["PERIOD.DATETIMEFROM.UTC"], errors="coerce"
    )
    df_a["date_key"] = df_a["DATE"].dt.date
    df_b["date_key"] = df_b["PERIOD.DATETIMEFROM.UTC"].dt.date

    # 3. 按日期分组 B，并为每组预建 BallTree（球面弧度坐标）
    trees = {}
    for d_key, sub in df_b.groupby("date_key"):
        rad = np.deg2rad(sub[["LATITUDE", "LONGITUDE"]].values)
        trees[d_key] = (BallTree(rad, metric="haversine"), sub)

    # 4. 遍历 A
    keep = []
    for _, row in tqdm(df_a.iterrows(), total=len(df_a), desc="Processing"):
        d_key = row["date_key"]
        if pd.isna(d_key) or d_key not in trees:
            continue
        tree, sub_b = trees[d_key]
        loc_rad = np.deg2rad([[row["LATITUDE"], row["LONGITUDE"]]])
        idx = tree.query_radius(loc_rad, r=dist_km / EARTH_RADIUS_KM)[0]
        if len(idx) == 0:
            continue
        max_aqi = sub_b.iloc[idx]["AQI"].max()
        row["max_aqi"] = max_aqi
        keep.append(row)

    # 5. 输出
    df_out = pd.DataFrame(keep).drop(columns=["date_key"])
    df_out.to_csv(out_csv, index=False)
    print(f"Done -> {out_csv}  共保留 {len(df_out)} 行")


def split_frshtt(s):
    """
    把 FRSHTT 字符串拆成 6 个 0/1 整数
    缺失/空值 -> 全 0
    """
    if pd.isna(s):
        return [0] * 6
    s = str(s).strip().zfill(6)[-6:]  # 补前导 0，取后 6 位
    return [int(ch) for ch in s]


def add_frshtt_flags(csv_in: str, csv_out: str | None = None):
    """
    csv_in : 原始文件路径
    csv_out: 输出文件路径，若为 None 则默认在原文件名后加 '_flags'
    """
    csv_in = Path(csv_in)
    df = pd.read_csv(csv_in)

    # 生成 6 列
    frshtt_cols = ["Fog", "Rain", "Snow", "Hail", "Thunder", "Tornado"]
    df[frshtt_cols] = pd.DataFrame(
        df["FRSHTT"].apply(split_frshtt).tolist(), index=df.index
    )

    # 构造输出路径
    if csv_out is None:
        csv_out = csv_in.with_name(csv_in.stem + "_flags.csv")

    df = flag_to_nan(df)
    df.to_csv(csv_out, index=False)
    print(f"Saved → {csv_out}")


def flag_to_nan(df):
    """把缺测码统一换成 NaN"""
    df = df.copy()
    df.loc[df["DEWP"].between(9999.8, 9999.95, inclusive="both"), "DEWP"] = np.nan
    df.loc[df["SLP"].between(9999.8, 9999.95, inclusive="both"), "SLP"] = np.nan
    df.loc[df["STP"].between(999.8, 999.95, inclusive="both"), "STP"] = np.nan
    df.loc[df["VISIB"].between(999.8, 999.95, inclusive="both"), "VISIB"] = np.nan
    df.loc[df["WDSP"].between(999.8, 999.95, inclusive="both"), "WDSP"] = np.nan
    df.loc[df["MXSPD"].between(999.8, 999.95, inclusive="both"), "MXSPD"] = np.nan
    df.loc[df["GUST"].between(999.8, 999.95, inclusive="both"), "GUST"] = np.nan
    df.loc[df["PRCP"].between(99.98, 99.995, inclusive="both"), "PRCP"] = np.nan
    df.loc[df["SNDP"].between(999.8, 999.95, inclusive="both"), "SNDP"] = np.nan

    return df


if __name__ == "__main__":
    noaa_filtered_path = os.path.join(
        os.path.dirname(__file__),
        "../../",
        "data/processed/NOAA_GSOD_US_2025_filtered.csv",
    )
    aqi_added_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/processed/US_sensor_with_aqi.csv"
    )
    merged_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/processed/noaa_openaq_aqi.csv"
    )
    add_nearby_max_aqi(noaa_filtered_path, aqi_added_path, merged_path)

    frshtt_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/processed/noaa_openaq_aqi_frshtt.csv"
    )
    add_frshtt_flags(merged_path, frshtt_path)
