import os, sys
import pandas as pd
import boto3, requests, pyarrow as pa
from datetime import date, timedelta
import tarfile
import glob

if __name__ == "__main__":
    ### NOAA数据ETL
    # 1. 下载 NOAA 2025年全球地面站
    noaa_file = "2025.tar.gz"
    url = (
        "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{0:s}".format(
            noaa_file
        )
    )
    local_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/raw/{0:s}".format(noaa_file)
    )
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2. 解压 & 读 CSV
    extract_to = os.path.join(
        os.path.dirname(__file__), "../../", "data/raw/{0:s}".format("2025_temp")
    )
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(local_path, "r:gz") as t:
        t.extractall(path=extract_to)

    # 3. 合并所有 csv（包里全是 *.csv）
    csv_files = glob.glob(os.path.join(extract_to, "*.csv"))
    print("concating")
    df_noaa = pd.concat(
        [pd.read_csv(f, parse_dates=["DATE"]) for f in csv_files], ignore_index=True
    )
    print("concat done")

    # 4. 筛选美国站
    mask = df_noaa["NAME"].astype(str).fillna("").str.endswith("US")
    us_noaa = df_noaa[mask]

    # 5. 保存
    us_noaa_path = os.path.join(
        os.path.dirname(__file__), "../../", "data/raw/NOAA_GSOD_US_2025.csv"
    )
    us_noaa.to_csv(us_noaa_path, index=False)

    # 6. 只保留需要的 18 列
    cols = [
        "DATE",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "NAME",
        "TEMP",
        "DEWP",
        "SLP",
        "STP",
        "VISIB",
        "WDSP",
        "MXSPD",
        "GUST",
        "MAX",
        "MIN",
        "PRCP",
        "SNDP",
        "FRSHTT",
    ]
    df = pd.read_csv(us_noaa_path, usecols=cols)

    df = df[cols]

    # 7. 写出到新的 csv
    filtered_noaa_path = os.path.join(
        os.path.dirname(__file__),
        "../../",
        "data/processed/NOAA_GSOD_US_2025_filtered.csv",
    )
    df.to_csv(filtered_noaa_path, index=False)
