"""
OpenAQ v3 API 传感器数据下载器
包含传感器经纬度信息
"""

import os, sys
from datetime import datetime, timedelta
import argparse
import pandas as pd
import time
import requests
from pandas import json_normalize
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAQSensorDownloaderComplete:
    def __init__(self, api_key: str):
        """
        初始化OpenAQ传感器下载器 - 完整版本

        Args:
            api_key: OpenAQ API密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v3"
        self.headers = {
            "X-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "OpenAQ-Sensor-Downloader/1.0",
        }
        logger.info("OpenAQ API客户端初始化成功")

    def get_us_locations_with_sensors(self, limit: int = 100) -> pd.DataFrame:
        """获取美国带有传感器信息的监测位置"""
        logger.info("获取美国的监测位置及其传感器信息...")

        locations = []
        page = 1

        while len(locations) < limit:
            try:
                params = {
                    "iso": "US",  # 明确指定美国
                    "limit": min(1e3, limit - len(locations)),
                    # 'limit': min(100, limit - len(locations)),
                    "page": page,
                }

                response = requests.get(
                    f"{self.base_url}/locations", headers=self.headers, params=params
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                locations.extend(results)
                logger.info(f"已获取 {len(locations)} 个位置...")

                # 检查是否还有更多数据
                meta = data.get("meta", {})
                if meta.get("page", 1) >= meta.get("pages", 1):
                    break

                page += 1
                time.sleep(0.2)

            except requests.exceptions.RequestException as e:
                logger.error(f"获取位置失败: {e}")
                break

        if locations:
            df = json_normalize(locations)
            logger.info(f"总共找到 {len(df)} 个监测位置")

            # 检查是否有传感器信息 - 修复pandas布尔索引问题
            has_sensors = False
            if "sensors" in df.columns:
                # 检查sensors列是否存在且不为空
                non_null_sensors = df["sensors"].notna()
                if non_null_sensors.any():
                    # 检查是否有实际的数据（列表长度大于0）
                    valid_sensors = df["sensors"].apply(
                        lambda x: isinstance(x, list) and len(x) > 0
                    )
                    if valid_sensors.any():
                        df_with_sensors = df[valid_sensors].copy()
                        has_sensors = True
                        logger.info(f"其中有传感器的位置数量: {len(df_with_sensors)}")

                        # 显示一些示例
                        if len(df_with_sensors) > 0:
                            logger.info(
                                f"示例位置: {df_with_sensors['name'].head(3).tolist()}"
                            )

                        return df_with_sensors

            if not has_sensors:
                logger.warning("没有找到有效的传感器信息")
                return df
        else:
            logger.warning("未找到美国的监测位置数据")
            return pd.DataFrame()

    def extract_sensors_with_coordinates(self, locations_df: pd.DataFrame) -> list:
        """从位置数据中提取传感器信息，包含经纬度坐标"""
        sensors = []

        if locations_df.empty:
            logger.warning("位置数据为空")
            return sensors

        if "sensors" not in locations_df.columns:
            logger.warning("位置数据中没有传感器列")
            return sensors

        # 获取坐标列名（处理不同的可能列名）
        lat_column = None
        lon_column = None

        possible_lat_columns = ["coordinates.latitude", "lat", "latitude"]
        possible_lon_columns = ["coordinates.longitude", "lon", "longitude"]

        for col in possible_lat_columns:
            if col in locations_df.columns:
                lat_column = col
                break

        for col in possible_lon_columns:
            if col in locations_df.columns:
                lon_column = col
                break

        logger.info(f"使用坐标列: latitude={lat_column}, longitude={lon_column}")

        for idx, location in locations_df.iterrows():
            location_id = location.get("id")
            location_name = location.get("name", f"Location_{location_id}")

            # 获取经纬度
            latitude = location.get(lat_column) if lat_column else None
            longitude = location.get(lon_column) if lon_column else None

            # 获取传感器数据
            sensors_data = location.get("sensors", [])

            # 确保sensors_data是列表
            if isinstance(sensors_data, list) and len(sensors_data) > 0:
                for sensor in sensors_data:
                    if isinstance(sensor, dict):
                        sensor_info = {
                            "sensor_id": sensor.get("id"),
                            "sensor_name": sensor.get("name"),
                            "location_id": location_id,
                            "location_name": location_name,
                            "latitude": latitude,
                            "longitude": longitude,
                            "parameter_id": (
                                sensor.get("parameter", {}).get("id")
                                if isinstance(sensor.get("parameter"), dict)
                                else None
                            ),
                            "parameter_name": (
                                sensor.get("parameter", {}).get("name")
                                if isinstance(sensor.get("parameter"), dict)
                                else None
                            ),
                            "parameter_units": (
                                sensor.get("parameter", {}).get("units")
                                if isinstance(sensor.get("parameter"), dict)
                                else None
                            ),
                        }
                        sensors.append(sensor_info)

        logger.info(f"从位置数据中提取了 {len(sensors)} 个传感器")

        # 显示一些示例
        if sensors:
            logger.info("传感器示例（前3个）:")
            for i, sensor in enumerate(sensors[:3]):
                logger.info(
                    f"  传感器 {i+1}: ID={sensor['sensor_id']}, 名称={sensor['sensor_name']}"
                )
                logger.info(f"    位置: {sensor['location_name']}")
                logger.info(f"    坐标: ({sensor['latitude']}, {sensor['longitude']})")
                logger.info(f"    参数: {sensor['parameter_name']}")

        return sensors

    def get_sensor_daily_data(
        self, sensor_id: int, days_back: int = 30
    ) -> pd.DataFrame:
        """获取传感器的日平均数据 - 最终版本"""
        logger.info(f"获取传感器 {sensor_id} 的日数据（最近{days_back}天）...")

        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            params = {
                "date_from": start_date.strftime("%Y-%m-%d"),
                "date_to": end_date.strftime("%Y-%m-%d"),
                "limit": 1000,
            }

            response = requests.get(
                f"{self.base_url}/sensors/{sensor_id}/days",
                headers=self.headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                logger.info(f"传感器 {sensor_id}: 获取到 {len(results)} 条日记录")

                if results:
                    df = json_normalize(results)
                    df["sensor_id"] = sensor_id
                    df["aggregation"] = "daily"  # 'hourly'
                    return df
            elif response.status_code == 404:
                logger.info(f"传感器 {sensor_id} 没有日数据")
            else:
                logger.warning(
                    f"传感器 {sensor_id} 日数据请求失败: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"获取传感器 {sensor_id} 日数据失败: {e}")

        return pd.DataFrame()

    def download_recent_sensor_data(
        self,
        country_code: str = "US",
        days_back: int = 30,
        max_sensors: int = 10,
        output_dir: str = "./openaq_recent_data",
    ):
        """
        下载最近时间的传感器数据 - 完整版本

        Args:
            country_code: ISO国家代码
            days_back: 回溯天数
            max_sensors: 最大传感器数量
            output_dir: 输出目录
        """
        logger.info(f"开始下载 {country_code} 最近 {days_back} 天的传感器日数据...")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取带有传感器的位置
        locations_df = self.get_us_locations_with_sensors(limit=10000)

        if locations_df.empty:
            logger.error(f"未找到 {country_code} 的监测位置")
            return

        # 提取传感器信息（包含坐标）
        sensors = self.extract_sensors_with_coordinates(locations_df)

        if not sensors:
            logger.error("未找到任何传感器")
            return

        # 保存传感器信息（包含坐标）
        sensors_df = pd.DataFrame(sensors)
        if not sensors_df.empty:
            sensors_file = os.path.join(
                output_dir, f"{country_code}_recent_sensors_with_coords.csv"
            )
            sensors_df.to_csv(sensors_file, index=False)
            logger.info(f"传感器信息（含坐标）已保存到: {sensors_file}")
            logger.info(f"传感器信息包含列: {sensors_df.columns.tolist()}")

            # 显示坐标统计信息
            if "latitude" in sensors_df.columns and "longitude" in sensors_df.columns:
                valid_coords = sensors_df[["latitude", "longitude"]].notna().all(axis=1)
                logger.info(f"有有效坐标的传感器数量: {valid_coords.sum()}")

                if valid_coords.any():
                    lat_range = (
                        sensors_df.loc[valid_coords, "latitude"].min(),
                        sensors_df.loc[valid_coords, "latitude"].max(),
                    )
                    lon_range = (
                        sensors_df.loc[valid_coords, "longitude"].min(),
                        sensors_df.loc[valid_coords, "longitude"].max(),
                    )
                    logger.info(f"纬度范围: {lat_range[0]:.4f} 到 {lat_range[1]:.4f}")
                    logger.info(f"经度范围: {lon_range[0]:.4f} 到 {lon_range[1]:.4f}")

        all_measurements = []
        successful_sensors = 0

        # 限制处理的传感器数量
        max_sensors = min(max_sensors, len(sensors))
        logger.info(f"将处理前 {max_sensors} 个传感器")

        # 遍历每个传感器获取数据
        for i, sensor in enumerate(sensors[:max_sensors]):
            sensor_id = sensor["sensor_id"]
            sensor_name = sensor.get("sensor_name", f"Sensor_{sensor_id}")
            location_id = sensor["location_id"]
            parameter_name = sensor.get("parameter_name", "Unknown")
            latitude = sensor.get("latitude")
            longitude = sensor.get("longitude")

            if not sensor_id:
                continue

            logger.info(
                f"处理传感器 {i+1}/{max_sensors}: {sensor_id} ({parameter_name})..."
            )
            if latitude is not None and longitude is not None:
                logger.info(f"  位置坐标: ({latitude:.4f}, {longitude:.4f})")

            # 获取日数据
            measurements_df = self.get_sensor_daily_data(sensor_id, days_back)

            if not measurements_df.empty:
                # 添加传感器和位置信息（包含坐标）
                measurements_df["sensor_id"] = sensor_id
                measurements_df["sensor_name"] = sensor_name
                measurements_df["location_id"] = location_id
                measurements_df["parameter_name"] = parameter_name
                measurements_df["country_code"] = country_code
                measurements_df["latitude"] = latitude
                measurements_df["longitude"] = longitude

                all_measurements.append(measurements_df)
                successful_sensors += 1
                logger.info(
                    f"  传感器 {sensor_id}: 添加 {len(measurements_df)} 条记录到总数据集"
                )
            else:
                logger.info(f"  传感器 {sensor_id}: 无数据")

            # 避免请求过快
            time.sleep(0.5)

        if all_measurements:
            # 合并所有数据
            final_df = pd.concat(all_measurements, ignore_index=True)

            # 保存合并数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

            output_file = os.path.join(
                output_dir, f"{country_code}_{date_str}_sensor_daily_with_coords.csv"
            )
            final_df.to_csv(output_file, index=False)

            logger.info(f"传感器数据下载完成!")
            logger.info(f"总记录数: {len(final_df)}")
            logger.info(f"成功传感器数: {successful_sensors}")
            logger.info(f"输出文件: {output_file}")

            # 显示数据摘要
            print("\n=== 数据摘要 ===")
            print(f"总记录数: {len(final_df)}")
            print(f"成功传感器数: {successful_sensors}")
            if "sensor_id" in final_df.columns:
                print(f"传感器数量: {final_df['sensor_id'].nunique()}")
            if "parameter_name" in final_df.columns:
                print(f"参数类型: {final_df['parameter_name'].unique()}")
            if "latitude" in final_df.columns and "longitude" in final_df.columns:
                valid_coords = final_df[["latitude", "longitude"]].notna().all(axis=1)
                print(f"有有效坐标的记录数: {valid_coords.sum()}")
                if valid_coords.any():
                    print(
                        f"纬度范围: {final_df.loc[valid_coords, 'latitude'].min():.4f} 到 {final_df.loc[valid_coords, 'latitude'].max():.4f}"
                    )
                    print(
                        f"经度范围: {final_df.loc[valid_coords, 'longitude'].min():.4f} 到 {final_df.loc[valid_coords, 'longitude'].max():.4f}"
                    )

            # 显示数据的前几列
            if len(final_df) > 0:
                print(f"数据列名: {final_df.columns.tolist()}")
                print("\n前5行数据:")
                print(final_df.head())

        else:
            logger.warning("没有获取到任何传感器数据")


if __name__ == "__main__":
    API_KEY_ENV_VAR = "OPENAQ_API_KEY"
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError("未设置环境变量OPENAQ_API_KEY")

    # 1. 创建下载器并下载数据
    downloader = OpenAQSensorDownloaderComplete(api_key)

    local_path = os.path.join(os.path.dirname(__file__), "../../", "data/raw/")
    downloader.download_recent_sensor_data(
        country_code="US",
        days_back=365,  # 30
        max_sensors=3000,  # 20
        output_dir=local_path,
    )

    # 2. 读取并只保留需要的 7 列
    cols = [
        "value",
        "parameter.name",
        "period.datetimeFrom.utc",
        "latitude",
        "longitude",
        "period.interval",
        "parameter.units",
    ]
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../../",
        "data/raw/US_20250101_20260118_sensor_daily_with_coords.csv",
    )
    df = pd.read_csv(csv_path, usecols=cols)

    df = df[cols]

    # 3. 写出到新的 csv
    filtered_path = os.path.join(
        os.path.dirname(__file__),
        "../../",
        "data/processed/US_20250101_20260118_sensor_filtered.csv",
    )
    df.to_csv(filtered_path, index=False)
