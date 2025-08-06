import sys
import os
import re
import shutil
import pyodbc
import fnmatch
import datetime
import pywintypes
import pandas as pd
import win32com.client as com
from decimal import Decimal
from dotenv import load_dotenv
from contextlib import redirect_stdout









def set_dpi_awareness():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError:
        print("ctypes 모듈을 가져오는 데 실패했습니다. Python 환경을 확인하세요.")
    except AttributeError:
        print("SetProcessDpiAwareness 함수가 지원되지 않는 환경입니다. Windows 8.1 이상에서만 지원됩니다.")
    except OSError as os_error:
        print(f"OS 관련 오류 발생: {os_error}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
set_dpi_awareness()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))









# ============================================================================ [ 현재 일시 설정 및 전날 시간 계산 ]

# 현재 시간을 자동으로 가져오는 부분
current_datetime = datetime.datetime.now()

# >> 아래 변수는 테스트를 위해 수동으로 현재시간을 지정하는 부분입니다. 현재 시각을 수동 지정하려면 아래 주석을 해제하세요.
current_datetime = datetime.datetime.strptime("2025070209", "%Y%m%d%H")

# 전날 날짜 계산
target_date = (current_datetime - datetime.timedelta(days=1)).strftime("%Y%m%d")
hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
target_stat_hours = [f"{target_date}{hour}" for hour in hours]

# 조회된 데이터를 담을 전역 변수 초기화

traffic_data_00 = []
traffic_data_01 = []
traffic_data_02 = []
traffic_data_03 = []
traffic_data_04 = []
traffic_data_05 = []
traffic_data_06 = []
traffic_data_07 = []
traffic_data_08 = []
traffic_data_09 = []
traffic_data_10 = []
traffic_data_11 = []
traffic_data_12 = []
traffic_data_13 = []
traffic_data_14 = []
traffic_data_15 = []
traffic_data_16 = []
traffic_data_17 = []
traffic_data_18 = []
traffic_data_19 = []
traffic_data_20 = []
traffic_data_21 = []
traffic_data_22 = []
traffic_data_23 = []









# ============================================================================ [ 전역상태설정 - DB접속정보 & 네트워크 ]

class Config:
    
    def __init__(self):
        load_dotenv(dotenv_path="C:/Digital Twin Simulation Program/.env")
        self.env = os.getenv("FLASK_ENV", "production")
        
        self.db_config = {
            "test": {
                "driver": "Tibero 5 ODBC Driver",
                "server": os.getenv("ENZERO_SERVER"),
                "port": os.getenv("ENZERO_PORT"),
                "db": os.getenv("ENZERO_DB"),
                "uid": os.getenv("ENZERO_UID"),
                "pwd": os.getenv("ENZERO_PWD")
            },
            "prod": {
                "dsn": os.getenv("DSNNAME"),
                "uid": os.getenv("DBUSER"),
                "pwd": os.getenv("DBPWD")
            }
        }

        # self.visum_paths = self._load_visum_paths()

    # 강릉 visum 네트워크 경로
    
    def _load_visum_paths(self):
        base_path = r"C:\Digital Twin Simulation Network\VISUM"
        file_list = [
            "강릉시 전국 전일 최종본(2025.07).ver"
        ]
        return {
            os.path.splitext(name)[0]: os.path.join(base_path, name)
            for name in file_list
        }

# ============================================================================ [ DB 연결 - 교통량 조회 ]

class DatabaseManager:
    
    def __init__(self, config: Config):
        self.config = config
        self.conn = self._connect()
        self.cursor = self.conn.cursor() if self.conn else None
        self.columns = [
            "STAT_HOUR", "CROSS_ID", "INFRA_TYPE", "SPECIALDAY", "VPHG", "VPHG", "VS", "LOS",
            "VOL_01", "VOL_02", "VOL_03", "VOL_04", "VOL_05", "VOL_06", "VOL_07", "VOL_08", "VOL_09",
            "VOL_10", "VOL_11", "VOL_12", "VOL_13", "VOL_14", "VOL_15", "VOL_16", "VOL_17", "VOL_18",
            "VOL_19", "VOL_20", "VOL_21", "VOL_22", "VOL_23",
            "WALKER_CNT", "CRASH_CNT", "DELAY_TIME"
        ]
        self.traffic_data_by_hour = {
            "00": [],
            "01": [],
            "02": [],
            "03": [],
            "04": [],
            "05": [],
            "06": [],
            "07": [],
            "08": [],
            "09": [],
            "10": [],
            "11": [],
            "12": [],
            "13": [],
            "14": [],
            "15": [],
            "16": [],
            "17": [],
            "18": [],
            "19": [],
            "20": [],
            "21": [],
            "22": [],
            "23": []
        }

    def _connect(self):
        try:
            if self.config.env == "test":
                db = self.config.db_config["test"]
                return pyodbc.connect(
                    f"DRIVER={db['driver']};"
                    f"SERVER={db['server']};"
                    f"PORT={db['port']};"
                    f"DB={db['db']};"
                    f"UID={db['uid']};"
                    f"PWD={db['pwd']};"
                )
            else:
                db = self.config.db_config["prod"]
                return pyodbc.connect(
                    f"DSN={db['dsn']};"
                    f"UID={db['uid']};"
                    f"PWD={db['pwd']};"
                )
        except Exception as e:
            print("⛔ DB 연결 실패:", e)
            return None

    def fetch_peak_traffic_data(self):
        try:
            def convert_row_to_dict(row, columns):
                return {
                    col: float(val) if isinstance(val, Decimal) else val
                    for col, val in zip(columns, row)
                }

            if not self.cursor:
                print(">>> DB 커서가 유효하지 않습니다.")
                return

            formatted_hours = "', '".join(target_stat_hours)
            query = f"""
                SELECT *
                FROM TOMMS.STAT_HOUR_CROSS
                WHERE STAT_HOUR IN ('{formatted_hours}')
                AND INFRA_TYPE = 'SMT'
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            data_dicts = [convert_row_to_dict(row, self.columns) for row in rows]

            # 딕셔너리 기반 저장
            for row in data_dicts:
                stat_hour = row["STAT_HOUR"]
                suffix = stat_hour[-2:]
                if suffix in self.traffic_data_by_hour:
                    self.traffic_data_by_hour[suffix].append(row)

            print(f"✅ [ 교통량 데이터 조회 완료 ] - 총 {len(data_dicts)}건")
            for hour in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
                print(f"✅ 시간대 {hour}: {len(self.traffic_data_by_hour[hour])}건")
                # ✅ 시간대 nn: nn건

        except Exception as e:
            print("⛔ 교통량 조회 중 오류:", e)











if __name__ == "__main__":
    
    # ------------------------------------------------------------ 로그 폴더 및 파일명 지정
    
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    start_time = datetime.datetime.now()
    log_filename = start_time.strftime("%Y%m%d_%H%M%S_simulation.log")
    log_path = os.path.join(log_folder, log_filename)

    # ------------------------------------------------------------ 로그파일로 출력 리디렉션
    
    with open(log_path, "w", encoding="utf-8") as log_file, redirect_stdout(log_file):
        print("🟢 시뮬레이션 시작")
        print(f"▶️ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # ------------------------------------------------------------ 실제 시뮬레이션 실행 코드
        
        config = Config()
        db = DatabaseManager(config)