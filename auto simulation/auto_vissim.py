import sys
import os
import re
import time
import random
import shutil
import pyodbc
import fnmatch
import datetime
import pythoncom
import pywintypes
import pandas as pd
import win32com.client as com
from win32com.client import gencache

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
current_datetime = datetime.datetime.strptime("2025070202", "%Y%m%d%H")

# 전날 날짜 계산
target_date = (current_datetime - datetime.timedelta(days=1)).strftime("%Y%m%d")
peak_hours = ['08', '11', '14', '17']
target_stat_hours = [f"{target_date}{hour}" for hour in peak_hours]  # ["2025070108", "2025070111", "2025070114", "2025070117"]

# 조회된 데이터를 담을 전역 변수 초기화
traffic_data_08 = []
traffic_data_11 = []
traffic_data_14 = []
traffic_data_17 = []









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

        self.vissim_paths = self._load_vissim_paths()

    # 경포, 송정동, 도심, 교동 네트워크 경로
    
    def _load_vissim_paths(self):
        base_path = r"C:\Digital Twin Simulation Network\VISSIM"
        file_list = [
            "gyeongpo.inpx",
            "songjung.inpx",
            "downtown.inpx",
            "gyodong.inpx"
        ]
        return {
            os.path.splitext(name)[0]: os.path.join(base_path, name)
            for name in file_list
        }

# ============================================================================ [ DB 연결 - 교통량 조회 ]

FIFTEEN_MINUTES = ["00", "15", "30", "45"]
INTERVAL_LABEL = {  # 키(mm) -> 구간 의미
    "00": "45~00분",  # hh00 은 직전 45~정각
    "15": "00~15분",
    "30": "15~30분",
    "45": "30~45분",
}

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.conn = self._connect()
        self.cursor = self.conn.cursor() if self.conn else None

        # 시간대별 집계(HOUR)
        self.traffic_data_by_hour = {"08": [], "11": [], "14": [], "17": []}

        # 15분 집계(HH -> MM -> list)
        self.traffic_data_by_15min = {
            "08": {"00": [], "15": [], "30": [], "45": []},
            "11": {"00": [], "15": [], "30": [], "45": []},
            "14": {"00": [], "15": [], "30": [], "45": []},
            "17": {"00": [], "15": [], "30": [], "45": []},
        }

    def _connect(self):
        try:
            if self.config.env == "test":
                db = self.config.db_config["test"]
                print(">>>>> ✅ 엔제로 데이터베이스 연결")
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
                print(">>>>> ✅ 강릉 데이터베이스 연결")
                return pyodbc.connect(
                    f"DSN={db['dsn']};"
                    f"UID={db['uid']};"
                    f"PWD={db['pwd']};"
                )
        except Exception as e:
            print("⛔ DB 연결 실패:", e)
            return None

    # cursor.description 기반 안전 매핑
    def _rows_to_dicts(self, rows):
        cols = [d[0] for d in self.cursor.description]
        out = []
        for r in rows:
            d = {}
            for c, v in zip(cols, r):
                d[c] = float(v) if isinstance(v, Decimal) else v
            out.append(d)
        return out

    @staticmethod
    def _expand_to_15min_timestamps(target_stat_hours):
        """
        input:  ['YYYYMMDD08','YYYYMMDD11',...]
        output: ['YYYYMMDD0800','YYYYMMDD0815','YYYYMMDD0830','YYYYMMDD0845', ...]
        """
        return [f"{hh}{mm}" for hh in target_stat_hours for mm in FIFTEEN_MINUTES]

    def fetch_peak_traffic_data(self, target_stat_hours):
        """
        target_stat_hours: ['YYYYMMDD08','YYYYMMDD11','YYYYMMDD14','YYYYMMDD17']
        - STAT_HOUR_CROSS  : STAT_HOUR, CROSS_ID, VOL, VOL_01..VOL_24
        - STAT_15MIN_CROSS : STAT_15MIN, CROSS_ID, VOL, VOL_01..VOL_24
        적재 대상:
        - self.traffic_data_by_hour[HH]     # HH in {"08","11","14","17"}
        - self.traffic_data_by_15min[HH][MM]  # MM in {"00","15","30","45"}
        """
        if not self.cursor:
            print(">>> DB 커서가 유효하지 않습니다.")
            return

        # 공통 컬럼 세트
        vol_cols = ["VOL"] + [f"VOL_{i:02d}" for i in range(1, 25)]

        try:
            # ---------------- (A) 시간대(HH) 데이터 ----------------
            hour_cols = ["STAT_HOUR", "CROSS_ID"] + vol_cols
            hour_col_str = ", ".join(hour_cols)

            placeholders = ", ".join(["?"] * len(target_stat_hours))
            sql_hour = f"""
                SELECT {hour_col_str}
                FROM TOMMS.STAT_HOUR_CROSS
                WHERE STAT_HOUR IN ({placeholders})
                AND INFRA_TYPE = 'SMT'
            """
            self.cursor.execute(sql_hour, target_stat_hours)
            hour_rows = self.cursor.fetchall()
            hour_dicts = self._rows_to_dicts(hour_rows)

            # 적재
            for row in hour_dicts:
                # 필수 키 존재 확인
                if "STAT_HOUR" not in row or "CROSS_ID" not in row:
                    continue
                hh = str(row["STAT_HOUR"])[-2:]  # '08','11','14','17'
                if hh in self.traffic_data_by_hour:
                    self.traffic_data_by_hour[hh].append(row)

            # ---------------- (B) 15분(HHMM) 데이터 ----------------
            min_cols = ["STAT_15MIN", "CROSS_ID"] + vol_cols
            min_col_str = ", ".join(min_cols)

            target_stat_15mins = [f"{hh}{mm}" for hh in target_stat_hours for mm in FIFTEEN_MINUTES]
            placeholders_15 = ", ".join(["?"] * len(target_stat_15mins))
            sql_15 = f"""
                SELECT {min_col_str}
                FROM TOMMS.STAT_15MIN_CROSS
                WHERE STAT_15MIN IN ({placeholders_15})
                AND INFRA_TYPE = 'SMT'
            """
            self.cursor.execute(sql_15, target_stat_15mins)
            min_rows = self.cursor.fetchall()
            min_dicts = self._rows_to_dicts(min_rows)

            for row in min_dicts:
                if "STAT_15MIN" not in row or "CROSS_ID" not in row:
                    continue
                ts = str(row["STAT_15MIN"])   # 'YYYYMMDDHHMM'
                hh, mm = ts[8:10], ts[10:12]  # HH, MM
                if hh in self.traffic_data_by_15min and mm in self.traffic_data_by_15min[hh]:
                    self.traffic_data_by_15min[hh][mm].append(row)

            # ---------------- (C) 검증 & 로그 ----------------
            print(f"✅ [시간대 데이터] 총 {len(hour_dicts)}건")
            for hh in ["08", "11", "14", "17"]:
                print(f"       {hh}시: {len(self.traffic_data_by_hour[hh])}건")

            print(f"✅ [15분 데이터] 총 {len(min_dicts)}건")
            for hh in ["08", "11", "14", "17"]:
                counts = {mm: len(self.traffic_data_by_15min[hh][mm]) for mm in FIFTEEN_MINUTES}
                total = sum(counts.values())
                print(f"       {hh}시 총 {total}건")
                print(f"            - {INTERVAL_LABEL['15']} (키=15): {counts['15']}건")
                print(f"            - {INTERVAL_LABEL['30']} (키=30): {counts['30']}건")
                print(f"            - {INTERVAL_LABEL['45']} (키=45): {counts['45']}건")
                print(f"            - {INTERVAL_LABEL['00']} (키=00): {counts['00']}건")

        except Exception as e:
            print("⛔ 교통량 조회 중 오류:", e)

# ============================================================================ [ DB 연결 - 교차로 방향별 movement 조회 ]

class NodeDirectionManager:

    def __init__(self, config: Config):
        self.config = config
        self.conn = self._connect()
        self.cursor = self.conn.cursor() if self.conn else None

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

    def fetch_node_dir_info(self):
        if not self.cursor:
            print("⛔ DB 커서가 유효하지 않습니다.")
            return pd.DataFrame()

        try:
            query = """
                SELECT
                    NI.CROSS_ID,
                    DI.NODE_ID,
                    DI.APPR_ID,
                    DI.MOVEMENT,
                    DI.DIRECTION
                FROM TOMMS.TFA_NODE_DIR_INFO DI
                JOIN TOMMS.TFA_NODE_INFO     NI
                ON NI.NODE_ID = DI.NODE_ID
                -- 필요 시 정렬
                ORDER BY DI.NODE_ID, DI.APPR_ID, DI.DIRECTION
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            cleaned_rows = []
            for row in rows:
                out = []
                for val in row:
                    # Tibero/pyodbc Decimal → int 변환 (정수 컬럼만)
                    if isinstance(val, Decimal):
                        out.append(int(val))
                    else:
                        out.append(val)
                cleaned_rows.append(tuple(out))

            df = pd.DataFrame(
                cleaned_rows,
                columns=["CROSS_ID", "NODE_ID", "APPR_ID", "MOVEMENT", "DIRECTION"]
            )

            print(f"✅ NODE_DIR_INFO 조회 완료 - {len(df)}건")
            return df

        except Exception as e:
            print("⛔ NODE_DIR_INFO 조회 실패:", e)
            return pd.DataFrame()

# ============================================================================ [ DB 연결 - 구간 정보 조회 ]

class VTTMInfoManager:

    def __init__(self, config: Config):
        self.config = config
        self.conn = self._connect()
        self.cursor = self.conn.cursor() if self.conn else None

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

    def fetch_vttm_info(self):
        if not self.cursor:
            print("⛔ DB 커서가 유효하지 않습니다.")
            return pd.DataFrame()

        try:
            query = """
                SELECT VTTM_ID, FROM_NODE_NAME, TO_NODE_NAME
                FROM TOMMS.TFA_VTTM_INFO
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            cleaned_rows = []
            for row in rows:
                cleaned_row = []
                for val in row:
                    if isinstance(val, Decimal):
                        cleaned_row.append(int(val))
                    else:
                        cleaned_row.append(val)
                cleaned_rows.append(tuple(cleaned_row))

            df = pd.DataFrame(cleaned_rows, columns=["VTTM_ID", "FROM_NODE_NAME", "TO_NODE_NAME"])
            print(f"✅ VTTM_INFO 조회 완료 - {len(df)}건")
            return df

        except Exception as e:
            print("⛔ VTTM_INFO 조회 실패:", e)
            return pd.DataFrame()









# ============================================================================ [ 구간 결과값 DB INSERT ]

def insert_vttm_results_to_db(df_vttm, db_manager):
    
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    insert_query = """
        INSERT INTO TOMMS.TFA_VTTM_HOUR_RESULT (
            STAT_HOUR, VTTM_ID, DISTANCE, VEHS, TRAVEL_TIME
        ) VALUES (?, ?, ?, ?, ?)
    """

    # NaN을 None으로 대체, 타입 형변환
    def clean_value(val, target_type):
        if pd.isna(val):
            return None
        try:
            if target_type == "int":
                return int(val)
            elif target_type == "float":
                return float(val)
            elif target_type == "str":
                return str(val)
        except:
            return None

    insert_data = []
    for _, row in df_vttm.iterrows():
        insert_data.append((
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("VTTM_ID"), "str"),
            clean_value(row.get("DISTANCE"), "float"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("TRAVEL_TIME"), "float")
        ))

    try:
        db_manager.cursor.executemany(insert_query, insert_data)
        db_manager.conn.commit()
        print(f"✅ VTTM_RESULT에 {len(insert_data)}건 삽입 완료")
    except Exception as e:
        print("⛔ DB 삽입 중 오류:", e)
        db_manager.conn.rollback()

# ============================================================================ [ 교차로 결과값 DB INSERT ]

def insert_node_results_to_db(df_node: pd.DataFrame, db_manager):
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    # --- 1) 클린업: 타입/공백 정리 ---
    df = df_node.copy()
    for c in ["STAT_HOUR", "TIMEINT", "NODE_ID"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 필수 키 누락 행 제거
    df = df[df["NODE_ID"].notna() & (df["NODE_ID"] != "")]

    # 수치 컬럼 정리
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    if "QLEN"  in df.columns: df["QLEN"]  = df["QLEN"].map(to_float)
    if "DELAY" in df.columns: df["DELAY"] = df["DELAY"].map(to_float)
    if "STOPS" in df.columns: df["STOPS"] = df["STOPS"].map(to_float)
    if "VEHS"  in df.columns: df["VEHS"]  = df["VEHS"].map(to_int)

    # --- 2) 부모키 존재할 때만 INSERT (FK 안전) ---
    insert_sql = """
        INSERT INTO TOMMS.TFA_NODE_15MIN_RESULT
            (STAT_HOUR, TIMEINT, NODE_ID, QLEN, VEHS, DELAY, STOPS)
        SELECT ?, ?, ?, ?, ?, ?, ?
        FROM DUAL
        WHERE EXISTS (
            SELECT 1 FROM TOMMS.TFA_NODE_INFO p
            WHERE p.NODE_ID = ?
        )
    """

    params = []
    for _, r in df.iterrows():
        params.append((
            r.get("STAT_HOUR"),
            r.get("TIMEINT"),
            r.get("NODE_ID"),
            r.get("QLEN"),
            r.get("VEHS"),
            r.get("DELAY"),
            r.get("STOPS"),
            r.get("NODE_ID"),  # EXISTS 검증용
        ))

    try:
        db_manager.cursor.fast_executemany = True  # pyodbc 성능 옵션
        db_manager.cursor.executemany(insert_sql, params)
        db_manager.conn.commit()
        print(f"✅ NODE_RESULT 삽입 시도 {len(params)}건 완료 (부모키 있는 행만 실제 삽입)")
    except Exception as e:
        print("⛔ NODE_RESULT 삽입 오류:", e)
        db_manager.conn.rollback()

# ============================================================================ [ 교차로 방향별 결과값 DB INSERT ]

def insert_node_dir_results_to_db(df_dir_node: pd.DataFrame, db_manager):
    
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    insert_query = """
        INSERT INTO TOMMS.TFA_NODE_DIR_15MIN_RESULT (
            STAT_HOUR, TIMEINT, NODE_ID, APPR_ID, DIRECTION, QLEN, VEHS, DELAY, STOPS
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def clean_value(val, target_type):
        if pd.isna(val):
            return None
        try:
            if target_type == "int":
                return int(val)
            elif target_type == "float":
                return float(val)
            elif target_type == "str":
                return str(val)
        except:
            return None

    insert_data = []
    for _, row in df_dir_node.iterrows():
        insert_data.append((
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("TIMEINT"), "str"),
            clean_value(row.get("NODE_ID"), "str"),
            clean_value(row.get("APPR_ID"), "int"),
            clean_value(row.get("DIRECTION"), "int"),
            clean_value(row.get("QLEN"), "float"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("DELAY"), "float"),
            clean_value(row.get("STOPS"), "float")
        ))

    try:
        db_manager.cursor.executemany(insert_query, insert_data)
        db_manager.conn.commit()
        print(f"✅ NODE_DIR_RESULT에 {len(insert_data)}건 삽입 완료")
    except Exception as e:
        print("⛔ NODE_DIR_RESULT 삽입 오류:", e)
        db_manager.conn.rollback()

# ============================================================================ [ Data Collection 결과값 DB INSERT - 통행시간 즉석 계산 ]

def insert_dc_to_db(dc: pd.DataFrame, db_manager):
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    insert_query = """
        INSERT INTO TOMMS.TFA_DC_HOUR_RESULT (
            STAT_HOUR, DC_ID, DISTANCE, VEHS, SPEED, TRAVEL_TIME
        ) VALUES (?, ?, ?, ?, ?, ?)
    """

    def clean_value(val, target_type):
        if pd.isna(val):
            return None
        try:
            if target_type == "int":
                return int(val)
            elif target_type == "float":
                return float(val)
            elif target_type == "str":
                return str(val)
        except:
            return None

    def compute_travel_time_seconds(row) -> float:
        """
        TRAVEL_TIME[s] = DISTANCE(m) * 3.6 / SPEED(km/h)
        SPEED가 0이거나 결측이면 0.0 반환
        """
        # 원래 값이 있으면 그대로 반영
        tt = row.get("TRAVEL_TIME")
        if pd.notna(tt):
            try:
                return round(float(tt), 1)
            except:
                pass

        d = row.get("DISTANCE")
        v = row.get("SPEED")
        try:
            d = float(d) if pd.notna(d) else None
            v = float(v) if pd.notna(v) else None
            if d is None or v is None or v <= 0:
                return 0.0   # ← SPEED=0, 결측 시 0.0 강제
            return round(d * 3.6 / v, 1)
        except:
            return 0.0

    insert_data = []
    for _, row in dc.iterrows():
        insert_data.append((
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("DC_ID"), "int"),
            clean_value(row.get("DISTANCE"), "float"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("SPEED"), "float"),
            compute_travel_time_seconds(row)
        ))

    try:
        db_manager.cursor.executemany(insert_query, insert_data)
        db_manager.conn.commit()
        print(f"✅ DC_RESULT에 {len(insert_data)}건 삽입 완료")
    except Exception as e:
        print("⛔ DC_RESULT 삽입 오류:", e)
        db_manager.conn.rollback()

# ============================================================================ [ Network Performance 결과값 DB INSERT ]

def insert_np_to_db(np: pd.DataFrame, db_manager):
    
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return
    
    insert_query = """
            INSERT INTO TOMMS.TFA_DISTRICT_HOUR_RESULT (
            DISTRICT_ID, STAT_HOUR, VEHS, COST
            ) VALUES (?, ?, ?, ?)
    """
    
    def clean_value(val, target_type):
        if pd.isna(val):
            return None
        try:
            if target_type == "int":
                return int(val)
            elif target_type == "float":
                return float(val)
            elif target_type == "str":
                return str(val)
        except:
            return None
    
    insert_data = []
    for _, row in np.iterrows():
        insert_data.append((
            clean_value(row.get("DISTRICT_ID"), "int"),
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("COST"), "float")
        ))

    try:
        db_manager.cursor.executemany(insert_query, insert_data)
        db_manager.conn.commit()
        print(f"✅ NP_RESULT에 {len(insert_data)}건 삽입 완료")
    except Exception as e:
        print("⛔ NP_RESULT 삽입 오류:", e)
        db_manager.conn.rollback()






# ============================================================================ [ 시뮬레이션 컨트롤러 ]

class VissimSimulationManager:
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.vissim = None
        self.paths = config.vissim_paths
        self.db = db_manager
        self._com_initialized = False
    
    # --- VISSIM COM 객체를 "성공할 때까지" 생성하는 유틸
    
    def _init_com(self):
        """STA로 COM 초기화 (여러 번 호출해도 안전하도록 가드)"""
        if not self._com_initialized:
            # COINIT_APARTMENTTHREADED = STA
            pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
            self._com_initialized = True
            print("🔧 COM initialized (STA)")

    def _uninit_com(self):
        """COM 해제 (참조 카운트 맞추기)"""
        if self._com_initialized:
            pythoncom.CoUninitialize()
            self._com_initialized = False
            print("🧹 COM uninitialized")
    
    def _ensure_vissim(self,
                    prog_ids=("Vissim.Vissim.22",),
                    max_attempts=8,
                    base_delay=1.5,
                    hard_timeout_sec=90) -> bool:
        self._init_com()
        start = time.time()
        last_err = None

        for attempt in range(1, max_attempts + 1):
            if time.time() - start > hard_timeout_sec:
                print(f"⛔ 하드 타임아웃 초과({hard_timeout_sec}s)")
                break

            for prog in prog_ids:
                try:
                    # 1) Dispatch
                    self.vissim = com.Dispatch(prog)
                    print(f"🔵 VISSIM COM 생성 성공: {prog} (attempt={attempt})")
                    return True
                except pywintypes.com_error as e1:
                    last_err = e1
                    # 2) EnsureDispatch 폴백
                    try:
                        self.vissim = gencache.EnsureDispatch(prog)
                        print(f"🔵 VISSIM COM 생성 성공(EnsureDispatch): {prog} (attempt={attempt})")
                        return True
                    except Exception as e2:
                        last_err = e2

            sleep_s = min(base_delay * (2 ** (attempt - 1)), 10.0) + random.uniform(0.0, 0.5)
            print(f"🔁 재시도 대기: {sleep_s:.1f}s (attempt={attempt}/{max_attempts})")
            time.sleep(sleep_s)

        print(f"⛔ 최종 실패: {repr(last_err)}")
        self.vissim = None
        return False

    # ============================================================================ [ 연계 - 실행 - 추출 - 저장 - 종료 ]

    def run_full_simulation(self, area):
        global target_stat_hours

        print(f"🔵 vissim 시뮬레이션에서 건네받은 분석대상 일시 : {target_date}")
        print(f"🔵 vissim 시뮬레이션에서 건네받은 분석대상 지구 : {area}")
        path = self.paths.get(area)

        if not path or not os.path.isfile(path):
            print(f"⛔ [ 경고 ] {area} 파일 없음: {path}")
            return

        # ✅ 반드시 생성될 때까지 시도
        if not self._ensure_vissim(prog_ids=("Vissim.Vissim.22",), max_attempts=8, base_delay=1.5, hard_timeout_sec=90):
            print("⛔ VISSIM 객체를 생성하지 못해 시뮬레이션 스킵")
            self._uninit_com()
            return

        try:
            # ------------------------------------------------------------ 반복된 시뮬레이션 루프
            for idx, (hour_key, traffic_hour_list) in enumerate(self.db.traffic_data_by_hour.items()):
                # hour_key 예: "08","11","14","17"
                try:
                    idx = peak_hours.index(hour_key)
                    full_stat_hour = target_stat_hours[idx]  # "YYYYMMDDHH"
                except ValueError:
                    print(f"⛔ [ 오류 ] 시간대 {hour_key}는 peak_hours에 없습니다.")
                    continue

                # 15분 데이터(없으면 빈 구조로 대체)
                traffic_15min_list = self.db.traffic_data_by_15min.get(
                    hour_key,
                    {"00": [], "15": [], "30": [], "45": []}
                )

                # 로그(요약)
                hh_total = len(traffic_hour_list)
                mm_counts = {mm: len(traffic_15min_list.get(mm, [])) for mm in ["00", "15", "30", "45"]}
                mm_total = sum(mm_counts.values())
                print(f"🔵 [ {area} ] ( {full_stat_hour} ) 시뮬레이션 시작 ===")
                print(f"    ├─ 시간대 데이터: {hh_total}건")
                print(f"    └─ 15분 데이터: 총 {mm_total}건 / 00:{mm_counts['00']} 15:{mm_counts['15']} 30:{mm_counts['30']} 45:{mm_counts['45']}")

                # [1] 이전 결과 삭제
                self.cleanup_att_files(area)

                # [2] 네트워크 파일 다시 로드 (최대 3회 소프트 재시도)
                for load_try in range(1, 4):
                    try:
                        self.vissim.LoadNet(path, False)
                        print(f"🔁 [ 네트워크 재로드 완료 ] {area} → {path} (try={load_try})")
                        break
                    except pywintypes.com_error as e:
                        print(f"⚠️ 재로드 실패(try={load_try}): {repr(e)}")
                        time.sleep(1.0 * load_try)
                else:
                    print(f"⛔ [ 오류 ] 네트워크 재로드 반복 실패: {path}")
                    continue

                # [3] 연계 → 실행 → 추출
                # ✅ 변경: 시간대/15분 데이터를 함께 전달
                self.apply_traffic_data(traffic_hour_list, traffic_15min_list)
                self.run_simulation()
                df_node, df_dir_node, df_vttm, dc, np = self.extract_results(stat_hour=full_stat_hour, area_name=area)

                # [5] DB 저장
                self.save_results((df_dir_node, df_node, df_vttm, dc, np), area, hour_key)

                # [6] 결과 파일 삭제
                self.cleanup_att_files(area)

        finally:
            # ------------------------------------------------------------ 종료
            self.close_simulation()
            self._uninit_com()  # ✅ COM 해제

    # ============================================================================ [ 연계 - vehicle input / static route ]

    def apply_traffic_data(self, traffic_hour_list, traffic_15min_list):
        """
        traffic_hour_list: 시간대(HH) 데이터 list[dict]
        traffic_15min_list: {"00": [...], "15": [...], "30": [...], "45": [...]}
            - "15" 리스트의 각 row는 '00~15' 구간을 의미
            - "30" → 15~30, "45" → 30~45, "00" → 45~00
        """

        # -----------------------------
        # 0) 사전 인덱싱 (성능 & 단순화)
        # -----------------------------
        def _row_to_volume_map(row):
            # VOL_XX → int 로만 구성 (None 제외)
            return {
                key.replace("VOL_", ""): int(val)
                for key, val in row.items()
                if key.startswith("VOL_") and val is not None
            }

        # 시간대: (cross_id, vol_key) → volume
        hour_map = {}
        for row in traffic_hour_list:
            cross_id = str(row.get("CROSS_ID"))
            vol_map = _row_to_volume_map(row)
            for vol_key, vol in vol_map.items():
                hour_map[(cross_id, vol_key)] = vol

        # 15분: mm별 (cross_id, vol_key) → volume
        mm_maps = {"00": {}, "15": {}, "30": {}, "45": {}}
        for mm in ["00", "15", "30", "45"]:
            rows = traffic_15min_list.get(mm, [])
            m = mm_maps[mm]
            for row in rows:
                cross_id = str(row.get("CROSS_ID"))
                vol_map = _row_to_volume_map(row)
                for vol_key, vol in vol_map.items():
                    m[(cross_id, vol_key)] = vol

        total_vi = 0
        total_route = 0

        print(f"🔵 [ 교통량 입력 시작 ] VI={len(traffic_hour_list)}건, 15분={'/'.join(f'{k}:{len(traffic_15min_list.get(k, []))}' for k in ['15','30','45','00'])}")

        # ------------------------------------------------------------ [ vehicle input 교통량 입력 ]
        vehicle_input_nos = self.vissim.Net.VehicleInputs.GetMultiAttValues('No')
        vehicle_input_node_ids = self.vissim.Net.VehicleInputs.GetMultiAttValues('Node_ID')
        vehicle_input_link_ids = self.vissim.Net.VehicleInputs.GetMultiAttValues('Link_ID')

        for no, node_id, link_id in zip(vehicle_input_nos, vehicle_input_node_ids, vehicle_input_link_ids):
            if node_id[1] is None or link_id[1] is None:
                continue

            cross_id = str(node_id[1])              # VI는 Node_ID 기준
            vol_key = f"{int(link_id[1]):02d}"      # Link_ID → "02","03"... 매칭

            vi_vol = hour_map.get((cross_id, vol_key))  # 시간대 교통량만 사용
            if vi_vol is None:
                continue

            vi = self.vissim.Net.VehicleInputs.ItemByKey(no[1])
            # 시간대 교통량으로 5 슬롯 동일 입력(원 코드 유지)
            vi.SetAttValue('Volume(1)', vi_vol)
            vi.SetAttValue('Volume(2)', vi_vol)
            vi.SetAttValue('Volume(3)', vi_vol)
            vi.SetAttValue('Volume(4)', vi_vol)
            vi.SetAttValue('Volume(5)', vi_vol)
            total_vi += 1

        # ------------------------------------------------------------ [ static route 교통량 입력 ]
        vrds = self.vissim.Net.VehicleRoutingDecisionsStatic
        num_decisions = self.vissim.Net.VehicleRoutingDecisionsStatic.Count

        # RelFlow 매핑 규칙
        # (1) = 시간대, (2)=00~15 → "15", (3)=15~30 → "30", (4)=30~45 → "45", (5)=45~00 → "00"
        mm_for_relflow = {2: "15", 3: "30", 4: "45", 5: "00"}

        for i in range(1, num_decisions + 1):
            if not vrds.ItemKeyExists(i):
                continue
            decision = vrds.ItemByKey(i)

            for route in decision.VehRoutSta.GetAll():
                sr_node_id = route.AttValue('VehRoutDec\\Node_ID')
                sr_turn_id = route.AttValue('Turn_ID')

                if sr_node_id is None or sr_turn_id is None:
                    continue

                cross_id = str(sr_node_id)
                vol_key = f"{int(sr_turn_id):02d}"

                # (1) 시간대 교통량 (RelFlow(1))
                sr_vol_hour = hour_map.get((cross_id, vol_key))
                if sr_vol_hour is None:
                    # 시간대 분기 없으면 전체 라우트 입력 스킵
                    # (필요 시 0 입력으로 유지하고 싶다면 아래 continue를 제거하고 0으로 세팅)
                    continue

                route.SetAttValue("RelFlow(1)", sr_vol_hour)

                # (2~5) 15분 교통량
                for rel_idx in [2, 3, 4, 5]:
                    mm = mm_for_relflow[rel_idx]
                    sr_vol_15m = mm_maps[mm].get((cross_id, vol_key), 0)
                    route.SetAttValue(f"RelFlow({rel_idx})", sr_vol_15m)

                total_route += 1

        print(f"✅ [ 입력 완료 ] VehicleInputs: {total_vi}개, StaticRoutes: {total_route}개")
        
    # ============================================================================ [ 실행 - simulation run ]

    def run_simulation(self):
        
        End_of_simulation = 4200
        self.vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation) # 시뮬레이션 4200초
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
        self.vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)
        self.vissim.Simulation.RunContinuous()

    # ============================================================================ [ 추출 - node results / vehicle input ]

    def extract_results(self, stat_hour: str, area_name: str):
        
        print(f"🔵 분석대상일시 [ {stat_hour} ] 분석대상지역 [ {area_name} ]") # 2025070108, 2025070111

        target_folder = r"C:\Digital Twin Simulation Network\VISSIM"
        results = {}

        # ------------------------------------------------------------ 삭제 대상 제거 (.results, .err, .lock 등)
        
        for file in os.listdir(target_folder):
            full_path = os.path.join(target_folder, file)

            if file.endswith(".err") or file.endswith(".lock"):
                try:
                    os.remove(full_path)
                    print(f"✅ [ 파일삭제 완료 ] : {file}")
                except Exception as e:
                    print(f"⛔ [ 파일삭제 실패 ]: {file} → {e}")

            elif file.endswith(".results") and os.path.isdir(full_path):
                try:
                    shutil.rmtree(full_path)
                    print(f"✅ [ 폴더삭제 완료 ] : {file}")
                except Exception as e:
                    print(f"⛔ [ 폴더삭제 실패 ]: {file} → {e}")

        # ------------------------------------------------------------ 파일 찾기

        def find_latest_index(base_name, result_type):
            pattern = re.compile(rf"{re.escape(base_name)}_{re.escape(result_type)}_(\d+)\.att")
            max_idx = 0
            for file in os.listdir(target_folder):
                m = pattern.match(file)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
            return f"{max_idx:03d}" if max_idx > 0 else None

        # ------------------------------------------------------------ 결과값 df로 할당하기

        def read_att_file(path):
            if not os.path.exists(path):
                print(f"⛔ 파일 없음: {path}")
                return pd.DataFrame()

            encodings = ["utf-8-sig", "utf-16", "cp949", "utf-8"]
            for enc in encodings:
                try:
                    with open(path, "r", encoding=enc, errors="strict") as f:
                        lines = f.read().splitlines()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                print(f"⛔ 인코딩 실패: {path}")
                return pd.DataFrame()

            # 보통 '$'가 들어간 라인이 2번 이상 존재, 두 번째가 헤더
            dollar_lines = [i for i, line in enumerate(lines) if "$" in line]
            if len(dollar_lines) < 2:
                print(f"⛔ 포맷 이상(헤더 탐지 실패): {path}")
                return pd.DataFrame()

            header_idx = dollar_lines[1]
            header_line = lines[header_idx]

            # ✅ 핵심: '$'부터 첫 ':'까지 제거 → 어떤 헤더 타입이 와도 동작
            # 예) $DATACOLLECTIONMEASUREMENTEVALUATION:SIMRUN;TIMEINT;... → SIMRUN;TIMEINT;...
            header_line = re.sub(r"^\$[^:]*:", "", header_line).strip()

            columns = [c.strip() for c in header_line.split(';') if c.strip()]
            data_lines = lines[header_idx + 1:]

            # 데이터 부분 파싱
            rows = []
            for line in data_lines:
                if not line.strip():
                    continue
                values = [v.strip() for v in line.split(';')]
                # 열 개수 보정
                if len(values) < len(columns):
                    values += [''] * (len(columns) - len(values))
                elif len(values) > len(columns):
                    values = values[:len(columns)]
                rows.append(dict(zip(columns, values)))

            df = pd.DataFrame(rows)
            return df

        # ------------------------------------------------------------ 각 구역 결과값 처리

        latest_index = find_latest_index(area_name, "Node Results")
        if not latest_index:
            print(f"⛔ {area_name}: 시뮬레이션 파일 없음")
            return {}

        # 파일 경로 정의
        node_file = os.path.join(target_folder, f"{area_name}_Node Results_{latest_index}.att")
        vttm_file = os.path.join(target_folder, f"{area_name}_Vehicle Travel Time Results_{latest_index}.att")
        dc_file = os.path.join(target_folder, f"{area_name}_Data Collection Results_{latest_index}.att")
        np_file = os.path.join(target_folder, f"{area_name}_Vehicle Network Performance Evaluation Results_{latest_index}.att")

        # 파일 읽기
        df_dir_node = read_att_file(node_file)
        df_vttm = read_att_file(vttm_file)
        dc = read_att_file(dc_file)
        np = read_att_file(np_file)

        print(f"✅ {area_name} - Node Results ({df_dir_node.shape[0]}행)")
        print(f"✅ {area_name} - Travel Time Results ({df_vttm.shape[0]}행)")
        print(f"✅ {area_name} - Data Collection Results ({dc.shape[0]}행)")
        print(f"✅ {area_name} - Vehicle Network Performance Evaluation Results ({np.shape[0]}행)")

        # 컬럼명 매핑
        district_map = {
            "gyodong": 1,
            "songjung": 2,
            "downtown": 3,
            "gyeongpo": 4
        }
        timeint_map = {
            '600-1500': '00-15',
            '1500-2400': '15-30',
            '2400-3300': '30-45',
            '3300-4200': '45-00',
        }
        node_col_map = {
            "VEHS(ALL)": "VEHS",
            "VEHDELAY(ALL)": "DELAY",
            "STOPS(ALL)": "STOPS",
            "MOVEMENT\\NODE\\NODE_ID": "NODE_ID",
            "MOVEMENT\\NODE\\SA": "SA_NO"
        }
        vttm_col_map = {
            "VEHICLETRAVELTIMEMEASUREMENT\\NAME": "VTTM_ID",
            "VEHS(ALL)": "VEHS",
            "TRAVTM(ALL)": "TRAVEL_TIME",
            "DISTTRAV(ALL)": "DISTANCE",
            "VEHICLETRAVELTIMEMEASUREMENT\\LINKS_ID": "LINK_ID",
            "VEHICLETRAVELTIMEMEASUREMENT\\UPDOWN": "UPDOWN",
            "VEHICLETRAVELTIMEMEASUREMENT\\SA": "SA_NO",
            "VEHICLETRAVELTIMEMEASUREMENT\\ROAD_NAME": "ROAD_NAME",
            "VEHICLETRAVELTIMEMEASUREMENT\\ACTIVE": "ACTIVE"
        }
        
        data_col_map = {
            "DATACOLLECTIONMEASUREMENT": "DC_ID",
            "DIST(ALL)": "DISTANCE",
            "VEHS(ALL)": "VEHS",
            "SPEEDAVGARITH(ALL)": "SPEED",
        }
        np_col_map = {
            "VEHACT(ALL)": "VEHS",
            "TRAVELCOST": "COST"
        }

        # 컬럼명 변경
        df_dir_node.rename(columns=node_col_map, inplace=True)
        df_vttm.rename(columns=vttm_col_map, inplace=True)
        dc.rename(columns=data_col_map, inplace=True)
        np.rename(columns=np_col_map, inplace=True)

        # ------------------------------------------------------------ 교차로 & 교차로 방향별 결과값 가공

        # 0) 매핑 테이블 로드
        node_dir_manager = NodeDirectionManager(config)
        df_node_dir_info = node_dir_manager.fetch_node_dir_info()  # [CROSS_ID, NODE_ID, APPR_ID, MOVEMENT, DIRECTION]

        # 1) 매핑 존재 확인
        if df_node_dir_info.empty:
            print("⛔ 방향 기준(매핑) 데이터가 없어 방향별 결과 생성 불가 → 스킵")
            df_dir_node = pd.DataFrame()
        else:
            # 2) 키/컬럼 정규화
            df_node_dir_info = df_node_dir_info.copy()
            df_node_dir_info.columns = [c.upper() for c in df_node_dir_info.columns]

            # ✅ 필요한 컬럼을 **CROSS_ID 포함**해서 보존 (이전 코드의 누락 지점)
            need_cols = ["MOVEMENT", "NODE_ID", "APPR_ID", "DIRECTION", "CROSS_ID"]
            df_node_dir_info = (
                df_node_dir_info[need_cols]
                .drop_duplicates(subset=["MOVEMENT"])  # movement별 유일 매핑 가정
            )

            # 타입 정규화 (매칭 실패 방지)
            df_node_dir_info["MOVEMENT"] = df_node_dir_info["MOVEMENT"].astype(str).str.strip()
            # NODE_ID: 10자리 문자열
            df_node_dir_info["NODE_ID"] = (
                df_node_dir_info["NODE_ID"].astype(str).str.strip().str.zfill(10)
            )
            # CROSS_ID: 정수 (NULL 있으면 NaN→drop 또는 0 처리 선택)
            df_node_dir_info["CROSS_ID"] = pd.to_numeric(df_node_dir_info["CROSS_ID"], errors="coerce")

            # 원본 결과 프레임 정규화
            df_dir_node = df_dir_node.copy()
            if "MOVEMENT" not in df_dir_node.columns:
                raise ValueError("원본 df_dir_node에 MOVEMENT 컬럼이 없습니다.")
            df_dir_node["MOVEMENT"] = df_dir_node["MOVEMENT"].astype(str).str.strip()

            # 3) MOVEMENT 기준 병합  (충돌 회피 위해 suffix 사용)
            df_dir_node = df_dir_node.merge(
                df_node_dir_info,
                on="MOVEMENT",
                how="left",
                suffixes=("", "_map"),
                validate="m:1"
            )
            print("✅ MOVEMENT 기반으로 CROSS_ID, NODE_ID, APPR_ID, DIRECTION 병합 완료")

            # 4) 충돌 컬럼 정리(통합): NODE_ID/APPR_ID/DIRECTION/CROSS_ID
            #    - 원본에 값 없으면 *_map 값으로 채움
            for col in ["NODE_ID", "APPR_ID", "DIRECTION", "CROSS_ID"]:
                map_col = f"{col}_map"
                if col in df_dir_node.columns and map_col in df_dir_node.columns:
                    df_dir_node[col] = df_dir_node[col].where(df_dir_node[col].notna(), df_dir_node[map_col])
                    df_dir_node.drop(columns=[map_col], inplace=True)
                elif map_col in df_dir_node.columns:
                    df_dir_node.rename(columns={map_col: col}, inplace=True)

            # 타입 재고정 (병합 후)
            df_dir_node["NODE_ID"] = df_dir_node["NODE_ID"].astype(str).str.strip().str.zfill(10)
            df_dir_node["CROSS_ID"] = pd.to_numeric(df_dir_node["CROSS_ID"], errors="coerce")

            # 5) 필수 컬럼 존재 보장 + 매핑 실패 진단
            required_cols = {"NODE_ID", "APPR_ID", "DIRECTION", "CROSS_ID"}
            missing = required_cols - set(df_dir_node.columns)
            if missing:
                cols = ", ".join(df_dir_node.columns)
                raise ValueError(f"필수 컬럼 누락: {missing}. 현재 컬럼들: {cols}")

            null_node_rows = df_dir_node[df_dir_node["NODE_ID"].isna() | (df_dir_node["NODE_ID"].astype(str).str.len() == 0)]
            if not null_node_rows.empty:
                bad_movs = null_node_rows["MOVEMENT"].dropna().unique().tolist()
                print(f"⚠️ MOVEMENT→NODE_ID 매핑 실패 건수={len(null_node_rows)} / 예시={bad_movs[:5]}")

            null_cross_rows = df_dir_node[df_dir_node["CROSS_ID"].isna()]
            if not null_cross_rows.empty:
                bad_movs = null_cross_rows["MOVEMENT"].dropna().unique().tolist()
                print(f"⚠️ MOVEMENT→CROSS_ID 매핑 실패 건수={len(null_cross_rows)} / 예시={bad_movs[:5]}")

            # 6) 공통 가공
            df_dir_node["STAT_HOUR"] = stat_hour
            df_dir_node["TIMEINT"] = df_dir_node["TIMEINT"].map(timeint_map).fillna(df_dir_node["TIMEINT"])
            df_dir_node.drop(columns=[c for c in ["SIMRUN"] if c in df_dir_node.columns], inplace=True)

            # NODE_ID 무결성 확보
            df_dir_node = df_dir_node[
                df_dir_node["NODE_ID"].notna() &
                (df_dir_node["NODE_ID"].astype(str).str.len() > 0)
            ]

            # 7) 교차로 / 방향별 분리
            #    ✅ 기존의 "MOVEMENT → CROSS_ID rename"은 **삭제**.
            #    우리가 방금 병합한 **진짜 CROSS_ID(숫자형)** 를 사용해야 한다.
            df_node = df_dir_node[df_dir_node["MOVEMENT"].apply(lambda x: '@' not in str(x))].copy()
            df_dir_node = df_dir_node[df_dir_node["MOVEMENT"].apply(lambda x: '@' in str(x))].copy()

            # 8) 컬럼 정렬
            base_cols = ["STAT_HOUR", "TIMEINT", "NODE_ID", "SA_NO", "QLEN", "VEHS", "DELAY", "STOPS"]
            dir_extra_cols = ["APPR_ID", "DIRECTION"]
            keep_extra = ["CROSS_ID", "NODE_NAME"]

            if not df_node.empty:
                # node 집계에는 CROSS_ID/NODE_NAME이 필요 없으면 제거
                cols_node = [c for c in base_cols if c in df_node.columns]
                df_node = df_node[cols_node]

            cols_dir = [c for c in (base_cols + dir_extra_cols + keep_extra) if c in df_dir_node.columns]
            df_dir_node = df_dir_node[cols_dir]

            # 9) 방향별 재가공
            df_dir_node = df_dir_node[df_dir_node["APPR_ID"].notna() & (df_dir_node["APPR_ID"] != "")]
            for c in ["QLEN", "DELAY", "STOPS", "VEHS"]:
                if c in df_dir_node.columns:
                    df_dir_node[c] = pd.to_numeric(df_dir_node[c], errors="coerce")

            group_cols = ["STAT_HOUR", "TIMEINT", "NODE_ID", "CROSS_ID", "NODE_NAME", "SA_NO", "APPR_ID", "DIRECTION"]
            have_cols = [c for c in group_cols if c in df_dir_node.columns]
            df_dir_node = (
                df_dir_node
                .groupby(have_cols, as_index=False)
                .agg({"QLEN": "mean", "DELAY": "mean", "STOPS": "mean", "VEHS": "sum"})
            )
            for c in ["QLEN", "DELAY", "STOPS"]:
                if c in df_dir_node.columns:
                    df_dir_node[c] = df_dir_node[c].round(2)

            # 10) INSERT 직전 스키마로 슬라이싱 (CROSS_ID가 필요 없으면 제외)
            df_dir_node = df_dir_node[["STAT_HOUR","TIMEINT","NODE_ID","APPR_ID","DIRECTION","QLEN","VEHS","DELAY","STOPS"]]
        
        # ------------------------------------------------------------ 구간 결과값 가공
        
        # 구간분석에서 활성화된 구간만 남기고 나머지는 삭제
        df_vttm = df_vttm[df_vttm["ACTIVE"] == str(1)].copy()
        
        df_vttm["STAT_HOUR"] = stat_hour
        
        # 필요 없는 컬럼 제거
        df_vttm.drop(columns=[col for col in ["SIMRUN", "VEHICLETRAVELTIMEMEASUREMENT", "TIMEINT"] if col in df_vttm.columns], inplace=True)

        # # VTTM_INFO 조회 및 병합
        vttm_info_manager = VTTMInfoManager(config)
        df_vttm_info = vttm_info_manager.fetch_vttm_info()

        if not df_vttm_info.empty:
            df_vttm = df_vttm.merge(df_vttm_info, on="VTTM_ID", how="left")
            print("✅ 구간 노드 정보 병합 완료")
        else:
            print("🔵 구간 노드 정보 병합 스킵 (데이터 없음)")

        # 컬럼 정렬
        # 권역, 분석대상일자, 분석대상시간, 구간아이디, 시점교차로명, 종점교차로명, 상하행구분, 거리(m), 통행량, 시간(초), SA번호, 대로명, 활성화여부
        desired_vttm_cols = ["STAT_HOUR", "TIMEINT", "VTTM_ID", "FROM_NODE_NAME", "TO_NODE_NAME", "UPDOWN", "DISTANCE", "VEHS", "TRAVEL_TIME", "SA_NO", "ROAD_NAME", "ACTIVE"]
        df_vttm = df_vttm[[col for col in desired_vttm_cols if col in df_vttm.columns]]
        
        # ------------------------------------------------------------ Data Collection 컬럼 제거
        district_code = district_map.get(area)
        
        dc.drop(columns=[c for c in ["DATACOLLECTIONMEASUREMENTEVALUATION:SIMRUN", "TIMEINT"] if c in dc.columns], inplace=True)
        dc["STAT_HOUR"] = stat_hour
        dc["DISTRICT_ID"] = district_code
        
        # ------------------------------------------------------------ Network Performance 컬럼 제거
        # DISTRICT, STAT_HOUR, VEHS, COST
        
        np.drop(columns=[c for c in ["VEHICLENETWORKPERFORMANCEMEASUREMENTEVALUATION:SIMRUN", "TIMEINT"] if c in np.columns], inplace=True)
        np["STAT_HOUR"] = stat_hour
        np["DISTRICT_ID"] = district_code
        np.drop(columns=["SIMRUN"], errors="ignore", inplace=True)
        
        # ------------------------------------------------------------ 교차로 방향별 결과값 / 교차로 결과값 엑셀 저장
        
        output_dir = os.path.join(target_folder, "results_csv")
        os.makedirs(output_dir, exist_ok=True)

        df_node.to_csv(os.path.join(output_dir, f"{area_name}_{stat_hour}_node.csv"), index=False, encoding="utf-8-sig")
        df_dir_node.to_csv(os.path.join(output_dir, f"{area_name}_{stat_hour}_dir_node.csv"), index=False, encoding="utf-8-sig")
        df_vttm.to_csv(os.path.join(output_dir, f"{area_name}_{stat_hour}_vttm.csv"), index=False, encoding="utf-8-sig")
        dc.to_csv(os.path.join(output_dir, f"{area_name}_{stat_hour}_dc.csv"), index=False, encoding="utf-8-sig")
        np.to_csv(os.path.join(output_dir, f"{area_name}_{stat_hour}_np.csv"), index=False, encoding="utf-8-sig")

        print(f"✅ CSV 저장 완료 → {output_dir}")

        return df_node.copy(), df_dir_node.copy(), df_vttm.copy(), dc.copy(), np.copy()

    # ============================================================================ [ 저장 ]

    def save_results(self, result, area_name, hour_key):

        df_dir_node, df_node, df_vttm, dc, np = result

        insert_vttm_results_to_db(df_vttm, self.db) # 구간 결과값 DB INSERT
        insert_node_results_to_db(df_node, self.db) # 교차로 결과값 DB INSERT
        insert_node_dir_results_to_db(df_dir_node, self.db) # 교차로 방향별 결과값 DB INSERT
        insert_dc_to_db(dc, self.db) # 교차로 방향별 결과값 DB INSERT
        insert_np_to_db(np, self.db) # 교차로 방향별 결과값 DB INSERT
        
        print(f"✅ [ DB 저장 완료 ] {area_name}-{hour_key} 결과 DB 저장 또는 파일 기록")

    # ============================================================================ [ 종료 ]

    def close_simulation(self):
        print("✅ [ 시뮬레이션 종료 ]")
        self.vissim = None

    # ============================================================================ [ 결과값 파일들 전부 삭제 ]

    def cleanup_att_files(self, area):
        import fnmatch
        import os

        target_folder = r"C:\Digital Twin Simulation Network\VISSIM"
        patterns = [
            f"{area}_Node Results_*.att",
            f"{area}_Vehicle Travel Time Results_*.att",
            f"{area}_Vehicle Network Performance Evaluation Results_*.att",
            f"{area}_Data Collection Results_*.att"
        ]

        deleted = 0
        for pattern in patterns:
            for file in os.listdir(target_folder):
                if fnmatch.fnmatch(file, pattern):
                    try:
                        os.remove(os.path.join(target_folder, file))
                        print(f"🧹 [ 삭제 완료 ] {file}")
                        deleted += 1
                    except Exception as e:
                        print(f"⛔ [ 삭제 실패 ] {file} → {e}")

        if deleted == 0:
            print("⚠️ 삭제할 att 파일이 없습니다.")









# ============================================================================ [ main 실행 ]

class Tee:
    """stdout을 파일과 콘솔에 동시에 출력"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == "__main__":
    
    # ------------------------------------------------------------ 로그 폴더 및 파일명 지정
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    start_time = datetime.datetime.now()
    log_filename = start_time.strftime("%Y%m%d_%H%M%S_vissim_simulation.log")
    log_path = os.path.join(log_folder, log_filename)

    # ------------------------------------------------------------ 로그파일 + 콘솔 동시에 출력
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)  # 콘솔(stdout) + 로그파일 동시 출력

        print("🟢 시뮬레이션 시작")
        print(f"▶️ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # ------------------------------------------------------------ 실제 시뮬레이션 실행 코드
        config = Config()
        db = DatabaseManager(config)
        vissim_manager = VissimSimulationManager(config, db)
        db.fetch_peak_traffic_data(target_stat_hours)
        area_list = ["gyodong", "gyeongpo", "downtown", "songjung"]  # 1, 4, 3, 2

        for area in area_list:
            vissim_manager.run_full_simulation(area)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("=" * 60)
        print("🔴 시뮬레이션 종료")
        print(f"⏹️ 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕒 총 소요 시간: {str(duration).split('.')[0]} (HH:MM:SS)")

    print(f"✅ 로그 저장 완료 → {log_path}")