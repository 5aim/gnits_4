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

    # 아레나, 송정동, 도심, 교동 네트워크 경로
    
    def _load_vissim_paths(self):
        base_path = r"C:\Digital Twin Simulation Network\VISSIM"
        file_list = [
            "아레나.inpx",
            "송정동.inpx",
            "도심(강릉역).inpx",
            "교동지구.inpx"
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
            "08": [],
            "11": [],
            "14": [],
            "17": []
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
            for hour in ["08", "11", "14", "17"]:
                print(f"✅ 시간대 {hour}: {len(self.traffic_data_by_hour[hour])}건")
                # ✅ 시간대 08: 96건
                # ✅ 시간대 11: 96건
                # ✅ 시간대 14: 96건
                # ✅ 시간대 17: 96건

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
                SELECT CROSS_ID, NODE_NAME, APPR_ID, MOVEMENT, DIRECTION
                FROM TOMMS.NODE_DIR_INFO
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

            df = pd.DataFrame(cleaned_rows, columns=["CROSS_ID", "NODE_NAME", "APPR_ID", "MOVEMENT", "DIRECTION"])
            
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
                FROM TOMMS.VTTM_INFO
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

# ---------------------------------------------------------------------------- [ 통행비용 추가 필요 ]

def insert_vttm_results_to_db(df_vttm, db_manager):
    
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    insert_query = """
        INSERT INTO VTTM_RESULT (
            DISTRICT, STAT_HOUR, VTTM_ID,
            FROM_NODE_NAME, TO_NODE_NAME, UPDOWN,
            DISTANCE, VEHS, TRAVEL_TIME,
            SA_NO, ROAD_NAME, ACTIVE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            clean_value(row.get("DISTRICT"), "int"),
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("VTTM_ID"), "str"),
            clean_value(row.get("FROM_NODE_NAME"), "str"),
            clean_value(row.get("TO_NODE_NAME"), "str"),
            clean_value(row.get("UPDOWN"), "int"),
            clean_value(row.get("DISTANCE"), "float"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("TRAVEL_TIME"), "float"),
            clean_value(row.get("SA_NO"), "str"),
            clean_value(row.get("ROAD_NAME"), "str"),
            clean_value(row.get("ACTIVE"), "int")
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

    insert_query = """
        INSERT INTO NODE_RESULT (
            DISTRICT, STAT_HOUR, TIMEINT,
            NODE_ID, SA_NO,
            QLEN, VEHS, DELAY, STOPS
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
    for _, row in df_node.iterrows():
        insert_data.append((
            clean_value(row.get("DISTRICT"), "int"),
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("TIMEINT"), "str"),
            clean_value(row.get("NODE_ID"), "str"),
            clean_value(row.get("SA_NO"), "str"),
            clean_value(row.get("QLEN"), "float"),
            clean_value(row.get("VEHS"), "int"),
            clean_value(row.get("DELAY"), "float"),
            clean_value(row.get("STOPS"), "float")
        ))

    try:
        db_manager.cursor.executemany(insert_query, insert_data)
        db_manager.conn.commit()
        print(f"✅ NODE_RESULT에 {len(insert_data)}건 삽입 완료")
    except Exception as e:
        print("⛔ NODE_RESULT 삽입 오류:", e)
        db_manager.conn.rollback()

# ============================================================================ [ 교차로 방향별 결과값 DB INSERT ]

def insert_node_dir_results_to_db(df_dir_node: pd.DataFrame, db_manager):
    
    if db_manager.cursor is None:
        print("⛔ DB 커서가 유효하지 않습니다.")
        return

    insert_query = """
        INSERT INTO NODE_DIR_RESULT (
            DISTRICT, STAT_HOUR, TIMEINT,
            NODE_ID, CROSS_ID, NODE_NAME,
            SA_NO, APPR_ID, DIRECTION,
            QLEN, VEHS, DELAY, STOPS
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            clean_value(row.get("DISTRICT"), "int"),
            clean_value(row.get("STAT_HOUR"), "str"),
            clean_value(row.get("TIMEINT"), "str"),
            clean_value(row.get("NODE_ID"), "str"),
            clean_value(row.get("CROSS_ID"), "int"),
            clean_value(row.get("NODE_NAME"), "str"),
            clean_value(row.get("SA_NO"), "str"),
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









# ============================================================================ [ 시뮬레이션 컨트롤러 ]

class VissimSimulationManager:
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.vissim = None
        self.paths = config.vissim_paths
        self.db = db_manager

    # ============================================================================ [ 연계 - 실행 - 추출 - 저장 - 종료 ]

    def run_full_simulation(self, area):
        global target_stat_hours

        district_map = {
            "교동지구": 1,
            "송정동": 2,
            "도심(강릉역)": 3,
            "아레나": 4
        }

        print(f"🔵 vissim 시뮬레이션에서 건네받은 분석대상 일시 : {target_date}")
        path = self.paths.get(area)

        if not path or not os.path.isfile(path):
            print(f"⛔ [ 경고 ] {area} 파일 없음: {path}")
            return

        # VISSIM 객체 생성 (한 번만)
        try:
            self.vissim = com.Dispatch("Vissim.Vissim.22")
            print("🔵 VISSIM COM 객체 생성")
        except pywintypes.com_error:
            print("⛔ [ 오류 ] VISSIM 객체 생성 실패")
            self.vissim = None
            return

        # ------------------------------------------------------------ 반복된 시뮬레이션 루프
        for idx, (hour_key, traffic_list) in enumerate(db.traffic_data_by_hour.items()):
            try:
                idx = peak_hours.index(hour_key)
                full_stat_hour = target_stat_hours[idx]
            except ValueError:
                print(f"⛔ [ 오류 ] 시간대 {hour_key}는 peak_hours에 없습니다.")
                continue

            print(f"🔵 [ {area} ] ( {full_stat_hour} ) 시뮬레이션 시작 ===")

            # [1] 이전 결과 삭제
            self.cleanup_att_files(area)

            # [2] 네트워크 파일 다시 로드 (상태 초기화)
            try:
                self.vissim.LoadNet(path, False)
                print(f"🔁 [ 네트워크 재로드 완료 ] {area} → {path}")
            except pywintypes.com_error:
                print(f"⛔ [ 오류 ] 네트워크 재로드 실패: {path}")
                continue

            # [3] 교통량 연계 → 시뮬레이션 실행 → 결과 추출
            self.apply_traffic_data(traffic_list)
            self.run_simulation()
            df_node, df_dir_node, df_vttm = self.extract_results(stat_hour=full_stat_hour, area_name=area)

            # [4] 지역코드 부여
            district_code = district_map.get(area)
            df_node["DISTRICT"] = district_code
            df_dir_node["DISTRICT"] = district_code
            df_vttm["DISTRICT"] = district_code

            # [5] 결과 DB 저장
            self.save_results((df_dir_node, df_node, df_vttm), area, hour_key)

            # [6] 결과 파일 삭제
            self.cleanup_att_files(area)

        # ------------------------------------------------------------ 종료
        self.close_simulation()

    # ============================================================================ [ 연계 - vehicle input / static route ]

    def apply_traffic_data(self, traffic_list):
        
        print(f"🔵 [ 교통량 입력 시작 ] 총 {len(traffic_list)}건")

        for idx, row in enumerate(traffic_list, 1):
            stat_hour = row.get("STAT_HOUR")
            cross_id = row.get("CROSS_ID")

            # VOL_xx 중 None이 아닌 값만 추출
            volume_data = {
                key.replace("VOL_", ""): int(value)
                for key, value in row.items()
                if key.startswith("VOL_") and value is not None
            }

            # ------------------------------------------------------------ [ vehicle input 교통량 입력 ]
            
            num_decisions = self.vissim.Net.VehicleRoutingDecisionsStatic.count
            vehicle_input_nos = self.vissim.Net.VehicleInputs.GetMultiAttValues('No')
            vehicle_input_node_ids = self.vissim.Net.VehicleInputs.GetMultiAttValues('Node_ID')
            vehicle_input_link_ids = self.vissim.Net.VehicleInputs.GetMultiAttValues('Link_ID')
            
            grouped_data_list = []

            for no, node_id, link_id in zip(vehicle_input_nos, vehicle_input_node_ids, vehicle_input_link_ids):
                if node_id[1] is None or link_id[1] is None:  # node_id와 link_id에 None이 있으면 제외
                    continue

                # CROSS_ID와 node_id가 일치하는지 확인
                if str(cross_id) != str(node_id[1]):
                    continue

                # VOL_xx에서 xx == link_id
                vol_key = f"{int(link_id[1]):02d}"  # 예: 4 → "04"
                vi_vol = volume_data.get(vol_key)

                if vi_vol is None:
                    continue  # 해당 방향 데이터 없음

                print(f"[ Vehicle Input ] (InputNo = {no[1]}) (NodeID = {node_id[1]}) (LinkID = {link_id[1]}) (Volume = {vi_vol})")

                # 교통량 입력
                vi = self.vissim.Net.VehicleInputs.ItemByKey(no[1])
                vi.SetAttValue('Volume(1)', vi_vol)
                vi.SetAttValue('Volume(2)', vi_vol)
                vi.SetAttValue('Volume(3)', vi_vol)
                vi.SetAttValue('Volume(4)', vi_vol)
                vi.SetAttValue('Volume(5)', vi_vol)
            
            # ------------------------------------------------------------ [ static route 교통량 입력 ]
            
            vrds = self.vissim.Net.VehicleRoutingDecisionsStatic
            num_decisions = self.vissim.Net.VehicleRoutingDecisionsStatic.Count
            
            for i in range(1, num_decisions + 1):
                if vrds.ItemKeyExists(i):
                    decision = vrds.ItemByKey(i)

                    for route in decision.VehRoutSta.GetAll():
                        sr_node_id = route.AttValue('VehRoutDec\\Node_ID')
                        sr_turn_id = route.AttValue('Turn_ID')
                        
                        # 조건: 둘 다 None이 아니어야 함
                        if sr_node_id is None or sr_turn_id is None:
                            continue

                        # CROSS_ID와 매칭
                        if str(cross_id) != str(sr_node_id):
                            continue

                        # Turn_ID에 해당하는 vol key 생성 → ex: 3 → "03"
                        vol_key = f"{int(sr_turn_id):02d}"
                        sr_vol = volume_data.get(vol_key)

                        if sr_vol is None:
                            continue  # 해당 방향에 대해 교통량 없음

                        print(f"[ Static Route ] (NodeID = {sr_node_id}) (TurnID = {sr_turn_id}) (Volume= {sr_vol})")
                        
                        route.SetAttValue("RelFlow(1)", sr_vol)
                        route.SetAttValue("RelFlow(2)", sr_vol)
                        route.SetAttValue("RelFlow(3)", sr_vol)
                        route.SetAttValue("RelFlow(4)", sr_vol)
                        route.SetAttValue("RelFlow(5)", sr_vol)

    # ============================================================================ [ 실행 - simulation run ]

    def run_simulation(self):
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
                match = pattern.match(file)
                if match:
                    idx = int(match.group(1))
                    max_idx = max(max_idx, idx)
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

            dollar_lines = [i for i, line in enumerate(lines) if "$" in line]
            if len(dollar_lines) < 2:
                print(f"⛔ 포맷 이상: {path}")
                return pd.DataFrame()

            header_idx = dollar_lines[1]
            columns = lines[header_idx].replace('$MOVEMENTEVALUATION:', '').replace('$VEHICLETRAVELTIMEMEASUREMENTEVALUATION:', '').strip().split(';')
            data_lines = lines[header_idx + 1:]

            rows = []
            for line in data_lines:
                if not line.strip():
                    continue
                values = line.strip().split(';')
                values = values[:len(columns)] + [''] * (len(columns) - len(values))
                rows.append(dict(zip(columns, values)))

            df = pd.DataFrame(rows)

            # 인코딩 깨진 열 자동 복구 시도 (한글 포함 추정 열 대상)
            for col in df.columns:
                try:
                    if df[col].str.contains("[가-힣]").any():
                        continue  # 이미 한글 정상
                    df[col] = df[col].apply(lambda x: x.encode('latin1').decode('cp949') if isinstance(x, str) else x)
                except Exception:
                    continue

            return df

        # ------------------------------------------------------------ 각 구역 결과값 처리

        latest_index = find_latest_index(area_name, "Node Results")
        if not latest_index:
            print(f"⛔ {area_name}: 시뮬레이션 파일 없음")
            return {}

        # 파일 경로 정의
        node_file = os.path.join(target_folder, f"{area_name}_Node Results_{latest_index}.att")
        vttm_file = os.path.join(target_folder, f"{area_name}_Vehicle Travel Time Results_{latest_index}.att")

        # 파일 읽기
        df_dir_node = read_att_file(node_file)
        df_vttm = read_att_file(vttm_file)

        print(f"✅ {area_name} - Node Results ({df_dir_node.shape[0]}행)")
        print(f"✅ {area_name} - Travel Time Results ({df_vttm.shape[0]}행)")

        # 컬럼명 매핑
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

        # 컬럼명 변경
        df_dir_node.rename(columns=node_col_map, inplace=True)
        df_vttm.rename(columns=vttm_col_map, inplace=True)

        # ------------------------------------------------------------ 교차로 & 교차로 방향별 결과값 가공

        # df_dir_node 정보 병합
        node_dir_manager = NodeDirectionManager(config)
        df_node_dir_info = node_dir_manager.fetch_node_dir_info()
        if not df_node_dir_info.empty:
            df_dir_node = df_dir_node.merge(df_node_dir_info, on="MOVEMENT", how="left")
            print("✅ DIRECTION, APPR_ID 병합 완료")
            
            unmatched = df_dir_node[~df_dir_node["MOVEMENT"].isin(df_node_dir_info["MOVEMENT"])]
            print("✅ 병합되지 않은 MOVEMENT 값 전체 목록:")
            print(unmatched["MOVEMENT"].unique().tolist())
        else:
            print("⛔ 방향 정보 병합 스킵 (데이터 없음)")

        # 공통 가공
        df_dir_node["STAT_HOUR"] = stat_hour
        df_dir_node["TIMEINT"] = df_dir_node["TIMEINT"].map(timeint_map).fillna(df_dir_node["TIMEINT"])
        df_dir_node["DISTRICT"] = area_name
        df_dir_node = df_dir_node[df_dir_node["NODE_ID"].notna() & (df_dir_node["NODE_ID"] != "")]
        df_dir_node.drop(columns=[col for col in ["SIMRUN"] if col in df_dir_node.columns], inplace=True)

        # 교차로 / 방향별 분리
        df_node = df_dir_node[df_dir_node["MOVEMENT"].apply(lambda x: '@' not in str(x))].copy()
        df_node.rename(columns={"MOVEMENT": "CROSS_ID"}, inplace=True)
        df_dir_node = df_dir_node[df_dir_node["MOVEMENT"].apply(lambda x: '@' in str(x))].copy()
        
        # 컬럼 정렬
        # 권역, 분석대상일자, 분석대상시간, 표준노드아이디, SA번호, 방향기준값, 대기행렬, 통행량, 지체시간(초), 정지횟수
        base_cols = ["DISTRICT", "STAT_HOUR", "TIMEINT", "NODE_ID", "CROSS_ID", "NODE_NAME", "SA_NO", "MOVEMENT", "QLEN", "VEHS", "DELAY", "STOPS"]
        # 접근로방향(시계방향값), 우직좌(1, 2, 3)
        dir_extra_cols = ["APPR_ID", "DIRECTION"]

        df_node = df_node[[col for col in base_cols if col in df_node.columns]]
        df_node.drop(columns=[col for col in ["CROSS_ID", "NODE_NAME"] if col in df_node.columns], inplace=True)
        df_dir_node = df_dir_node[[col for col in base_cols + dir_extra_cols if col in df_dir_node.columns]]
        
        # ------------------------------------------------------------ 방향별 교차로 재가공
        
        # [1] APPR_ID 없는 행 제거
        df_dir_node = df_dir_node[df_dir_node["APPR_ID"].notna() & (df_dir_node["APPR_ID"] != "")].copy()

        # [2] 숫자 컬럼을 float으로 변환 (에러 방지)
        for col in ["QLEN", "DELAY", "STOPS", "VEHS"]:
            df_dir_node[col] = pd.to_numeric(df_dir_node[col], errors='coerce')

        # [3] 그룹 기준 정의
        group_cols = ["DISTRICT", "STAT_HOUR", "TIMEINT", "NODE_ID", "CROSS_ID", "NODE_NAME", "SA_NO", "APPR_ID", "DIRECTION"]

        # [4] QLEN, DELAY, STOPS은 평균, VEHS는 합계 처리
        df_dir_node = (
            df_dir_node
            .groupby(group_cols, as_index=False)
            .agg({
                "QLEN": "mean",
                "DELAY": "mean",
                "STOPS": "mean",
                "VEHS": "sum"
            })
        )

        # [5] 평균 항목 소수점 둘째 자리로 반올림
        df_dir_node["QLEN"] = df_dir_node["QLEN"].round(2)
        df_dir_node["DELAY"] = df_dir_node["DELAY"].round(2)
        df_dir_node["STOPS"] = df_dir_node["STOPS"].round(2)

        # [6] MOVEMENT 컬럼 제거 (존재 시)
        if "MOVEMENT" in df_dir_node.columns:
            df_dir_node.drop(columns=["MOVEMENT"], inplace=True)
        
        # ------------------------------------------------------------ 구간 결과값 가공
        
        # 구간분석에서 활성화된 구간만 남기고 나머지는 삭제
        df_vttm = df_vttm[df_vttm["ACTIVE"] == str(1)].copy()
        
        df_vttm["STAT_HOUR"] = stat_hour
        df_vttm["DISTRICT"] = area_name
        
        # 필요 없는 컬럼 제거
        df_vttm.drop(columns=[col for col in ["SIMRUN", "VEHICLETRAVELTIMEMEASUREMENT", "TIMEINT"] if col in df_vttm.columns], inplace=True)

        # # VTTM_INFO 조회 및 병합
        vttm_info_manager = VTTMInfoManager(config)
        df_vttm_info = vttm_info_manager.fetch_vttm_info()

        if not df_vttm_info.empty:
            df_vttm = df_vttm.merge(df_vttm_info, on="VTTM_ID", how="left")
            print("🔵 구간 노드 정보 병합 완료")
        else:
            print("🔵 구간 노드 정보 병합 스킵 (데이터 없음)")

        # 컬럼 정렬
        # 권역, 분석대상일자, 분석대상시간, 구간아이디, 시점교차로명, 종점교차로명, 상하행구분, 거리(m), 통행량, 시간(초), SA번호, 대로명, 활성화여부
        desired_vttm_cols = ["DISTRICT", "STAT_HOUR", "TIMEINT", "VTTM_ID", "FROM_NODE_NAME", "TO_NODE_NAME", "UPDOWN", "DISTANCE", "VEHS", "TRAVEL_TIME", "SA_NO", "ROAD_NAME", "ACTIVE"]
        df_vttm = df_vttm[[col for col in desired_vttm_cols if col in df_vttm.columns]]
        
        # ------------------------------------------------------------ 교차로 방향별 결과값 / 교차로 결과값 엑셀 저장

        return df_node.copy(), df_dir_node.copy(), df_vttm.copy()

    # ============================================================================ [ 저장 ]

    def save_results(self, result, area_name, hour_key):

        df_dir_node, df_node, df_vttm = result

        insert_vttm_results_to_db(df_vttm, self.db) # 구간 결과값 DB INSERT
        insert_node_results_to_db(df_node, self.db) # 교차로 결과값 DB INSERT
        insert_node_dir_results_to_db(df_dir_node, self.db) # 교차로 방향별 결과값 DB INSERT
        
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
            f"{area}_Vehicle Travel Time Results_*.att"
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
        vissim_manager = VissimSimulationManager(config, db)

        db.fetch_peak_traffic_data()

        area_list = ["아레나", "송정동", "도심(강릉역)", "교동지구"]
        for area in area_list:
            vissim_manager.run_full_simulation(area)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("=" * 60)
        print("🔴 시뮬레이션 종료")
        print(f"⏹️ 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕒 총 소요 시간: {str(duration).split('.')[0]} (HH:MM:SS)")

    print(f"✅ 로그 저장 완료 → {log_path}")