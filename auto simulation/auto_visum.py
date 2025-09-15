import sys
import os
import re
import time
import shutil
import pyodbc
import fnmatch
import datetime
import pywintypes
import pandas as pd
import win32com.client as com
import numpy as np

from pprint import pprint
from decimal import Decimal
from dotenv import load_dotenv
from contextlib import redirect_stdout
from collections import defaultdict
from typing import Dict, List, Tuple, Optional








# ================================================== [ DPI 설정 ]

def set_dpi_awareness() -> None:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError:
        print(" ⛔ ctypes 모듈을 가져오는 데 실패했습니다. Python 환경을 확인하세요.")
    except AttributeError:
        print(" ⛔ SetProcessDpiAwareness 함수가 지원되지 않는 환경입니다. Windows 8.1 이상에서만 지원됩니다.")
    except OSError as os_error:
        print(f" ⛔ OS 관련 오류 발생: {os_error}")
    except Exception as e:
        print(f" ⛔ 알 수 없는 오류 발생: {e}")

set_dpi_awareness()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))








# ================================================== [ 공용 유틸 ]

# Decimal → int/float 변환, 나머지는 그대로.
def to_py(val):
    if isinstance(val, Decimal):
        return int(val) if val == int(val) else float(val)
    return val

# 형식 검사: YYYYMMDDHH (10자리 숫자)
def is_valid_stat_hour(s: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", s))

# VOL_00 ~ VOL_23 같은 컬럼을 이름 기준 정렬한 리스트로 반환. 반환 형태: [(컬럼명, 컬럼인덱스), ...]
def extract_vol_columns(cols: List[str]) -> List[Tuple[str, int]]:
    # 인덱스는 외부에서 매핑된 dict 로 가져오는 것이 일반적이므로, 여기선 이름만 정렬.
    vol_names = sorted([c for c in cols if re.fullmatch(r"VOL_\d{2}", c)])
    return vol_names

def to_db_py(v):
    # None/NaN 처리
    if v is None:
        return None
    # pandas의 NA류도 None으로
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # numpy 스칼라 → 파이썬 스칼라
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # Decimal → int/float
    if isinstance(v, Decimal):
        return int(v) if v == int(v) else float(v)

    # 그 외는 원본 유지(문자열 등)
    return v





# ================================================== [ 현재 일시/타깃 시간 계산 ]

def compute_target_hours(now: Optional[datetime.datetime] = None,
                        pick_hours: List[str] = None) -> Tuple[str, List[str]]:
    
    """
    기준시각(now) 기준 전날 날짜(YYYYMMDD)와, 전날의 특정 HH 리스트(YYYYMMDDHH)를 돌려줌.
    """
    
    if now is None:
        now = datetime.datetime.now()
    if pick_hours is None:
        pick_hours = ["08", "11", "14", "17"]

    target_date = (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
    target_stat_hours = [f"{target_date}{h}" for h in pick_hours]
    print(f">>>>> ✅ 데이터 조회 기준 시간 : {target_date}")
    return target_date, target_stat_hours







# ================================================== [ 환경설정 및 DB 접속정보 로딩 ]

class Config:

    def __init__(self):
        load_dotenv(dotenv_path="C:/Digital Twin Simulation Program/.env")
        raw_env = (os.getenv("FLASK_ENV", "prod") or "").lower()
        self.env = "prod" if raw_env in ("prod", "production") else "test"

        self.db_config = {
            "test": {
                "driver": "Tibero 5 ODBC Driver",
                "server": os.getenv("ENZERO_SERVER"),
                "port": os.getenv("ENZERO_PORT"),
                "db": os.getenv("ENZERO_DB"),
                "uid": os.getenv("ENZERO_UID"),
                "pwd": os.getenv("ENZERO_PWD"),
            },
            "prod": {
                "dsn": os.getenv("DSNNAME"),
                "uid": os.getenv("DBUSER"),
                "pwd": os.getenv("DBPWD"),
            },
        }








# ================================================== [ DB ]

class DatabaseManager:
    """
    Tibero 기반 시간대/일별 교통량 조회 매니저
    """
    def __init__(self, config: Config):
        self.config = config
        self.conn = self._connect()
        self.cursor = self.conn.cursor() if self.conn else None

        # 사용할 컬럼들(참고용)
        self.columns = [
            "STAT_HOUR", "CROSS_ID",
            "VOL_00","VOL_01","VOL_02","VOL_03","VOL_04","VOL_05",
            "VOL_06","VOL_07","VOL_08","VOL_09","VOL_10","VOL_11",
            "VOL_12","VOL_13","VOL_14","VOL_15","VOL_16","VOL_17",
            "VOL_18","VOL_19","VOL_20","VOL_21","VOL_22","VOL_23",
        ]

    def _connect(self):
        try:
            if self.config.env == "test":
                db = self.config.db_config["test"]
                print(">>>>> ✅ 엔제로 DB 서버 연결. 테스트 버전으로 설정합니다.")
                return pyodbc.connect(
                    f"DRIVER={db['driver']};SERVER={db['server']};PORT={db['port']};"
                    f"DB={db['db']};UID={db['uid']};PWD={db['pwd']};"
                )
            else:
                db = self.config.db_config["prod"]
                print(">>>>> ✅ 강릉시 티베로 DB 서버 연결. 배포 버전입니다.")
                return pyodbc.connect(f"DSN={db['dsn']};UID={db['uid']};PWD={db['pwd']}")
        except Exception as e:
            print("⛔ DB 연결 실패:", e)
            return None

    def _exec(self, sql: str, params: Tuple = ()) -> Tuple[List[tuple], List[str]]:
        """
        공용 쿼리 실행. rows(tuple list)와 cols(list[str]) 반환.
        """
        if not self.cursor:
            raise RuntimeError("DB 커넥션이 없습니다.")
        self.cursor.execute(sql, params)
        rows = [tuple(r) for r in self.cursor.fetchall()]
        cols = [col[0] for col in self.cursor.description]
        return rows, cols

    def fetch_and_process_data(self, target_stat_hours: List[str]) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], str]:
        """
        입력: 동일 날짜의 STAT_HOUR(YYYYMMDDHH) 리스트
        처리: 시간대별 조회 → 일별(하루) 조회
        반환: (traffic_data_by_hour, traffic_data_by_day, query_day)
        """

        # ---------- 0) 시간 파라미터 정규화 ----------
        def _clean_hour(x: str) -> str:
            return (x or "").strip().strip("'\"")

        hours: List[str] = sorted({h for h in (_clean_hour(h) for h in target_stat_hours) if is_valid_stat_hour(h)})
        print(f">>>>> ✅ target_stat_hours(clean): {hours}")
        if not hours:
            print("⛔ 유효한 STAT_HOUR가 없습니다.")
            return {f"{h:02d}": [] for h in range(24)}, {}, None  # type: ignore[return-value]

        # ---------- 1) 조회 일자 확정 ----------
        day: str = hours[0][:8]
        if not re.fullmatch(r"\d{8}", day):
            raise ValueError(f"STAT_DAY 형식 오류: {repr(day)}")
        print(f">>>>> ✅ 조회 일자: {day}")

        # ---------- 2) 시간대별 조회 ----------
        ph_hours = ", ".join(["?"] * len(hours))
        sql_hour = f"""
            SELECT *
            FROM TOMMS.STAT_HOUR_CROSS
            WHERE STAT_HOUR IN ({ph_hours})
            AND TRIM(UPPER(INFRA_TYPE)) = 'SMT'
        """
        print(f"🔎 sql_hour params: {[repr(h) for h in hours]}")
        rows_hour, cols_hour = self._exec(sql_hour, tuple(hours))
        print(f">>>>> ✅ 시간대별 조회 행수: {len(rows_hour)}")

        # 컬럼 인덱스 매핑(시간)
        col_idx_h = {c: i for i, c in enumerate(cols_hour)}
        idx_stat_hour = col_idx_h["STAT_HOUR"]
        idx_cross_id_h = col_idx_h["CROSS_ID"]

        vol_names_h = extract_vol_columns(cols_hour)
        vol_idx_pairs_h = [(name, col_idx_h[name]) for name in vol_names_h]

        traffic_data_by_hour: Dict[str, List[dict]] = {f"{h:02d}": [] for h in range(24)}
        for r in rows_hour:
            stat_hour = str(r[idx_stat_hour]).strip()
            hh = stat_hour[-2:]
            if hh in traffic_data_by_hour:
                cross_id = str(r[idx_cross_id_h]).strip()
                traffic_data_by_hour[hh].append({
                    "cross_id": cross_id,
                    "data": [
                        {"direction": name, "value": to_py(r[idx])}
                        for name, idx in vol_idx_pairs_h
                        if to_py(r[idx]) is not None
                    ]
                })
        print(">>>>> ✅ 시간대별 가공 완료.")

        # ---------- 3) 일별 조회 (검증된 리터럴, fallback 없음) ----------
        traffic_data_by_day: Dict[str, List[dict]] = {day: []}

        sql_day = f"""
            SELECT *
            FROM TOMMS.STAT_DAY_CROSS
            WHERE STAT_DAY = '{day}'
            AND TRIM(UPPER(INFRA_TYPE)) = 'SMT'
        """
        print(f"🔎 sql_day(literal): {sql_day.strip()}")
        rows_day, cols_day = self._exec(sql_day)
        print(f">>>>> ✅ 일별 조회 행수: {len(rows_day)}")

        if rows_day:
            col_idx_d = {c: i for i, c in enumerate(cols_day)}
            idx_cross_id_d = col_idx_d["CROSS_ID"]
            vol_names_d = extract_vol_columns(cols_day)
            vol_idx_pairs_d = [(name, col_idx_d[name]) for name in vol_names_d]

            for r in rows_day:
                cross_id = str(r[idx_cross_id_d]).strip()
                traffic_data_by_day[day].append({
                    "cross_id": cross_id,
                    "data": [
                        {"direction": name, "value": to_py(r[idx])}
                        for name, idx in vol_idx_pairs_d
                        if to_py(r[idx]) is not None
                    ]
                })
        else:
            # 진단만 남김
            cnt_sql = f"""
                SELECT COUNT(*) AS CNT
                FROM TOMMS.STAT_DAY_CROSS
                WHERE STAT_DAY = '{day}'
                AND TRIM(UPPER(INFRA_TYPE)) = 'SMT'
            """
            cnt_rows, _ = self._exec(cnt_sql)
            cnt = cnt_rows[0][0] if cnt_rows else 0
            print(f"⛔ 일별 0건(카운트={cnt}). fallback 미사용.")

        # ---------- 4) 요약 및 반환 ----------
        total_day = len(traffic_data_by_day.get(day, []))
        print(">>>>> ✅ 전일 교통량 가공 완료.")
        print(f"      ✅ 일별 총 {total_day}건 / 일수=1")
        for hh, lst in traffic_data_by_hour.items():
            if lst:
                print(f"      ✅ 시간대 {hh}시 : {len(lst)}건")

        return traffic_data_by_hour, traffic_data_by_day, day

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            if self.conn:
                self.conn.close()
                self.conn = None
            print(">>>>> ✅ DB 연결을 종료합니다.")
        except Exception as e:
            print("⛔ DB 종료 중 에러가 발생합니다. ", e)








# ================================================== [ VISUM ]

class VisumSimulationManager:
    
    # --------------------------------------------------------------- [ 시간대별 Active 번호 (고정 시나리오) ]
    
    HOUR_TO_PROC = {
        8: 312,
        11: 408,
        14: 505,
        17: 601,
    }

    # --------------------------------------------------------------- [ init ]

    def __init__(
        self,
        base_path: str,
        default_version_name: str,
        prev_day_proc_no: int = 22,
        csv_out_dir: str = r"C:/Digital Twin Simulation network/VISUM/result_export",
        table_map: dict = None,     # {"prev_day": "TABLE_A", "hourly": "TABLE_B"}
    ):
        self.base_path = base_path
        self.default_version_name = default_version_name
        self.prev_day_proc_no = prev_day_proc_no
        self.csv_out_dir = csv_out_dir
        os.makedirs(self.csv_out_dir, exist_ok=True)

        self.table_map = table_map or {
            "prev_day": "LINK_RESULT_DAY",
            "hourly": "LINK_RESULT_HOUR",
        }

        self.visum = None
        # last_run: {"type": "prev_day"|"hourly"|None, "hour": "00"~"23"|None, "stat_day": "YYYYMMDD"|None}
        self.last_run = {"type": None, "hour": None, "stat_day": None}











    # --------------------------------------------------------------- [ 기준일 관리 ]
    
    def set_stat_day(self, yyyymmdd: str):
        if not (isinstance(yyyymmdd, str) and len(yyyymmdd) == 8 and yyyymmdd.isdigit()):
            raise ValueError("STAT_DAY must be 'YYYYMMDD' string.")
        self.last_run["stat_day"] = yyyymmdd
        print(f">>>>> ✅ VISUM set_stat_day 함수에서 건네받은 일자는 [ {self.last_run} ] 입니다.")

    def ensure_stat_day(self, preferred_day: str | None, payload_days: list[str] | None = None) -> str:
        """
        기준일을 단 한 번만 확정:
        1) preferred_day가 유효하면 그걸 사용
        2) 없으면 payload_days에서 최신일자를 선택
        이후 self.last_run['stat_day']를 설정하고, 이후에는 절대 바꾸지 않음
        """
        if self.last_run.get("stat_day"):
            return self.last_run["stat_day"]

        cand = None
        if isinstance(preferred_day, str) and len(preferred_day) == 8 and preferred_day.isdigit():
            cand = preferred_day
        elif payload_days:
            # payload_days가 문자열 'YYYYMMDD' 리스트라고 가정
            cand = max([d for d in payload_days if isinstance(d, str) and len(d) == 8 and d.isdigit()], default=None)

        if not cand:
            raise ValueError("⛔ 기준일(STAT_DAY)을 확정할 수 없습니다. preferred_day 또는 payload_days를 제공하세요.")

        self.last_run["stat_day"] = cand
        return cand

    def _require_stat_day(self) -> str:
        sd = self.last_run.get("stat_day")
        if not (isinstance(sd, str) and len(sd) == 8 and sd.isdigit()):
            raise ValueError("⛔ STAT_DAY(yyyymmdd)가 설정되지 않았습니다. vis.set_stat_day('YYYYMMDD') 또는 ensure_stat_day()를 먼저 호출하세요.")
        return sd

    def _update_last_run(self, run_type: str, hour_label: str | None):
        sd = self.last_run.get("stat_day")  # 보존
        self.last_run = {"type": run_type, "hour": hour_label, "stat_day": sd}

    # --------------------------------------------------------------- [ VISUM LOAD & CLOSE ]
    
    def open(self, filename: str = None):
        self.visum = com.Dispatch("Visum.Visum.22")

        ver_filename = filename or self.default_version_name
        ver_path = os.path.join(self.base_path, ver_filename)
        if not os.path.isfile(ver_path):
            print(f"⛔ VISUM 네트워크 파일을 찾을 수 없습니다: {ver_path}")
            self.visum = None
            return

        self.visum.LoadVersion(ver_path)
        print(f">>>>> ✅ VISUM 네트워크 로드 완료: {ver_path}")

    def close(self):
        if self.visum:
            self.visum = None
            print(">>>>> ✅ VISUM 세션 종료")











    # --------------------------------------------------------------- [ 교통량 연계(공통) ]
    
    def insert_turn_volumes(self, data_list, verbose: bool = False) -> int:
        if not self.visum:
            print("⛔ Visum 객체가 없습니다.")
            return 0
        
        print(">>>>> ✅ 조회된 교통량 데이터의 VISUM 연계를 시작합니다.")

        turns = self.visum.Net.Turns
        gn_node_list = [int(x[1]) if x[1] is not None else 0 for x in turns.GetMultiAttValues("gn_node_id")]
        gn_dir_list  = [int(x[1]) if x[1] is not None else 0 for x in turns.GetMultiAttValues("gn_direction_id")]
        from_list    = [int(x[1]) if x[1] is not None else 0 for x in turns.GetMultiAttValues("FromNodeNo")]
        via_list     = [int(x[1]) if x[1] is not None else 0 for x in turns.GetMultiAttValues("ViaNodeNo")]
        to_list      = [int(x[1]) if x[1] is not None else 0 for x in turns.GetMultiAttValues("ToNodeNo")]

        key2nodes = {
            (gn_node_list[i], gn_dir_list[i]): (from_list[i], via_list[i], to_list[i])
            for i in range(len(gn_node_list))
            if gn_node_list[i] and gn_dir_list[i]
        }

        updates = 0
        for item in (data_list or []):
            try:
                cross_id = int(item.get("cross_id"))
            except Exception:
                if verbose:
                    print(f"⛔ cross_id 파싱 실패: {item}")
                continue

            for d in item.get("data", []):
                dir_str = d.get("direction")
                val = d.get("value")
                if val is None or not dir_str:
                    continue

                try:
                    direction_num = int(dir_str.split("_")[-1])
                except Exception:
                    if verbose:
                        print(f"⛔ direction 파싱 실패 — cross_id={cross_id}, direction={dir_str}")
                    continue

                nodes = key2nodes.get((cross_id, direction_num))
                if not nodes:
                    if verbose:
                        print(f"⛔ 연계 실패 >>> CROSS_ID={cross_id}, DIR={direction_num:02d} (Turn 미존재)")
                    continue

                f, v, t = nodes
                try:
                    turn = turns.ItemByKey(f, v, t)
                    for att in ("ABT_TOTAL1", "ABT_TOTAL2", "TOTAL_TRAFFIC_VOL"):
                        turn.SetAttValue(att, val)
                    updates += 1
                except Exception as e:
                    print(f"⛔ SetAttValue 실패 — nodes=({f},{v},{t}), err={e}")

        return updates

    # --------------------------------------------------------------- [ 시뮬레이션 실행(공통) ]
    
    def _execute_procedure(self, proc_no: int):
        ops = self.visum.Procedures.Operations
        try:
            print(f">>>>> ✅ Procedure Sequence Set Active Number : {proc_no}")
            ops.ItemByKey(proc_no).SetAttValue("Active", 1)
            self.visum.Procedures.Execute()
        finally:
            try:
                ops.ItemByKey(proc_no).SetAttValue("Active", 0)
            except Exception:
                pass

    def simulate_prev_day(self):
        self._require_stat_day()
        self._execute_procedure(self.prev_day_proc_no)
        self._update_last_run("prev_day", None)
        print(f"      ✅ 전일 시뮬레이션 완료 (Active={self.prev_day_proc_no})")

    def simulate_hour(self, hour: int):
        self._require_stat_day()
        proc = self.HOUR_TO_PROC.get(int(hour))
        if proc is None:
            print(f"⛔ {hour}시 프로시저 번호를 찾을 수 없습니다.")
            return
        self._execute_procedure(proc)
        self._update_last_run("hourly", f"{int(hour):02d}")
        print(f">>>>> ✅ {int(hour):02d}시 시뮬레이션 완료 (Active={proc})")











    # --------------------------------------------------------------- [ 링크 결과값 추출 ]
    
    def get_links_result_df(self) -> pd.DataFrame:
        """
        Visum에서 링크 결과값을 추출해 DataFrame으로 반환.
        - 추출 컬럼: LINK_ID, vc(VolCapRatioPrT(AP)), vehs(VolVehPrT(AP)), speed(TCur_PrTSys(a))
        - LINK_ID가 'A, B, C' 같이 쉼표로 합쳐진 경우 개별 행으로 분리하여 동일한 vc/vehs/speed 복제
        - LINK_ID는 VARCHAR(10)로 가공(트림 후 최대 10자, 빈문자 제외)
        - STAT_DAY/STAT_HOUR는 여기서 세팅하지 않음(각 INSERT 함수에서 처리)
        """
        if not self.visum:
            print("⛔ Visum 객체가 없습니다.")
            return pd.DataFrame(columns=["LINK_ID", "vc", "vehs", "speed"])

        run_type = (self.last_run or {}).get("type")
        if run_type not in ("prev_day", "hourly"):
            print("⛔ 실행 이력 없음 — simulate 호출 후 사용하세요.")
            return pd.DataFrame(columns=["LINK_ID", "vc", "vehs", "speed"])

        # ▶ 필요한 속성만 추출
        attrs = [
            "LINK_ID",
            "VolCapRatioPrT(AP)",   # → vc
            "VolVehPrT(AP)",        # → vehs
            "TCur_PrTSys(a)",       # → speed
        ]

        rows = self.visum.Net.Links.GetMultipleAttributes(attrs, True)

        records, truncated_ids, empty_ids = [], set(), 0

        for row in rows:
            base = dict(zip(attrs, row))
            raw_id = base.get("LINK_ID")

            if not raw_id:               # None / 빈문자 → 스킵 카운트
                empty_ids += 1
                continue

            # 1) 쉼표 기준 분리(공백 트림) → 빈 토큰 제거
            link_ids = [tok.strip() for tok in str(raw_id).split(",") if tok.strip()]

            # 2) 각 LINK_ID 별로 한 행씩 복제 (vc/vehs/speed 동일 복제)
            for lid in link_ids:
                lid_trim = lid[:10]      # VARCHAR(10) 보장
                if len(lid) > 10:
                    truncated_ids.add(lid)

                rec = {
                    "LINK_ID": lid_trim,
                    "vc":    base.get("VolCapRatioPrT(AP)"),
                    "vehs":  base.get("VolVehPrT(AP)"),
                    "speed": base.get("TCur_PrTSys(a)"),
                }
                records.append(rec)

        # 데이터프레임 구성
        df = pd.DataFrame.from_records(records, columns=["LINK_ID", "vc", "vehs", "speed"])

        if df.empty:
            print(f"      📊 DataFrame 크기: 0 행 × 4 열 (빈 LINK_ID {empty_ids}건)")
            if truncated_ids:
                ex = list(sorted(truncated_ids))[:5]
                print(f"⚠️ 10자 초과 LINK_ID {len(truncated_ids)}건(예시 최대 5개): {ex}")
            return df

        # 숫자형 보정
        df["vc"]    = pd.to_numeric(df["vc"], errors="coerce")        # float
        df["vehs"]  = pd.to_numeric(df["vehs"], errors="coerce").fillna(0).astype(int)  # int
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")      # float

        # LINK_ID 문자열 보정(공백/빈문자 제거 후 None은 드랍)
        df["LINK_ID"] = df["LINK_ID"].astype(str).str.strip()
        df = df[df["LINK_ID"] != ""]

        # LINK_ID 기준 중복 제거 (첫 등장 우선)
        before = len(df)
        df.drop_duplicates(subset=["LINK_ID"], keep="first", inplace=True)
        after = len(df)

        # 정렬
        df.sort_values(by=["LINK_ID"], inplace=True, ignore_index=True)

        print(
            f"      📊 DataFrame 크기: {len(df)} 행 × {len(df.columns)} 열"
            f" (중복 제거: {before - after}건, 빈 LINK_ID: {empty_ids}건)"
        )
        if truncated_ids:
            ex = list(sorted(truncated_ids))[:5]
            print(f"⚠️ 10자 초과 LINK_ID {len(truncated_ids)}건(예시 최대 5개): {ex}")

        return df

    # --------------------------------------------------------------- [ CSV/DB I/O ]
    
    def export_csv(self, df: pd.DataFrame, run_type: str, hour: str | None) -> str:
        if df.empty:
            print("⛔ CSV 내보낼 데이터가 없습니다.")
            return ""
        os.makedirs(self.csv_out_dir, exist_ok=True)  # 디렉터리 보장

        sd = self._require_stat_day()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = "prevday" if run_type == "prev_day" else f"h{hour}"
        fname = f"links_{sd}_{kind}_{ts}.csv"
        fpath = os.path.join(self.csv_out_dir, fname)

        # 엑셀 검토 편의 위해 na_rep 지정(선택)
        df.to_csv(fpath, index=False, encoding="utf-8-sig", na_rep="")
        print(f">>>>> ✅ 결과값을 CSV파일로 저장을 완료하였습니다. 경로: {fpath}")
        return fpath
    
    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "DISTRICT" in out.columns:
            out["DISTRICT"] = pd.to_numeric(out["DISTRICT"], errors="coerce")
        if "UPDOWN" in out.columns:
            out["UPDOWN"] = pd.to_numeric(out["UPDOWN"], errors="coerce")
        if "vc" in out.columns:
            out["vc"] = pd.to_numeric(out["vc"], errors="coerce").round(2)
        if "vehs" in out.columns:
            out["vehs"] = pd.to_numeric(out["vehs"], errors="coerce").fillna(0).astype(int)
        if "speed" in out.columns:
            out["speed"] = pd.to_numeric(out["speed"], errors="coerce").round(2)
        return out

    def _rename_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.rename(columns={"sa": "SA_NO", "vc": "VC", "vehs": "VEHS", "speed": "SPEED"}, inplace=True)
        return out

    def _coerce_str_fields(self, df: pd.DataFrame, fields=("LINK_ID","SA_NO","ROAD_NAME")) -> pd.DataFrame:
        out = df.copy()
        for c in fields:
            if c in out.columns:
                out[c] = out[c].astype(object).where(pd.notna(out[c]), None)
                out[c] = out[c].map(lambda x: str(x) if x is not None else None)
        return out









    def read_gpa_file_get_road_link(self, db_conn, gpa_file_path: str, stat_hour: str):
        """
            - TDA_ROAD_VOL_INFO.ROAD_ID 전체 조회
            - gpa_file_path/{ROAD_ID}.gpa 적용
            - Links에서 ['LINK_ID','전일_용량']만 추출 → 쉼표 분리/10자 제한/빈값 제거
            - 각 ROAD_ID마다 df 생성, df['STAT_HOUR']=stat_hour 세팅
            - 곧바로 insert_hour_road_results(df, db_conn, stat_hour, road_id) 호출 (본문은 이후 구현)
        """
        if not hasattr(self, "visum") or self.visum is None:
            print("⛔ Visum 객체가 없습니다. GPA 적용 스킵")
            return {}

        road_results: dict[str, pd.DataFrame] = {}

        try:
            cur = db_conn.cursor()
            cur.execute("SELECT ROAD_ID FROM TOMMS.TDA_ROAD_VOL_INFO")
            road_ids = [str(r[0]).strip() for r in cur.fetchall() if r and r[0] is not None]
            road_ids = sorted(set(road_ids))
            print(f"🔎 GPA 대상 ROAD_ID {len(road_ids)}건")

            missing, failed, applied = [], [], 0

            for rid in road_ids:
                gpa_file = os.path.join(gpa_file_path, f"{rid}.gpa")  # ROAD_ID가 곧 파일명
                if not os.path.isabs(gpa_file):
                    gpa_file = os.path.abspath(gpa_file)

                if not os.path.isfile(gpa_file):
                    missing.append(gpa_file)
                    continue

                try:
                    # 1) GPA 적용
                    self.visum.Net.GraphicParameters.Open(gpa_file)
                    applied += 1

                    # 2) 링크 속성 추출
                    attrs = ["LINK_ID", "전일_용량"]
                    rows = self.visum.Net.Links.GetMultipleAttributes(attrs, True)

                    # 3) 가공: LINK_ID 분리/정리
                    records, truncated_ids, empty_ids = [], set(), 0
                    for row in rows:
                        base = dict(zip(attrs, row))
                        raw_id = base.get("LINK_ID")
                        if not raw_id:
                            empty_ids += 1
                            continue

                        link_ids = [tok.strip() for tok in str(raw_id).split(",") if tok.strip()]
                        for lid in link_ids:
                            lid_trim = lid[:10]
                            if len(lid) > 10:
                                truncated_ids.add(lid)
                            records.append({
                                "ROAD_ID": rid,
                                "LINK_ID": lid_trim,
                                "전일_용량": base.get("전일_용량"),
                            })

                    df = pd.DataFrame.from_records(records, columns=["ROAD_ID", "LINK_ID", "전일_용량"])
                    if df.empty:
                        print(f"      📊 ROAD_ID={rid} → 0행 (빈 LINK_ID {empty_ids}건)")
                        if truncated_ids:
                            ex = list(sorted(truncated_ids))[:5]
                            print(f"      ⚠️ 10자 초과 LINK_ID {len(truncated_ids)}건(예시≤5): {ex}")
                        road_results[rid] = df
                        continue

                    # 숫자/문자 보정
                    df["전일_용량"] = pd.to_numeric(df["전일_용량"], errors="coerce")
                    df["LINK_ID"] = df["LINK_ID"].astype(str).str.strip()
                    df = df[df["LINK_ID"] != ""]

                    # 중복 제거/정렬
                    before = len(df)
                    df.drop_duplicates(subset=["LINK_ID"], keep="first", inplace=True)
                    after = len(df)
                    df.sort_values(by=["LINK_ID"], inplace=True, ignore_index=True)

                    # 🔵 STAT_HOUR 세팅
                    df["STAT_HOUR"] = stat_hour

                    print(f"      ✅ ROAD_ID={rid} 결과: {len(df)}행 (중복 {before - after}건, 빈 LINK_ID {empty_ids}건)")
                    if truncated_ids:
                        ex = list(sorted(truncated_ids))[:5]
                        print(f"      ⚠️ 10자 초과 LINK_ID {len(truncated_ids)}건(예시≤5): {ex}")

                    road_results[rid] = df

                    # 👉 여기서 바로 시간대 road 결과 INSERT 호출(본문은 나중 구현)
                    self.insert_hour_road_results(df, db_conn=db_conn, stat_hour=stat_hour, road_id=rid)

                except Exception as e:
                    failed.append((gpa_file, str(e)))

            print(f"🖼️ GPA 적용 완료 — 성공 {applied}건 / 미존재 {len(missing)}건 / 실패 {len(failed)}건")
            if missing:
                os.makedirs("./output", exist_ok=True)
                with open("./output/missing_gpa_files.txt", "w", encoding="utf-8") as f:
                    for p in missing:
                        f.write(p + "\n")
                print("📂 미존재 GPA 파일 목록 저장: ./output/missing_gpa_files.txt")
            if failed:
                with open("./output/failed_gpa_files.txt", "w", encoding="utf-8") as f:
                    for p, msg in failed:
                        f.write(f"{p}\t{msg}\n")
                print("📂 실패 GPA 파일 목록 저장: ./output/failed_gpa_files.txt")

            return road_results

        except Exception as ex:
            print(f"⛔ GPA 적용 처리 중 오류: {ex}")
            return {}

    # --------------------------------------------------------------- [ 전일 결과값 insert ]

    def insert_day_link_results(self, df: pd.DataFrame, db_conn, chunk_size: int = 20000) -> int:
        if df is None or df.empty or db_conn is None:
            print("⛔ 전일 INSERT: DF/DB 누락")
            return 0

        # 최종 스키마(순서 고정)
        required = ["STAT_DAY", "LINK_ID", "VC", "VEHS", "SPEED"]

        # 1) 준비: 불필요 컬럼 제거
        work = df.copy()
        for c in ("run_type", "hour"):
            if c in work.columns:
                work.drop(columns=[c], inplace=True)

        # 2) 컬럼명 통일 + STAT_DAY 세팅
        rename_map = {"vc": "VC", "vehs": "VEHS", "speed": "SPEED", "link_id": "LINK_ID"}
        work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns}, inplace=True)
        stat_day = self._require_stat_day()  # 'YYYYMMDD'
        work["STAT_DAY"] = stat_day

        # 3) 타입 보정
        if "LINK_ID" in work.columns:
            work["LINK_ID"] = (
                work["LINK_ID"].astype(str).str.strip()
                .map(lambda x: x if x != "" else None)
                .map(lambda x: x[:10] if x is not None else None)
            )
        if "VC" in work.columns:
            work["VC"] = pd.to_numeric(work["VC"], errors="coerce")
        if "VEHS" in work.columns:
            work["VEHS"] = pd.to_numeric(work["VEHS"], errors="coerce").astype("Int64")
        if "SPEED" in work.columns:
            work["SPEED"] = pd.to_numeric(work["SPEED"], errors="coerce")
            work.loc[work["SPEED"] >= 360000, "SPEED"] = 0  # 이상치 컷(옵션)

        # 4) 필수 컬럼 체크 + 정렬
        missing = [c for c in required if c not in work.columns]
        if missing:
            print(f"⛔ 필수 컬럼 누락: {missing}")
            return 0
        work = work[required]

        # 5) NULL LINK_ID 제거
        before = len(work)
        work = work[work["LINK_ID"].notna()]
        if before != len(work):
            print(f"⚠️ LINK_ID NULL {before - len(work)}행 제거")

        # 6) FK 사전검사: TDA_LINK_INFO에 존재하는 LINK_ID만 남김
        cur = db_conn.cursor()
        unique_ids = sorted(set(work["LINK_ID"].tolist()))
        valid_ids = set()
        if unique_ids:
            BATCH = 900  # placeholder 제한 대비
            for i in range(0, len(unique_ids), BATCH):
                batch = unique_ids[i:i + BATCH]
                placeholders = ", ".join(["?"] * len(batch))
                sql_chk = f"SELECT LINK_ID FROM TOMMS.TDA_LINK_INFO WHERE LINK_ID IN ({placeholders})"
                cur.execute(sql_chk, batch)
                valid_ids.update(r[0] for r in cur.fetchall())

        missing_ids = sorted(set(unique_ids) - valid_ids)
        if missing_ids:
            print(f"⚠️ FK 미존재 LINK_ID {len(missing_ids)}건 — INSERT 제외 (예시 10개): {missing_ids[:10]}")
            os.makedirs("./output", exist_ok=True)
            miss_path = f"./output/missing_day_link_ids_{stat_day}.txt"
            with open(miss_path, "w", encoding="utf-8") as f:
                for lid in missing_ids:
                    f.write(str(lid) + "\n")
            print(f"📂 미존재 LINK_ID 목록 저장: {miss_path}")

        work = work[work["LINK_ID"].isin(valid_ids)]
        if work.empty:
            print("⛔ 유효 LINK_ID가 없어 INSERT 스킵")
            return 0

        # 7) DB 바인딩 값 변환
        def _to_db_value(v):
            if v is pd.NA:
                return None
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                return None
            return v

        # 8) INSERT 준비
        sql = f"INSERT INTO TOMMS.TDA_LINK_DAY_RESULT ({', '.join(required)}) VALUES ({', '.join(['?']*len(required))})"
        cur.fast_executemany = True  # 배치 성능

        # 9) SQL 로그 저장(검증용)
        os.makedirs("./output", exist_ok=True)
        sql_log_path = f"./output/day_link_result_insert_{stat_day}.sql.txt"

        def _sql_literal(v):
            if v is None:
                return "NULL"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return str(float(v))
            s = str(v).replace("'", "''")
            return f"'{s}'"

        total = 0
        try:
            with open(sql_log_path, "w", encoding="utf-8") as f_log:
                for s in range(0, len(work), chunk_size):
                    chunk = work.iloc[s:s+chunk_size]
                    data = [tuple(_to_db_value(v) for v in row) for row in chunk.itertuples(index=False, name=None)]

                    # 로그용 실제 SQL
                    for row in data:
                        values_str = [_sql_literal(v) for v in row]
                        f_log.write(
                            f"INSERT INTO TOMMS.TDA_LINK_DAY_RESULT ({', '.join(required)}) VALUES ({', '.join(values_str)});\n"
                        )

                    cur.executemany(sql, data)
                    total += len(data)

            db_conn.commit()
            print(f"      ✅ TDA_LINK_DAY_RESULT INSERT — {total}행 (STAT_DAY={stat_day})")
            print(f"      📂 SQL 로그 저장: {sql_log_path}")

            return total

        except Exception as ex:
            db_conn.rollback()
            print(f"⛔ TDA_LINK_DAY_RESULT INSERT 오류 — 롤백: {ex}")
            print(f"📂 SQL 로그(실패 시점까지): {sql_log_path}")
            return 0
    
    # --------------------------------------------------------------- [ 전일 시뮬레이션 파이프라인 ]
    
    def run_prev_day_pipeline(self, day_payload, db_conn=None, preferred_day: str | None = None):
        # 0) 기준일 확정 (단 한 번)
        payload_days = list(day_payload.keys()) if isinstance(day_payload, dict) else None
        chosen_day = self.ensure_stat_day(preferred_day, payload_days)

        # 1) payload 파싱
        if isinstance(day_payload, dict):
            payload_list = day_payload.get(chosen_day, [])
            print(f">>>>> ✅ 전일 교통량 연계 시뮬레이션 시작\n      대상일:{chosen_day}\n      entries:{len(payload_list)}")
        else:
            payload_list = day_payload or []
            print(f">>>>> ✅ 전일 교통량 연계 시뮬레이션 시작\n      대상일:{chosen_day}\n      (list 입력) entries:{len(payload_list)}")

        # 2) 교통량 주입
        upd = self.insert_turn_volumes(payload_list, verbose=True)
        print(f">>>>> ✅ 전일 주입 건수: {upd}")
        if upd == 0:
            print("⛔ 전일 주입 0건 — 매핑/입력값 확인 권장")

        # 3) 시뮬 → 결과 → CSV → DB
        self.simulate_prev_day()

        # 가드체크: 전일로 세팅됐는지 확인
        assert self.last_run.get("type") == "prev_day" and self.last_run.get("hour") is None, \
            f"last_run 불일치: {self.last_run}"

        df = self.get_links_result_df()
        
        self.insert_day_link_results(df, db_conn=db.conn)

        print(">>>>> ✅ 전일 시뮬레이션 완료")

    # --------------------------------------------------------------- [ 시간대별 결과값 insert ]
    
    def insert_hour_link_results(self, df: pd.DataFrame, db_conn, chunk_size: int = 20000) -> int:
        if df is None or df.empty or db_conn is None:
            print("⛔ 시간대 INSERT: DF/DB 누락")
            return 0

        stat_day = self._require_stat_day()              # 'YYYYMMDD'
        hour_lbl = str((self.last_run or {}).get("hour") or "").zfill(2)
        if not hour_lbl:
            print("⛔ last_run.hour 없음 — simulate_hour 이후 호출 필요")
            return 0
        stat_hour = stat_day + hour_lbl                  # 'YYYYMMDDHH'

        required = ["STAT_HOUR", "LINK_ID", "VC", "VEHS", "SPEED"]

        # 1) 작업용 복사 & 불필요 컬럼 제거
        work = df.copy()
        for c in ("run_type", "hour"):
            if c in work.columns:
                work.drop(columns=[c], inplace=True)

        # 2) 컬럼명 통일 + STAT_HOUR 세팅
        work.rename(columns={
            "link_id": "LINK_ID",
            "vc": "VC",
            "vehs": "VEHS",
            "speed": "SPEED",
        }, inplace=True)
        work["STAT_HOUR"] = stat_hour

        # 3) 타입 보정
        if "LINK_ID" in work.columns:
            work["LINK_ID"] = (
                work["LINK_ID"].astype(str).str.strip()
                .map(lambda x: x if x != "" else None)
                .map(lambda x: x[:10] if x is not None else None)
            )
        if "VC" in work.columns:
            work["VC"] = pd.to_numeric(work["VC"], errors="coerce")
        if "VEHS" in work.columns:
            work["VEHS"] = pd.to_numeric(work["VEHS"], errors="coerce").astype("Int64")
        if "SPEED" in work.columns:
            work["SPEED"] = pd.to_numeric(work["SPEED"], errors="coerce")
            work.loc[work["SPEED"] >= 360000, "SPEED"] = 0

        # 4) 필수 컬럼 체크 + 정렬
        missing = [c for c in required if c not in work.columns]
        if missing:
            print(f"⛔ 필수 컬럼 누락: {missing}")
            return 0
        work = work[required]

        # 5) NULL LINK_ID 제거
        before = len(work)
        work = work[work["LINK_ID"].notna()]
        if before != len(work):
            print(f"⚠️ LINK_ID NULL {before - len(work)}행 제거")

        # 6) FK 사전검사: TDA_LINK_INFO에 존재하는 LINK_ID만 남김
        cur = db_conn.cursor()
        unique_ids = sorted(set(work["LINK_ID"].tolist()))
        valid_ids = set()
        if unique_ids:
            BATCH = 900  # placeholder 제한 대비
            for i in range(0, len(unique_ids), BATCH):
                batch = unique_ids[i:i + BATCH]
                placeholders = ", ".join(["?"] * len(batch))
                sql_chk = f"SELECT LINK_ID FROM TOMMS.TDA_LINK_INFO WHERE LINK_ID IN ({placeholders})"
                cur.execute(sql_chk, batch)
                valid_ids.update(r[0] for r in cur.fetchall())

        missing_ids = sorted(set(unique_ids) - valid_ids)
        if missing_ids:
            print(f"⚠️ FK 미존재 LINK_ID {len(missing_ids)}건 — INSERT 대상에서 제외 (예시 10개): {missing_ids[:10]}")
            os.makedirs("./output", exist_ok=True)
            miss_path = f"./output/missing_link_ids_{stat_hour}.txt"
            with open(miss_path, "w", encoding="utf-8") as f:
                for lid in missing_ids:
                    f.write(str(lid) + "\n")
            print(f"📂 미존재 LINK_ID 목록 저장: {miss_path}")

        work = work[work["LINK_ID"].isin(valid_ids)]
        if work.empty:
            print("⛔ 유효 LINK_ID가 없어 INSERT 스킵")
            return 0

        # 7) 파라미터 바인딩 값 변환
        def _to_db_value(v):
            if v is pd.NA:
                return None
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                return None
            return v

        # 8) INSERT
        sql = f"INSERT INTO TOMMS.TDA_LINK_HOUR_RESULT ({', '.join(required)}) VALUES ({', '.join(['?']*len(required))})"
        cur.fast_executemany = True

        total = 0
        try:
            for s in range(0, len(work), chunk_size):
                chunk = work.iloc[s:s + chunk_size]
                data = [tuple(_to_db_value(v) for v in row) for row in chunk.itertuples(index=False, name=None)]
                cur.executemany(sql, data)
                total += len(data)

            db_conn.commit()
            print(f"      ✅ TDA_LINK_HOUR_RESULT INSERT — {stat_hour} {total}행")
            return total

        except Exception as ex:
            db_conn.rollback()
            print(f"⛔ TDA_LINK_HOUR_RESULT INSERT 오류 — 롤백: {ex}")
            return 0
    
    # --------------------------------------------------------------- [ 시간대별 road_id 결과값 insert ]
    
    def insert_hour_road_results(
        self,
        df: pd.DataFrame,
        db_conn,
        stat_hour: str,
        road_id: str,
        chunk_size: int = 20000,
    ) -> int:
        """
        TDA_ROAD_HOUR_RESULT 스키마
        - STAT_HOUR (VARCHAR10, NN)
        - ROAD_ID   (VARCHAR10, NN)
        - LINK_ID   (VARCHAR10,  Y)  # 하지만 LINK_ID 없는 행은 여기서 드롭
        - FB_VEHS   (NUMBER(9),  Y)  # '전일_용량' 매핑
        """
        if df is None or df.empty or db_conn is None:
            print("⛔ ROAD HOUR INSERT: DF/DB 누락")
            return 0

        # 0) 결과 테이블명
        table = "TOMMS.TDA_ROAD_VOL_HOUR_RESULT"

        # 1) 스키마 정규화
        work = df.copy()

        # 컬럼명 통일: 전일_용량 → FB_VEHS, link_id→LINK_ID 등
        work.rename(columns={
            "전일_용량": "FB_VEHS",
            "link_id": "LINK_ID",
            "stat_hour": "STAT_HOUR",
            "road_id": "ROAD_ID",
        }, inplace=True)

        # 파라미터로 받은 STAT_HOUR/ROAD_ID를 강제 세팅 (신뢰원 통일)
        work["STAT_HOUR"] = str(stat_hour)
        work["ROAD_ID"]   = str(road_id)

        # 필요 컬럼만 유지 (순서 고정)
        required = ["STAT_HOUR", "ROAD_ID", "LINK_ID", "FB_VEHS"]
        for c in required:
            if c not in work.columns:
                work[c] = pd.Series(dtype="object")  # 누락 컬럼 생성
        work = work[required]

        # 2) 타입 보정
        # LINK_ID: 문자열 10자, 공백/빈문자 None
        work["LINK_ID"] = (
            work["LINK_ID"].astype(str).str.strip()
            .map(lambda x: None if x == "" or x.lower() == "none" else x[:10])
        )
        # FB_VEHS: 정수(Int64)로
        work["FB_VEHS"] = pd.to_numeric(work["FB_VEHS"], errors="coerce").astype("Int64")

        # 3) LINK_ID 없는 행 제거(요구사항: 값이 없으면 모두 날림)
        before = len(work)
        work = work[work["LINK_ID"].notna()]
        if before != len(work):
            print(f"⚠️ LINK_ID NULL {before - len(work)}행 제거 (ROAD_ID={road_id}, STAT_HOUR={stat_hour})")

        if work.empty:
            print(f"⛔ INSERT 스킵 — 유효행 0 (ROAD_ID={road_id}, STAT_HOUR={stat_hour})")
            return 0

        # 4) FK 사전검사: TDA_LINK_INFO(LINK_ID)
        cur = db_conn.cursor()
        unique_ids = sorted(set(work["LINK_ID"].tolist()))
        valid_ids = set()
        if unique_ids:
            BATCH = 900
            for i in range(0, len(unique_ids), BATCH):
                batch = unique_ids[i:i+BATCH]
                placeholders = ", ".join(["?"] * len(batch))
                sql_chk = f"SELECT LINK_ID FROM TOMMS.TDA_LINK_INFO WHERE LINK_ID IN ({placeholders})"
                cur.execute(sql_chk, batch)
                valid_ids.update(r[0] for r in cur.fetchall())
        missing_ids = sorted(set(unique_ids) - valid_ids)
        if missing_ids:
            print(f"⚠️ LINK_ID FK 미존재 {len(missing_ids)}건 — 제외 (예시≤10): {missing_ids[:10]}")
            os.makedirs("./output", exist_ok=True)
            miss_path = f"./output/missing_hour_road_link_ids_{stat_hour}_{road_id}.txt"
            with open(miss_path, "w", encoding="utf-8") as f:
                for lid in missing_ids:
                    f.write(str(lid) + "\n")
            print(f"📂 FK 미존재 LINK_ID 저장: {miss_path}")
        work = work[work["LINK_ID"].isin(valid_ids)]

        if work.empty:
            print(f"⛔ INSERT 스킵 — FK 유효행 0 (ROAD_ID={road_id}, STAT_HOUR={stat_hour})")
            return 0

        # 5) 바인딩 값 변환
        def _to_db_value(v):
            if v is pd.NA:
                return None
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                return None
            return v

        # 6) INSERT
        sql = f"INSERT INTO {table} ({', '.join(required)}) VALUES ({', '.join(['?']*len(required))})"
        cur.fast_executemany = True

        # (선택) SQL 로그
        os.makedirs("./output", exist_ok=True)
        log_path = f"./output/road_hour_result_insert_{stat_hour}_{road_id}.sql.txt"
        def _sql_literal(v):
            if v is None: return "NULL"
            if isinstance(v, (int, np.integer)): return str(int(v))
            if isinstance(v, (float, np.floating)): return str(float(v))
            s = str(v).replace("'", "''"); return f"'{s}'"

        total = 0
        try:
            with open(log_path, "w", encoding="utf-8") as f_log:
                for s in range(0, len(work), chunk_size):
                    chunk = work.iloc[s:s+chunk_size]
                    data = [tuple(_to_db_value(v) for v in row) for row in chunk.itertuples(index=False, name=None)]
                    # 로그용 SQL
                    for row in data:
                        values_str = [_sql_literal(v) for v in row]
                        f_log.write(
                            f"INSERT INTO {table} ({', '.join(required)}) "
                            f"VALUES ({', '.join(values_str)});\n"
                        )
                    cur.executemany(sql, data)
                    total += len(data)

            db_conn.commit()
            print(f"      ✅ ROAD_HOUR_RESULT INSERT — STAT_HOUR={stat_hour}, ROAD_ID={road_id}, 행수={total}")
            print(f"      📂 SQL 로그: {log_path}")
            return total

        except Exception as ex:
            db_conn.rollback()
            print(f"⛔ ROAD_HOUR_RESULT INSERT 오류 — 롤백: {ex}")
            print(f"📂 SQL 로그(실패 시점까지): {log_path}")
            return 0
    
    # --------------------------------------------------------------- [ 시간대별 시뮬레이션 파이프라인 ]

    def run_hourly_pipeline(self, hourly_payload_map: dict, db_conn=None):
        """
        고정 순서: 08 → 11 → 14 → 17
        STAT_DAY는 이미 ensure_stat_day/set_stat_day로 확정되어 있어야 함.
        """
        
        # 🔵 GPA 파일 경로 지정
        gpa_file_path = r"C:\Digital Twin Simulation Network\VISUM\gpa_file"
        
        self._require_stat_day()
        stat_day = self.last_run['stat_day']
        print(f">>>>> ✅ 시간대 교통량 연계 시뮬레이션 시작\n      STAT_DAY={self.last_run['stat_day']}")

        for hh in [8, 11, 14, 17]:
            key = f"{hh:02d}"
            payload = hourly_payload_map.get(key, [])
            print(f">>>>> ✅ {key}시 처리 시작 — 교차로:{len(payload)}")

            upd = self.insert_turn_volumes(payload, verbose=False)
            print(f">>>>> ✅ {key}시 주입 건수: {upd}")

            self.simulate_hour(hh)

            # 가드체크: 시간대 세팅 확인
            assert self.last_run.get("type") == "hourly" and self.last_run.get("hour") == key, \
                f"last_run 불일치: {self.last_run}"

            df = self.get_links_result_df()
            self.insert_hour_link_results(df, db_conn=db.conn)
            
            # 🔵 INSERT 이후: GPA 파일 적용
            stat_hour = f"{stat_day}{key}"
            if gpa_file_path:
                pass
                # self.read_gpa_file_get_road_link(db_conn, gpa_file_path, stat_hour)

        print(">>>>> ✅ 시간대별 시뮬레이션 완료")











# ====================================================================================== [ main 실행함수 ]

class DualLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)   # 콘솔 출력
        self.log.write(message)        # 파일 기록

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    # 로그 폴더 생성
    log_dir = r"C:\Digital Twin Simulation Program\auto simulation\logs"
    os.makedirs(log_dir, exist_ok=True)

    # 파일명: 날짜+시간 접두어
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{ts}_visum_simulation.log")

    # dual logger 세팅
    sys.stdout = DualLogger(log_file)

    print(">>>>> ✅ VISUM 자동화 시뮬레이션을 시작합니다.")
    USE_FIXED_TIME = True  # 실시간 단위로 변경하려면 False

    if USE_FIXED_TIME:
        fixed_now = datetime.datetime.strptime("2025070201", "%Y%m%d%H")
        query_day, target_stat_hours = compute_target_hours(fixed_now, ["08", "11", "14", "17"])
    else:
        query_day, target_stat_hours = compute_target_hours(None, ["08", "11", "14", "17"])

    # 1) DB
    config = Config()
    db = DatabaseManager(config)
    print(">>>>> ✅ Config, DB 클래스가 선언되어 main 함수 내 설정이 완료되었습니다.")

    try:
        print(">>>>> ✅ 교통량 데이터 조회를 시작합니다.")
        traffic_by_hour, traffic_by_day, query_day_from_db = db.fetch_and_process_data(target_stat_hours)

        stat_day_final = query_day_from_db or query_day

        vis = VisumSimulationManager(
            base_path=r"C:/Digital Twin Simulation network/VISUM",
            default_version_name="강릉시 전국 전일 최종본.ver",
            prev_day_proc_no=22,
            csv_out_dir=r"C:/Digital Twin Simulation network/VISUM/result_export",
        )

        print(">>>>> ✅ Visum 클래스가 선언되어 main 함수 내 설정이 완료되었습니다.")

        # 4) Visum open & load
        vis.open()
        vis.set_stat_day(stat_day_final)

        # 5) 전일 파이프라인
        vis.run_prev_day_pipeline(traffic_by_day, db_conn=db.conn, preferred_day=stat_day_final)

        # 6) 시간대 파이프라인
        vis.run_hourly_pipeline(traffic_by_hour, db_conn=db.conn)

    finally:
        if 'vis' in locals():
            vis.close()
        db.close()
        print(f"📂 로그 저장 완료 → {log_file}")