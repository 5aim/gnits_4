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

    def fetch_and_process_data(self, target_stat_hours: List[str]):
        """
        입력된 STAT_HOUR(YYYYMMDDHH) 집합에 대해:
          - STAT_HOUR_CROSS에서 시간대별 자료 수집
          - 해당 일자(YYYYMMDD)를 모아 STAT_DAY_CROSS에서 일별 자료 수집
        반환:
          - traffic_data_by_hour: { "00":[{cross_id, data:[{direction, value}...] }], ... }
          - traffic_data_by_day : { "YYYYMMDD":[{cross_id, data:[...]}], ... }
          - query_day: 최초 요청 시각의 YYYYMMDD
        """
        # 1) 입력 검증 & 파라미터 구성
        target_stat_hours = [h for h in target_stat_hours if is_valid_stat_hour(h)]
        print(f">>>>> ✅ target_stat_hours 변수가 설정되었습니다. : {target_stat_hours}")
        if not target_stat_hours:
            print("⛔ 유효한 STAT_HOUR가 없습니다.")
            return {f"{h:02d}": [] for h in range(24)}, {}, None

        # WHERE (STAT_HOUR = ? OR STAT_HOUR = ? ...)
        where_clause = " OR ".join(["STAT_HOUR = ?"] * len(target_stat_hours))
        sql_hour = f"""
            SELECT *
            FROM STAT_HOUR_CROSS
            WHERE ({where_clause})
              AND INFRA_TYPE = 'SMT'
        """

        # 2) 시간대별 조회
        rows, cols = self._exec(sql_hour, tuple(target_stat_hours))
        print(f">>>>> ✅ 시간대별 조회된 행의 갯수입니다. : {len(rows)}")

        col_idx = {c: i for i, c in enumerate(cols)}
        idx_stat_hour = col_idx["STAT_HOUR"]
        idx_cross_id  = col_idx["CROSS_ID"]

        vol_names = extract_vol_columns(cols)  # 이름만 추출 후 정렬
        vol_idx_pairs = [(name, col_idx[name]) for name in vol_names]

        traffic_data_by_hour: Dict[str, List[dict]] = {f"{h:02d}": [] for h in range(24)}
        stat_days = set()

        for r in rows:
            stat_hour = str(r[idx_stat_hour]).strip()
            yyyymmdd = stat_hour[:8]
            stat_days.add(yyyymmdd)

            hh = stat_hour[-2:]
            if hh not in traffic_data_by_hour:
                print(f"⛔ 예상치 못한 hh 값 발견: {hh}")
                continue

            cross_id = str(r[idx_cross_id]).strip()
            traffic_data_by_hour[hh].append({
                "cross_id": cross_id,
                "data": [
                    {"direction": name, "value": to_py(r[idx])}
                    for name, idx in vol_idx_pairs
                    if to_py(r[idx]) is not None
                ]
            })
        print(f">>>>> ✅ 시간대별 교통량을 VISUM에 연계하기 위한 가공을 완료하였습니다.")

        # 3) 일별 조회
        traffic_data_by_day: Dict[str, List[dict]] = {}

        for day in stat_days:
            sql_day = """
                SELECT *
                FROM STAT_DAY_CROSS
                WHERE STAT_DAY = ?
                  AND INFRA_TYPE = 'SMT'
            """
            rows_day, cols_day = self._exec(sql_day, (day,))
            col_idx_day = {c: i for i, c in enumerate(cols_day)}

            idx_cross_id_day = col_idx_day["CROSS_ID"]
            vol_names_day = extract_vol_columns(cols_day)
            vol_idx_pairs_day = [(name, col_idx_day[name]) for name in vol_names_day]

            items = []
            for r in rows_day:
                cross_id = str(r[idx_cross_id_day]).strip()
                items.append({
                    "cross_id": cross_id,
                    "data": [
                        {"direction": name, "value": to_py(r[idx])}
                        for name, idx in vol_idx_pairs_day
                        if to_py(r[idx]) is not None
                    ]
                })
            traffic_data_by_day[day] = items

        # 4) 요약 출력 (루프 밖에서 한 번만)
        total_day = sum(len(v) for v in traffic_data_by_day.values())
        if stat_days:
            print(f">>>>> ✅ 전일 교통량을 VISUM에 연계하기 위한 가공을 완료하였습니다.")
            print(f"      ✅ 전일 교통량 총 {total_day}건 ({len(traffic_data_by_day)}일)")
        for hh, lst in traffic_data_by_hour.items():
            if lst:
                print(f"      ✅ 시간대별 교통량 {hh}시 : {len(lst)}건")

        # 5) 첫 쿼리 기준 day
        query_day = target_stat_hours[0][:8] if target_stat_hours else None

        return traffic_data_by_hour, traffic_data_by_day, query_day

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
        if not self.visum:
            print("⛔ Visum 객체가 없습니다.")
            return pd.DataFrame()

        run_type = (self.last_run or {}).get("type")
        hour_lbl = (self.last_run or {}).get("hour")
        if run_type not in ("prev_day", "hourly"):
            print("⛔ 실행 이력 없음 — simulate 호출 후 사용하세요.")
            return pd.DataFrame()

        stat_day = self._require_stat_day()

        attrs = [
            "AREA", "SUBAREA", "LINK_ID", "ROAD_NAME", "UPDOWN",
            "VolCapRatioPrT(AP)", "VolVehPrT(AP)", "TCur_PrTSys(a)",
        ]

        rows = self.visum.Net.Links.GetMultipleAttributes(attrs, True)

        records_expanded, invalid_ids = [], set()

        for row in rows:
            base = dict(zip(attrs, row))
            raw_id = base.get("LINK_ID")

            if not raw_id:  # None, 빈 문자열 등 제외
                continue

            # 1) 쉼표 분해 + 트림
            link_ids = [x.strip() for x in str(raw_id).split(",") if x.strip()]

            # 2) 각 link_id별로 한 행씩 복제 (10자리 숫자만 유효)
            for lid in link_ids:
                if len(lid) == 10 and lid.isdigit():
                    rec = base.copy()
                    rec["LINK_ID"] = lid  # 개별 ID로 치환
                    records_expanded.append(rec)
                else:
                    invalid_ids.add(lid)

        # 데이터프레임 구성
        df = pd.DataFrame.from_records(records_expanded)

        if df.empty:
            # 그래도 스키마는 유지
            df = pd.DataFrame(columns=[
                "AREA", "SUBAREA", "LINK_ID", "ROAD_NAME", "UPDOWN",
                "vc", "vehs", "speed", "STAT_DAY"
            ])
            df["STAT_DAY"] = stat_day
            print("      📊 DataFrame 크기: 0 행 × {0} 열".format(len(df.columns)))
            if invalid_ids:
                print(f"⚠️ 유효하지 않은 LINK_ID 예시(최대 5개): {list(sorted(invalid_ids))[:5]}")
            return df

        # 컬럼명 통일 + 숫자형 보정(소수 둘째 자리)
        df.rename(columns={
            "VolCapRatioPrT(AP)": "vc",
            "VolVehPrT(AP)": "vehs",
            "SUBAREA": "sa",
            "AREA": "DISTRICT",
            "TCur_PrTSys(a)": "speed"
        }, inplace=True)

        df["DISTRICT"]   = pd.to_numeric(df["DISTRICT"], errors="coerce")
        df["UPDOWN"] = pd.to_numeric(df["UPDOWN"], errors="coerce")
        df["vc"]     = pd.to_numeric(df["vc"], errors="coerce").round(2)
        df["vehs"] = pd.to_numeric(df["vehs"], errors="coerce").fillna(0).astype(int)
        df["speed"]  = pd.to_numeric(df["speed"], errors="coerce").round(2)

        # 문자열 컬럼은 확실히 문자열/None로
        for c in ("LINK_ID", "sa", "ROAD_NAME"):
            if c in df.columns:
                df[c] = df[c].astype(object).where(pd.notna(df[c]), None)
                df[c] = df[c].map(lambda x: str(x) if x is not None else None)

        # 3) LINK_ID 기준 중복 제거 (첫 등장 행 우선)
        before = len(df)
        df.drop_duplicates(subset=["LINK_ID"], keep="first", inplace=True)
        after = len(df)

        # (선택) 정렬
        df.sort_values(by=["LINK_ID"], inplace=True, ignore_index=True)

        df["STAT_DAY"] = stat_day

        print(f"      📊 DataFrame 크기: {len(df)} 행 × {len(df.columns)} 열"
            f" (중복 제거: {before - after}건)")
        if invalid_ids:
            print(f"⚠️ 유효하지 않은 LINK_ID 예시(최대 5개): {list(sorted(invalid_ids))[:5]}")

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











    # --------------------------------------------------------------- [ 전일 결과값 insert ]

    def insert_day_link_results(self, df: pd.DataFrame, db_conn, chunk_size: int = 20000) -> int:
        if df is None or df.empty or db_conn is None:
            print("⛔ 전일 INSERT: DF/DB 누락")
            return 0

        required = ["STAT_DAY","LINK_ID","DISTRICT","SA_NO","ROAD_NAME","UPDOWN","VC","VEHS","SPEED"]

        # 1) 준비 및 불필요 컬럼 제거
        work = df.copy()
        for c in ("run_type", "hour"):
            if c in work.columns:
                work.drop(columns=[c], inplace=True)

        # 2) 스키마 컬럼명 맞춤 + STAT_DAY 세팅
        work.rename(columns={"sa": "SA_NO", "vc": "VC", "vehs": "VEHS", "speed": "SPEED"}, inplace=True)
        work["STAT_DAY"] = self._require_stat_day()

        # 3) 숫자형 보정
        #    - DISTRICT, UPDOWN, VEHS: 정수형 (nullable Int64)
        #    - VC: float(그대로), SPEED: 숫자화 + 360000 이상은 0
        work["DISTRICT"]   = pd.to_numeric(work.get("DISTRICT"), errors="coerce").astype("Int64")
        work["UPDOWN"] = pd.to_numeric(work.get("UPDOWN"), errors="coerce").astype("Int64")
        work["VEHS"] = pd.to_numeric(work.get("VEHS"), errors="coerce").astype("Int64")
        if "VC" in work.columns:
            work["VC"] = pd.to_numeric(work["VC"], errors="coerce")
        if "SPEED" in work.columns:
            work["SPEED"] = pd.to_numeric(work["SPEED"], errors="coerce")
            work.loc[work["SPEED"] >= 360000, "SPEED"] = 0  # 비정상 속도값 방지

        # 4) 문자열 컬럼: 공백/빈문자 -> None
        def _str_or_none(x):
            if pd.isna(x):
                return None
            s = str(x).strip()
            return s if s != "" else None

        for c in ("LINK_ID", "SA_NO", "ROAD_NAME"):
            if c in work.columns:
                work[c] = work[c].map(_str_or_none)

        # 길이 제한(스키마 보호)
        if "ROAD_NAME" in work.columns:
            work["ROAD_NAME"] = work["ROAD_NAME"].map(lambda x: x[:200] if x is not None else None)
        if "LINK_ID" in work.columns:
            work["LINK_ID"] = work["LINK_ID"].map(lambda x: x[:400] if x is not None else None)
        if "SA_NO" in work.columns:
            work["SA_NO"] = work["SA_NO"].map(lambda x: x[:10]  if x is not None else None)

        # 5) 필수 컬럼 체크 + 정렬
        missing = [c for c in required if c not in work.columns]
        if missing:
            print(f"⛔ 필수 컬럼 누락: {missing}")
            return 0
        work = work[required]  # 열 정렬만 수행, 행 필터링 없음

        # 6) 파라미터 바인딩용 Python 기본형으로 변환
        def _to_db_value(v):
            # pandas NA -> None
            if v is pd.NA:
                return None
            # numpy -> python 기본형
            if isinstance(v, np.generic):
                v = v.item()
            # float NaN -> None
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                return None
            return v

        # Tibero(오라클 호환)에서 파라미터에 Python None을 넘기면 DB의 NULL로 들어간다.
        sql = f"INSERT INTO TOMMS.DAY_LINK_RESULT ({', '.join(required)}) VALUES ({', '.join(['?']*len(required))})"
        cur = db_conn.cursor()
        cur.fast_executemany = False

        # 7) SQL 로그 저장(검증용)
        os.makedirs("./output", exist_ok=True)
        sql_log_path = "./output/day_link_result_insert.sql.txt"

        def _sql_literal(v):
            if v is None:
                return "NULL"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                # 소수점 표현 안정화
                return str(float(v))
            # 문자열 이스케이프
            s = str(v).replace("'", "''")
            return "'" + s + "'"

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
                            f"INSERT INTO TOMMS.DAY_LINK_RESULT ({', '.join(required)}) VALUES ({', '.join(values_str)});\n"
                        )

                    cur.executemany(sql, data)
                    total += len(data)

            db_conn.commit()
            print(f"      ✅ DAY_LINK_RESULT INSERT — {total}행")
            print(f"      📂 SQL 로그 저장: {sql_log_path}")
            return total

        except Exception as ex:
            db_conn.rollback()
            print(f"⛔ DAY_LINK_RESULT INSERT 오류 — 롤백: {ex}")
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

        stat_day = self._require_stat_day()
        hour_lbl = (self.last_run or {}).get("hour")
        if not hour_lbl:
            print("⛔ last_run.hour 없음 — simulate_hour 이후 호출 필요")
            return 0

        required = ["STAT_HOUR","LINK_ID","DISTRICT","SA_NO","ROAD_NAME","UPDOWN","VC","VEHS","SPEED"]

        # 1) 작업용 복사 & 불필요 컬럼 제거
        work = df.copy()
        for c in ("run_type", "hour"):
            if c in work.columns:
                work.drop(columns=[c], inplace=True)

        # 2) 스키마 컬럼명 정리 + STAT_HOUR 세팅
        work.rename(columns={"sa": "SA_NO", "vc": "VC", "vehs": "VEHS", "speed": "SPEED"}, inplace=True)
        work["STAT_HOUR"] = stat_day + hour_lbl  # 예: 2025070109

        # 3) 숫자형 보정
        #    - DISTRICT/UPDOWN/VEHS: nullable 정수(Int64)
        #    - VC: float 그대로(라운딩/스케일링 없음)
        #    - SPEED: 숫자화 + (>=360000 → 0), 그 외 원본
        work["DISTRICT"]   = pd.to_numeric(work.get("DISTRICT"), errors="coerce").astype("Int64")
        work["UPDOWN"] = pd.to_numeric(work.get("UPDOWN"), errors="coerce").astype("Int64")
        work["VEHS"] = pd.to_numeric(work.get("VEHS"), errors="coerce").astype("Int64")
        if "VC" in work.columns:
            work["VC"] = pd.to_numeric(work["VC"], errors="coerce")
        if "SPEED" in work.columns:
            work["SPEED"] = pd.to_numeric(work["SPEED"], errors="coerce")
            work.loc[work["SPEED"] >= 360000, "SPEED"] = 0

        # 4) 문자열: 공백 제거 후 빈 값은 None
        def _str_or_none(x):
            if pd.isna(x):
                return None
            s = str(x).strip()
            return s if s != "" else None

        for c in ("LINK_ID", "SA_NO", "ROAD_NAME"):
            if c in work.columns:
                work[c] = work[c].map(_str_or_none)

        # 길이 제한(스키마 보호)
        if "ROAD_NAME" in work.columns:
            work["ROAD_NAME"] = work["ROAD_NAME"].map(lambda x: x[:200] if x is not None else None)
        if "LINK_ID" in work.columns:
            work["LINK_ID"] = work["LINK_ID"].map(lambda x: x[:400] if x is not None else None)
        if "SA_NO" in work.columns:
            work["SA_NO"] = work["SA_NO"].map(lambda x: x[:10]  if x is not None else None)

        # 5) 필수 컬럼 체크 + 컬럼 정렬(행 필터링 없음)
        missing = [c for c in required if c not in work.columns]
        if missing:
            print(f"⛔ 필수 컬럼 누락: {missing}")
            return 0
        work = work[required]

        # 6) 파라미터 바인딩용 Python 기본형으로 변환 (pandas/NumPy → 기본형, NA/NaN → None)
        def _to_db_value(v):
            if v is pd.NA:
                return None
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
                return None
            return v

        sql = f"INSERT INTO TOMMS.HOUR_LINK_RESULT ({', '.join(required)}) VALUES ({', '.join(['?']*len(required))})"
        cur = db_conn.cursor()
        cur.fast_executemany = True

        total = 0
        try:
            for s in range(0, len(work), chunk_size):
                chunk = work.iloc[s:s+chunk_size]
                data = [tuple(_to_db_value(v) for v in row) for row in chunk.itertuples(index=False, name=None)]
                cur.executemany(sql, data)
                total += len(data)

            db_conn.commit()
            print(f"      ✅ HOUR_LINK_RESULT INSERT — {stat_day+hour_lbl} {total}행")
            return total

        except Exception as ex:
            db_conn.rollback()
            print(f"⛔ HOUR_LINK_RESULT INSERT 오류 — 롤백: {ex}")
            return 0

    # --------------------------------------------------------------- [ 시간대별 시뮬레이션 파이프라인 ]

    def run_hourly_pipeline(self, hourly_payload_map: dict, db_conn=None):
        """
        고정 순서: 08 → 11 → 14 → 17
        STAT_DAY는 이미 ensure_stat_day/set_stat_day로 확정되어 있어야 함.
        """
        self._require_stat_day()
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

        print(">>>>> ✅ 시간대별 시뮬레이션 완료")











# ====================================================================================== [ main 실행함수 ]

if __name__ == "__main__":
    
    print(">>>>> ✅ VISUM 자동화 시뮬레이션을 시작합니다.")
    USE_FIXED_TIME = True # 실시간 단위로 변경하려면 이 값을 False로 설정.
    
    if USE_FIXED_TIME:
        fixed_now = datetime.datetime.strptime("2025070204", "%Y%m%d%H")
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
        
        # query_day 우선순위: DB에서 유추한 값이 있으면 그걸 사용
        stat_day_final = query_day_from_db or query_day
        
        vis = VisumSimulationManager(
            base_path=r"C:/Digital Twin Simulation network/VISUM",
            default_version_name="강릉시 전국 전일 최종본.ver",
            prev_day_proc_no=22,
            csv_out_dir=r"C:/Digital Twin Simulation network/VISUM/result_export",
        )
        
        print(">>>>> ✅ Visum 클래스가 선언되어 main 함수 내 설정이 완료되었습니다.")

        # 4) Visum open & load
        vis.open()  # default_version_name 사용
        vis.set_stat_day(stat_day_final)  # ★ 기준일 확정(한 번만)

        # 5) 전일 파이프라인
        vis.run_prev_day_pipeline(traffic_by_day, db_conn=db.conn, preferred_day=stat_day_final) # traffic_by_day: {"YYYYMMDD": [ ... ]} 구조

        # 6) 시간대 파이프라인(08→11→14→17)
        vis.run_hourly_pipeline(traffic_by_hour, db_conn=db.conn) # traffic_by_hour: {"00":[...], ..., "23":[...]}

    finally:
        # 7) 마무리
        if 'vis' in locals():
            vis.close()
        db.close()