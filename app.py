import pyodbc, os, subprocess, pathlib, json, hashlib
import pandas as pd
import numpy as np
import pytz

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from windows import set_dpi_awareness









set_dpi_awareness()
app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)









# ========================================================= [ 현재시간 ]

KST = pytz.timezone("Asia/Seoul")

def resolve_dataset_date(now_kst: datetime) -> str:
    """
    KST 06:00 이전  -> 전전날( D-2 )
    KST 06:00 이후  -> 전날  ( D-1 )
    """
    base = now_kst.date()
    if now_kst.hour < 6:
        target = base - timedelta(days=2)
    else:
        target = base - timedelta(days=1)
    return target.strftime("%Y%m%d")

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ========================================================= [ Etag 생성 ]

def parse_hours_param(raw: str):
    """'01,02,03,04' -> {1,2,3,4}; 비거나 유효값 없으면 None"""
    if not raw:
        return None
    hset = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            h = int(tok)
            if 0 <= h <= 23:
                hset.add(h)
        except:
            pass
    return hset or None

def make_etag(dataset_date: str, hours_filter_set, total_rows: int) -> str:
    """간단 ETag: 날짜 + 시간대 + 총행수 → md5"""
    hours_key = ",".join(sorted(f"{h:02d}" for h in (hours_filter_set or [])))
    base = f"{dataset_date}|{hours_key}|{total_rows}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()









# ========================================================= [ VISUM 자동화 코드 실행 ]

def run_visum_script():
    script_path = os.path.join(os.path.dirname(__file__), 'auto simulation', 'auto_visum.py')
    print(f"✅ [ {get_current_time()} ] Vissim 자동화 시뮬레이션 실행")
    subprocess.Popen(['python', script_path])
    
# ========================================================= [ VISSIM 자동화 코드 실행 ]

def run_vissim_script():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    script_path = os.path.join(base_dir, 'auto simulation', 'auto_vissim.py')
    
    if not os.path.exists(script_path):
        print(f"❌ 경로 오류: {script_path} 파일이 존재하지 않습니다.")
        return
    
    print(f"✅ [ {get_current_time()} ] Vissim 자동화 시뮬레이션 실행")
    subprocess.Popen(['python', script_path], cwd=os.path.dirname(script_path))









# ========================================================= [ 자동화 시뮬레이션 스케쥴러 설정 ]

# nohup python app.py > server.log 2>&1 &
scheduler = BackgroundScheduler()
scheduler.add_job(run_visum_script, 'cron', hour=2, minute=0)
scheduler.add_job(run_vissim_script, 'cron', hour=1, minute=0)
scheduler.start()

# ========================================================= [ 티베로 연결 ]

load_dotenv()
FLASK_ENV = os.getenv("FLASK_ENV", "production")

# 강릉 센터용
DSNNAME = os.getenv("DSNNAME")
DBUSER = os.getenv("DBUSER")
DBPWD = os.getenv("DBPWD")

# 엔제로 테스트용
ENZERO_SERVER = os.getenv("ENZERO_SERVER")
ENZERO_PORT = os.getenv("ENZERO_PORT")
ENZERO_DB = os.getenv("ENZERO_DB")
ENZERO_UID = os.getenv("ENZERO_UID")
ENZERO_PWD = os.getenv("ENZERO_PWD")

def get_connection():
    if FLASK_ENV == "test":
        print(f">>> [INFO] Flask 환경 설정: {FLASK_ENV}")
        return pyodbc.connect(
            f"DRIVER=Tibero 5 ODBC Driver;"
            f"SERVER={ENZERO_SERVER};"
            f"PORT={ENZERO_PORT};"
            f"DB={ENZERO_DB};"
            f"UID={ENZERO_UID};"
            f"PWD={ENZERO_PWD};"
        )
    else:
        print(f">>> [INFO] Flask 환경 설정: {FLASK_ENV}")
        return pyodbc.connect(
            f"DSN={DSNNAME};"
            f"UID={DBUSER};"
            f"PWD={DBPWD}"
        )









# ========================================================= [ Decimal 처리 함수 ]

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return obj

# ========================================================= [ 권역별 매핑 ]

district_mapping = {
    1: "교동지구",
    2: "송정동",
    3: "도심",
    4: "아레나"
}
hourly_mapping = {
    "08": "오전첨두 08시 ~ 09시",
    "11": "오전비첨두 11시 ~ 12시",
    "14": "오후비첨두 14시 ~ 15시",
    "17": "오후첨두 17시 ~ 18시"
}

# ========================================================= [ 지체시간 기준 LOS ]

def get_los(delay):
    if delay < 15: return "A"
    elif delay < 30: return "B"
    elif delay < 50: return "C"
    elif delay < 70: return "D"
    elif delay < 100: return "E"
    elif delay < 220: return "F"
    elif delay < 340: return "FF"
    else: return "FFF"









# ========================================================= [ 로그인 ]

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

# ========================================================= [ 회원가입 ]

@app.route('/sign-up')
def sign_up():
    return render_template('sign_up.html')

# ========================================================= [ 메인페이지 ]

@app.route('/home')
def home():
    return render_template('home.html')

# ========================================================= [ 딥러닝 학습 ]

# @app.route('/gndl-learn-start', methods=['GET'])
# def deeplearning_learn_start():
#     try:
#         logs = []
        
#         VENV_PYTHON = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
        
#         base_dir = os.path.join(os.getcwd(), "gndl")
#         gnn_data_dir = os.path.join(base_dir, "gnn_data")

#         pkl_files = ["node_features.pkl", "edge_list.pkl", "node_index.pkl"]
#         pkl_paths = [os.path.join(gnn_data_dir, f) for f in pkl_files]

#         # STEP 1: 전처리 필요 여부 확인
#         if not all(os.path.exists(p) for p in pkl_paths):
#             logs.append("🟡 전처리 데이터 없음 → 1.preprocess.py 실행")
#             result = subprocess.run([VENV_PYTHON, "gndl/1.preprocess.py"], capture_output=True, text=True)
#             logs.append(result.stdout or result.stderr)
#         else:
#             logs.append("✅ 전처리된 데이터가 존재하여 건너뜀")

#         # STEP 2: 학습 모델 파일 체크
#         model_path = os.path.join(gnn_data_dir, "best_model.pt")
#         if not os.path.exists(model_path):
#             logs.append("🟡 모델 없음 → 2.gnn.py 실행")
#             result = subprocess.run([VENV_PYTHON, "gndl/2.gnn.py"], capture_output=True, text=True)
#             logs.append(result.stdout or result.stderr)
#         else:
#             logs.append("✅ 학습된 모델이 존재하여 건너뜀")

#         # STEP 3: 시각화 결과 확인
#         visual_path = os.path.join(gnn_data_dir, "visual_result.png")
#         if not os.path.exists(visual_path):
#             logs.append("🟡 시각화 없음 → 3.gnn_visual.py 실행")
#             result = subprocess.run([VENV_PYTHON, "gndl/3.gnn_visual.py"], capture_output=True, text=True)
#             logs.append(result.stdout or result.stderr)
#         else:
#             logs.append("✅ 시각화 결과가 존재하여 건너뜀")

#         # STEP 4: 미래 예측 결과 확인
#         pred_path = os.path.join(gnn_data_dir, "future_prediction.json")
#         if not os.path.exists(pred_path):
#             logs.append("🟡 미래 예측 없음 → 4.future_prediction.py 실행")
#             result = subprocess.run([VENV_PYTHON, "gndl/4.future_prediction.py"], capture_output=True, text=True)
#             logs.append(result.stdout or result.stderr)
#         else:
#             logs.append("✅ 미래 예측 결과가 존재하여 건너뜀")

#         return jsonify({
#             "status": "success",
#             "message": "GNN 딥러닝 전체 파이프라인 실행 완료",
#             "logs": logs
#         })

#     except Exception as e:
#         return jsonify({
#             "status": "fail",
#             "message": "❌ GNN 학습 중 오류 발생",
#             "error": str(e),
#             "logs": logs
#         })









# ========================================================= [ 모니터링 1 - 시간대별 교통수요 분석정보 ]

@app.route('/monitoring/visum-hourly-vc', methods=['GET'])
def visum_hourly_vc():
    pass

# ========================================================= [ 모니터링 2 - 교통존간 통행정보 ]

@app.route('/monitoring/visum-zone-od', methods=['GET'])
def visum_zone_od():
    try:
        # ---- query params ----
        from_filter = request.args.get('from')   # ex) '810011'
        top_param   = request.args.get('top', default='10')
        try:
            top_n = max(1, int(top_param))
        except:
            top_n = 10

        conn = get_connection()
        cursor = conn.cursor()

        # ---- 동적 WHERE (단일 from 필터) + qmark(?) 플레이스홀더 ----
        inner_where = ""
        params = []
        if from_filter:
            inner_where = "WHERE o.FROM_ZONE_ID = ?"
            params.append(from_filter)

        query = f"""
            SELECT *
            FROM (
                SELECT
                    o.FROM_ZONE_ID,
                    o.FROM_ZONE_NAME,
                    o.TO_ZONE_ID,
                    o.TO_ZONE_NAME,
                    NVL(o.AUTO_MATRIX_VALUE,0)
                    + NVL(o.BUS_MATRIX_VALUE,0)
                    + NVL(o.HGV_MATRIX_VALUE,0)              AS OD_MATRIX_VALUE,
                    f.LON AS FROM_LON, f.LAT AS FROM_LAT,
                    t.LON AS TO_LON,   t.LAT AS TO_LAT,
                    ROW_NUMBER() OVER (
                        PARTITION BY o.FROM_ZONE_ID
                        ORDER BY NVL(o.AUTO_MATRIX_VALUE,0)
                               + NVL(o.BUS_MATRIX_VALUE,0)
                               + NVL(o.HGV_MATRIX_VALUE,0) DESC
                    ) AS RN
                FROM VISUM_ZONE_OD o
                LEFT JOIN VISUM_ZONE_INFO f ON f.ZONE_ID = o.FROM_ZONE_ID
                LEFT JOIN VISUM_ZONE_INFO t ON t.ZONE_ID = o.TO_ZONE_ID
                {inner_where}
            )
            WHERE RN <= ?
            ORDER BY FROM_ZONE_ID, RN
        """
        params.append(top_n)
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        idx  = {c: i for i, c in enumerate(cols)}

        # ---- from별 그룹 ----
        from collections import defaultdict
        groups = defaultdict(lambda: {
            "from_zone_id": None,
            "from_zone_name": None,
            "from_lon": None, "from_lat": None,
            "items": []  # list of dicts
        })

        rank_counter = defaultdict(int)
        for r in rows:
            fid  = r[idx["FROM_ZONE_ID"]]
            fnm  = r[idx["FROM_ZONE_NAME"]]
            flon = r[idx["FROM_LON"]]; flat = r[idx["FROM_LAT"]]
            tid  = r[idx["TO_ZONE_ID"]]
            tnm  = r[idx["TO_ZONE_NAME"]]
            tlon = r[idx["TO_LON"]];   tlat = r[idx["TO_LAT"]]
            val  = r[idx["OD_MATRIX_VALUE"]]

            g = groups[fid]
            if g["from_zone_id"] is None:
                g["from_zone_id"]   = str(fid) if fid is not None else None
                g["from_zone_name"] = fnm
                g["from_lon"]       = float(flon) if flon is not None else None
                g["from_lat"]       = float(flat) if flat is not None else None

            rank_counter[fid] += 1
            g["items"].append({
                "to_zone_id": str(tid) if tid is not None else None,
                "to_zone_name": tnm,
                "to_lon": float(tlon) if tlon is not None else None,
                "to_lat": float(tlat) if tlat is not None else None,
                "value": round(float(val), 2) if val is not None else 0.0,
                "rank": rank_counter[fid]
            })

        # ---- 항상 GeoJSON만 반환 (좌표는 [lat, lon]) ----
        payload = []
        for _, g in groups.items():
            # Points FeatureCollection: from 1개 + to N개
            points = {"type": "FeatureCollection", "features": []}
            # from point
            points["features"].append({
                "type": "Feature",
                "properties": {
                    "role": "from",
                    "zone_id": g["from_zone_id"],
                    "zone_name": g["from_zone_name"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [g["from_lat"], g["from_lon"]]   # [lat, lon]
                }
            })
            # to points
            for it in g["items"]:
                points["features"].append({
                    "type": "Feature",
                    "properties": {
                        "role": "to",
                        "zone_id": it["to_zone_id"],
                        "zone_name": it["to_zone_name"],
                        "value": it["value"],
                        "rank": it["rank"]
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [it["to_lat"], it["to_lon"]]   # [lat, lon]
                    }
                })

            # Lines FeatureCollection: from→to N개
            lines = {"type": "FeatureCollection", "features": []}
            for it in g["items"]:
                lines["features"].append({
                    "type": "Feature",
                    "properties": {
                        "from_zone_id": g["from_zone_id"],
                        "from_zone_name": g["from_zone_name"],
                        "to_zone_id": it["to_zone_id"],
                        "to_zone_name": it["to_zone_name"],
                        "value": it["value"],
                        "rank": it["rank"]
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [g["from_lat"], g["from_lon"]],   # [lat, lon]
                            [it["to_lat"], it["to_lon"]]       # [lat, lon]
                        ]
                    }
                })

            payload.append({
                "from_zone": {
                    "id": g["from_zone_id"],
                    "name": g["from_zone_name"],
                    "coordinates": [g["from_lat"], g["from_lon"]]  # [lat, lon]
                },
                "points": points,
                "lines": lines,
                "meta": {"top": top_n, "format": "geojson", "coord_order": "latlon"}
            })

        result = payload[0] if from_filter and len(payload) == 1 else payload
        conn.close()

        # ---- 헤더: ETag / X-Dataset-Date / X-Next-Update ----
        body = json.dumps(result, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'  # 따옴표 포함

        # 데이터셋 날짜: 이 테이블에는 날짜 컬럼이 없으니 KST 오늘(YYYYMMDD)로 표기
        dataset_date = datetime.now(KST).strftime("%Y%m%d")

        # 다음 업데이트 시각: 매일 06:00 KST
        now_kst = datetime.now(KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # If-None-Match 처리(따옴표 유무 모두 허용)
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/","").strip()
        if inm == etag.strip('"'):
            resp = make_response("", 304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = dataset_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # 200 응답
        resp = make_response(body, 200)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = dataset_date
        resp.headers["X-Next-Update"] = x_next_update
        return resp

    except Exception as e:
        resp = make_response(jsonify({
            "status": "fail",
            "message": "OD 분석 중 에러 발생",
            "error": str(e),
            "timestamp": get_current_time()
        }), 500)
        resp.headers["Cache-Control"] = "no-store"
        return resp

# ========================================================= [ 모니터링 3 - 분석지역별 교통흐름 통계정보 ] - 4k 

@app.route('/monitoring/statistics-traffic-flow/node-result', methods=['GET'])
def statistics_traffic_flow():
    try:
        # --- 0) 테스트: 고정 날짜 + hours 필터 ---
        now_kst = datetime.now(KST)  # 참고용
        # rule_date = resolve_dataset_date(now_kst)  # ▶ 배포 시 이 줄로 복원
        rule_date = "20250701"  # ▶ 테스트용 강제 날짜(배포 시 삭제/주석)
        hours_filter = parse_hours_param((request.args.get('hours') or '').strip())
        
        map_center_coordinates = [
            {
                "district":"교동지구",
                "coordinates": [128.874273, 37.765208]
            },
            {
                "district":"송정동",
                "coordinates": [128.924538, 37.771808]
            },
            {
                "district":"도심",
                "coordinates": [128.897176, 37.755575]
            },
            {
                "district":"아레나",
                "coordinates": [128.891529, 37.787484]
            }
        ]

        conn = get_connection()
        cursor = conn.cursor()

        # --- 1) 지정 날짜로 데이터 조회 (없으면 404) ---
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, NODE_ID, DELAY, VEHS
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows] if rows else []
        if not rows:
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        df_result = pd.DataFrame(rows, columns=["DISTRICT", "STAT_HOUR", "NODE_ID", "DELAY", "VEHS"])
        df_result["DELAY"] = pd.to_numeric(df_result["DELAY"], errors="coerce")
        df_result["VEHS"]  = pd.to_numeric(df_result["VEHS"], errors="coerce").fillna(0).astype(int)
        df_result["DISTRICT"] = df_result["DISTRICT"].apply(lambda x: int(x) if pd.notna(x) else None)

        cursor.execute("SELECT NODE_ID, LAT, LON FROM NODE_INFO")
        info_rows = cursor.fetchall()
        if not info_rows:
            return jsonify({"status": "fail", "message": "NODE_INFO 데이터가 없습니다."}), 404
        info_rows = [tuple(r) for r in info_rows]
        df_info = pd.DataFrame(info_rows, columns=["NODE_ID", "LAT", "LON"])

        # --- 2) 병합 + 필터 ---
        df = pd.merge(df_result, df_info, on="NODE_ID", how="left")
        df = df.dropna(subset=["LAT", "LON", "DISTRICT"])
        df["DISTRICT_NAME"] = df["DISTRICT"].map(district_mapping)
        df["HOUR"] = df["STAT_HOUR"].str[-2:].astype(int)  # 0~23

        if hours_filter is not None:
            df = df[df["HOUR"].isin(hours_filter)]

        # 공통 헤더 계산(다음 업데이트 시각: 매일 06:00 KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        if df.empty:
            # 빈 결과에도 ETag, X-Next-Update, X-Dataset-Date 제공
            etag = f'"{make_etag(rule_date, hours_filter, total_rows=0)}"'
            resp = Response(status=204)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # --- 3) 집계 ---
        vehs_df = (
            df.groupby(["HOUR", "DISTRICT", "DISTRICT_NAME"])["VEHS"]
              .sum().reset_index().rename(columns={"VEHS": "VEHS_SUM"})
        )
        node_df = df[["HOUR","DISTRICT","DISTRICT_NAME","NODE_ID","DELAY","LAT","LON"]].copy()
        node_df["DELAY"] = node_df["DELAY"].round(2)
        node_df["LOS"] = node_df["DELAY"].apply(get_los)

        # --- 4) ETag / If-None-Match ---
        etag = f'"{make_etag(rule_date, hours_filter, total_rows=len(df))}"'
        # 따옴표/Weak 태그 허용 비교
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/","").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # --- 5) 시간 블록 구성 (라벨: "n월 n일 HH시 ~ HH+1시") ---
        mm = int(rule_date[4:6]); dd = int(rule_date[6:8])
        hours = sorted(df["HOUR"].unique().tolist())
        hour_blocks = []

        for h in hours:
            h_next = (h + 1) % 24
            hour_label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"

            # 시간 h의 지구별 총 교통량 → total_traffic_value 로 외부에 분리
            recs = (
                vehs_df[vehs_df["HOUR"] == h]
                .sort_values(["DISTRICT"])
                .to_dict("records")
            )
            total_traffic_value = [
                {
                    "code": int(rec["DISTRICT"]),
                    "district": rec["DISTRICT_NAME"],
                    "vehs": int(rec["VEHS_SUM"])
                }
                for rec in recs
            ]

            # district_data 내부에는 code, district, geojson(좌표+los만)만 유지
            district_data = []
            for rec in recs:
                code = int(rec["DISTRICT"])
                name = rec["DISTRICT_NAME"]

                sub = node_df[(node_df["HOUR"] == h) & (node_df["DISTRICT"] == code)]
                features = []
                for _, r in sub.iterrows():
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            # 현재 메타 'latlon'을 유지 (표준 GeoJSON 사용 시 [lon, lat]로 교체)
                            "coordinates": [float(r["LAT"]), float(r["LON"])]
                        },
                        "properties": {
                            "los": r["LOS"]
                        }
                    })

                district_data.append({
                    "code": code,
                    "district": name,
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": features
                    }
                })

            hour_blocks.append({
                "hour_label": hour_label,
                "total_traffic_value": total_traffic_value,
                "district_data": district_data
            })

        payload = {
            "status": "success",
            "target_date": rule_date,
            "map_center_coordinates": map_center_coordinates,
            "data": hour_blocks,
            "meta": {
                "coord_order": "latlon",
                "hours_filter": sorted(list(hours_filter)) if hours_filter else None,
                "note": "테스트 모드: target_date 고정(2025-07-01)"
            }
        }

        body = json.dumps(payload, ensure_ascii=False)
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        return resp

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ========================================================= [ 모니터링 4 - 도로구간별 통행량 정보 ]

@app.route('/monitoring/road-traffic-info', methods=['GET'])
def road_traffic_info():
    pass

# ========================================================= [ 모니터링 5 - 교차로별 통행정보 ]

@app.route('/monitoring/node-result', methods=['GET'])
def node_result_summary():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1) 최신 날짜(YYYYMMDD)
        cursor.execute("""
            SELECT STAT_DATE FROM (
                SELECT SUBSTR(STAT_HOUR, 1, 8) AS STAT_DATE
                FROM NODE_RESULT
                GROUP BY SUBSTR(STAT_HOUR, 1, 8)
                ORDER BY STAT_DATE DESC
            )
            WHERE ROWNUM = 1
        """)
        latest_date_row = cursor.fetchone()
        if not latest_date_row:
            return jsonify({"status": "fail", "message": "STAT_HOUR 날짜 데이터가 없습니다."}), 404
        latest_date = latest_date_row[0]
        mm, dd = int(latest_date[4:6]), int(latest_date[6:8])

        # 2) 데이터 조회
        cursor.execute("""
            SELECT STAT_HOUR, TIMEINT, NODE_ID, QLEN, VEHS, DELAY, STOPS
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows]
        if not rows:
            return jsonify({"status": "fail", "message": "해당 날짜에 대한 교차로 데이터가 없습니다."}), 404

        # 3) DataFrame 생성
        df = pd.DataFrame(rows, columns=["STAT_HOUR", "TIMEINT", "NODE_ID", "QLEN", "VEHS", "DELAY", "STOPS"])
        df[["QLEN", "VEHS", "DELAY", "STOPS"]] = df[["QLEN", "VEHS", "DELAY", "STOPS"]].apply(pd.to_numeric, errors="coerce")
        df["DATE"] = df["STAT_HOUR"].str[:8]
        df["HOUR"] = df["STAT_HOUR"].str[8:10]

        # 4) 평균값 계산
        df_avg = df.groupby(["DATE", "HOUR", "NODE_ID"], as_index=False).agg({
            "QLEN": "mean",
            "VEHS": "mean",
            "DELAY": "mean",
            "STOPS": "mean"
        }).round(2)

        # 5) 교차로 이름 매핑
        cursor.execute("SELECT NODE_ID, CROSS_NAME FROM NODE_INFO")
        node_info_rows = cursor.fetchall()
        node_info = [tuple(r) for r in node_info_rows]
        df_node_info = pd.DataFrame(node_info, columns=["NODE_ID", "NODE_NAME"]).drop_duplicates(subset="NODE_ID")

        df_merged = df_avg.merge(df_node_info, on="NODE_ID", how="left")
        df_merged = df_merged[df_merged["NODE_NAME"].notna()].copy()

        # 6) LOS 계산
        def los_alpha(delay):
            if delay < 15: return "A"
            elif delay < 30: return "B"
            elif delay < 50: return "C"
            elif delay < 70: return "D"
            elif delay < 100: return "E"
            elif delay < 220: return "F"
            elif delay < 340: return "FF"
            else: return "FFF"

        los_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"FF":6,"FFF":6}
        df_merged["LOS_NUM"] = df_merged["DELAY"].apply(lambda d: los_map[los_alpha(d)])

        # 7) 컬럼 소문자
        df_merged.rename(columns={
            "NODE_NAME": "node_name",
            "QLEN": "qlen",
            "VEHS": "vehs",
            "DELAY": "delay",
            "STOPS": "stops"
        }, inplace=True)

        # 8) 시간 블록 데이터 구성
        data_blocks = []
        for hour, group in df_merged.groupby("HOUR"):
            h = int(hour); h_next = (h + 1) % 24
            hour_label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"

            items = []
            for _, r in group.iterrows():
                qlen  = float(r["qlen"])  if pd.notna(r["qlen"])  else 0.0
                vehs  = float(r["vehs"])  if pd.notna(r["vehs"])  else 0.0
                delay = float(r["delay"]) if pd.notna(r["delay"]) else 0.0
                stops = float(r["stops"]) if pd.notna(r["stops"]) else 0.0
                los   = int(r["LOS_NUM"]) if pd.notna(r["LOS_NUM"]) else None

                items.append({
                    "node_name": str(r["node_name"]),
                    "qlen": round(qlen, 2),
                    "vehs": round(vehs, 2),
                    "delay": round(delay, 2),
                    "stops": round(stops, 2),
                    "los": los,
                    "max_qlen": round(qlen * 1.5, 2),
                    "max_vehs": round(vehs * 1.5, 2),
                    "max_delay": round(delay * 1.5, 2),
                    "max_stops": 5,
                    "max_los": 6
                })

            data_blocks.append({"hour_label": hour_label, "items": items})

        # 9) Payload & ETag
        payload = {"target_date": latest_date, "data": data_blocks}
        body = json.dumps(payload, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'

        # 10) 다음 업데이트 시간 (매일 06:00 KST)
        now_kst = datetime.now(KST)
        next_update_kst = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update_kst:
            next_update_kst += timedelta(days=1)  # 이미 지난 경우 다음날 06시

        x_next_update_str = next_update_kst.isoformat()

        # 11) If-None-Match 처리
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/", "").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = latest_date
            resp.headers["X-Next-Update"] = x_next_update_str
            return resp

        # 12) 정상 응답
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = latest_date
        resp.headers["X-Next-Update"] = x_next_update_str
        return resp

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "교차로 결과 조회 중 오류 발생",
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500









# ========================================================= [ 신호운영 1 - 도로축별 통계정보 ]

@app.route('/signal/vttm-result', methods=['GET'])
def vttm_result_summary():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1) 가장 최신 날짜(YYYYMMDD)
        cursor.execute("""
            SELECT STAT_DATE FROM (
                SELECT SUBSTR(STAT_HOUR, 1, 8) AS STAT_DATE
                FROM NODE_RESULT
                GROUP BY SUBSTR(STAT_HOUR, 1, 8)
                ORDER BY STAT_DATE DESC
            )
            WHERE ROWNUM = 1
        """)
        latest_date_row = cursor.fetchone()
        if not latest_date_row:
            return jsonify({"status": "fail", "message": "STAT_HOUR 날짜 데이터가 없습니다."}), 404
        latest_date = latest_date_row[0]

        # 2) 해당 날짜 VTTM 결과
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, FROM_NODE_NAME, TO_NODE_NAME, UPDOWN, DISTANCE, TRAVEL_TIME
            FROM VTTM_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows] if rows else []
        if not rows:
            return jsonify({"status": "fail", "message": "해당 날짜에 대한 VTTM 데이터가 없습니다."}), 404

        columns = ['DISTRICT', 'STAT_HOUR', 'FROM_NODE_NAME', 'TO_NODE_NAME', 'UPDOWN', 'DISTANCE', 'TRAVEL_TIME']

        # (district_name, hour_label, segment_key) -> {'0': {...}, '1': {...}}
        pair_buffer = defaultdict(dict)

        for row in rows:
            record = dict(zip(columns, row))
            district_id = record['DISTRICT']
            stat_hour   = record['STAT_HOUR']

            # "HH시 ~ HH+1시"
            hour_val   = int(stat_hour[-2:])
            hour_label = f"{hour_val}시 ~ {(hour_val + 1) % 24}시"

            district_name = district_mapping.get(district_id, f"기타지역-{district_id}")

            from_node = str(record['FROM_NODE_NAME'])
            to_node   = str(record['TO_NODE_NAME'])
            updown    = str(record['UPDOWN'])
            distance  = float(record['DISTANCE'] or 0)
            ttime_val = float(record['TRAVEL_TIME'] or 0)
            tcost_val = float((record.get('TRAVEL_COST') or 0))  # SELECT에 없으면 0

            travel_time  = round(ttime_val, 1) if ttime_val > 0 else 0.0
            travel_speed = round((distance / ttime_val) * 3.6, 1) if ttime_val > 0 else 0.0
            travel_cost  = round(tcost_val, 1) if tcost_val > 0 else 0.0

            segment_key = tuple(sorted([from_node, to_node]))
            key = (district_name, hour_label, segment_key)

            pair_buffer[key][updown] = {
                "from_node": from_node,
                "to_node": to_node,
                "travel_time": travel_time,
                "travel_speed": travel_speed,
                "travel_cost": travel_cost
            }

        # hour_label -> district_name -> items[]
        hour_district_map = defaultdict(lambda: defaultdict(list))

        for (district_name, hour_label, segment_key), directions in pair_buffer.items():
            if '0' in directions and '1' in directions:
                from_node_data = directions['0']
                to_node_data   = directions['1']

                dir_list = [
                    {
                        "updown": 0,
                        "travel_time": from_node_data['travel_time'],
                        "travel_speed": from_node_data['travel_speed'],
                        "travel_cost": from_node_data['travel_cost']
                    },
                    {
                        "updown": 1,
                        "travel_time": to_node_data['travel_time'],
                        "travel_speed": to_node_data['travel_speed'],
                        "travel_cost": to_node_data['travel_cost']
                    }
                ]

                data_collection = {
                    "traffic_vol": 0,  # TODO: 쿼리 연동 시 실제 값으로 대체
                    "travel_speed": from_node_data['travel_speed'],
                    "travel_time": from_node_data['travel_time']
                }

                hour_district_map[hour_label][district_name].append({
                    "from_node": segment_key[0],
                    "to_node": segment_key[1],
                    "directions": dir_list,
                    "data_collection": data_collection
                })

        # 배열로 변환 (value 중심)
        data_blocks = []
        for hour_label, districts in hour_district_map.items():
            for district_name, items in districts.items():
                data_blocks.append({
                    "hour_label": hour_label,
                    "district": district_name,
                    "items": items
                })

        payload = {
            "status": "success",
            "target_date": latest_date,
            "data": data_blocks
        }

        # ---- ETag / X-Next-Update / X-Dataset-Date ----
        body = json.dumps(payload, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'  # 쌍따옴표 포함

        now_kst = datetime.now(KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # If-None-Match 처리(따옴표/Weak 허용)
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/", "").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = latest_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # 200 응답
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = latest_date
        resp.headers["X-Next-Update"] = x_next_update
        return resp

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "교차로 결과 조회 중 오류 발생",
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# ========================================================= [ 신호운영 2 - 지점별 통행정보 ]

@app.route('/signal/vttm-traffic-info', methods=['GET'])
def vttm_traffic_info():
    pass

# ========================================================= [ 신호운영 3 - 시간대별 교통혼잡 정보 ]

@app.route('/signal/hourly-congested-info', methods=['GET'])
def hourly_congested_info_data():
    pass

def hourly_congested_info_map_data():
    pass

# ========================================================= [ 신호운영 4 - 교차로별 효과지표 분석정보 ]

@app.route('/signal/node-approach-result', methods=['GET'])
def node_approach_result():
    hour_filter = (request.args.get('hour') or '').strip()  # '08','11','14','17' 등 2자리
    if not (len(hour_filter) == 2 and hour_filter.isdigit() and 0 <= int(hour_filter) <= 23):
        return jsonify({
            "status": "fail",
            "message": f"유효하지 않은 hour 파라미터입니다: {hour_filter}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 400

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1) 최신 일자(YYYYMMDD)
        cursor.execute("""
            SELECT STAT_DATE FROM (
                SELECT SUBSTR(STAT_HOUR, 1, 8) AS STAT_DATE
                FROM NODE_DIR_RESULT
                GROUP BY SUBSTR(STAT_HOUR, 1, 8)
                ORDER BY STAT_DATE DESC
            )
            WHERE ROWNUM = 1
        """)
        latest_date_row = cursor.fetchone()
        if not latest_date_row:
            return jsonify({
                "status": "fail",
                "message": "NODE_DIR_RESULT에 STAT_HOUR 데이터가 없습니다.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        latest_date = latest_date_row[0]  # 'YYYYMMDD'
        mm, dd = int(latest_date[4:6]), int(latest_date[6:8])
        h = int(hour_filter); h_next = (h + 1) % 24
        label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"

        # 2) 최신 일자의 전체 행 중 요청 시(hour)만 필터링
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, TIMEINT, NODE_ID, CROSS_ID, SA_NO,
                   APPR_ID, DIRECTION, QLEN, VEHS, DELAY, STOPS
            FROM NODE_DIR_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        rows = [tuple(r) for r in cursor.fetchall()]
        cols = ['DISTRICT', 'STAT_HOUR', 'TIMEINT', 'NODE_ID', 'CROSS_ID', 'SA_NO',
                'APPR_ID', 'DIRECTION', 'QLEN', 'VEHS', 'DELAY', 'STOPS']
        df = pd.DataFrame(rows, columns=cols)

        # 숫자 컬럼 변환(문자 열은 그대로 유지)
        for col in ["APPR_ID","DIRECTION","QLEN","VEHS","DELAY","STOPS","CROSS_ID"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # hour 필터
        df = df[df["STAT_HOUR"].astype(str).str[-2:] == hour_filter].copy()
        if df.empty:
            return jsonify({
                "status": "fail",
                "message": f"{label}에 해당하는 데이터가 없습니다.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        # TIMEINT는 문자열 4개만 유효
        ORDERED_SLICES = ['00-15', '15-30', '30-45', '45-00']
        df = df[df['TIMEINT'].isin(ORDERED_SLICES)].copy()

        # 3) NODE_DIR_INFO 조회(메타)
        cursor.execute("""
            SELECT CROSS_ID, DISTRICT, NODE_ID, NODE_NAME, CROSS_TYPE, INT_TYPE,
                   APPR_ID, DIRECTION, APPR_NAME
            FROM NODE_DIR_INFO
        """)
        info_rows = [tuple(r) for r in cursor.fetchall()]
        info_cols  = ['CROSS_ID','DISTRICT','NODE_ID','NODE_NAME','CROSS_TYPE','INT_TYPE',
                      'APPR_ID','DIRECTION','APPR_NAME']
        df_info = pd.DataFrame(info_rows, columns=info_cols)
        for col in ["CROSS_ID","APPR_ID","DIRECTION","CROSS_TYPE"]:
            if col in df_info.columns:
                df_info[col] = pd.to_numeric(df_info[col], errors="coerce")

        df_node_meta = df_info.drop_duplicates(subset=['NODE_ID'])[['NODE_ID','NODE_NAME','CROSS_TYPE','INT_TYPE']].set_index('NODE_ID')
        df_appr_meta = df_info[['NODE_ID','APPR_ID','DIRECTION','APPR_NAME']].dropna()
        if 'APPR_ID' in df_appr_meta.columns:
            df_appr_meta['APPR_ID'] = pd.to_numeric(df_appr_meta['APPR_ID'], errors="coerce")

        # 4) 결과 가공
        nodes = []

        for node_id, df_node in df.groupby('NODE_ID'):
            if node_id not in df_node_meta.index:
                continue

            node_meta = df_node_meta.loc[node_id]
            node_name = node_meta['NODE_NAME']
            cross_id  = df_node['CROSS_ID'].dropna().iloc[0] if not df_node['CROSS_ID'].dropna().empty else None
            sa_no     = df_node['SA_NO'].dropna().iloc[0] if 'SA_NO' in df_node.columns and not df_node['SA_NO'].dropna().empty else None

            # --- 시간대 전체 접근로 요약 (hourly) ---
            hourly_items = []
            all_vehs_total, all_delay_sum, all_delay_count = 0, 0.0, 0

            for appr_id, df_appr in df_node.groupby('APPR_ID'):
                vehs = int(df_appr['VEHS'].sum(skipna=True) or 0)
                delay_vals = df_appr['DELAY'].dropna().astype(float).tolist()
                delay_avg = round(sum(delay_vals) / len(delay_vals), 1) if delay_vals else 0.0
                los = get_los(delay_avg)

                all_vehs_total += vehs
                all_delay_sum  += sum(delay_vals)
                all_delay_count += len(delay_vals)

                match = df_appr_meta[(df_appr_meta['NODE_ID'] == node_id) & (df_appr_meta['APPR_ID'] == appr_id)]
                appr_name = match.iloc[0]['APPR_NAME'] if not match.empty else "미지정"

                hourly_items.append({
                    "appr_name": appr_name,
                    "vehs": vehs,
                    "delay": delay_avg,
                    "los": los
                })

            total_delay = round(all_delay_sum / all_delay_count, 1) if all_delay_count > 0 else 0.0
            total_los = get_los(total_delay)

            result_obj = {
                "node_name": node_name,
                "cross_id": int(cross_id) if pd.notna(cross_id) else None,
                "sa_no": sa_no,
                "cross_type": int(node_meta['CROSS_TYPE']) if pd.notna(node_meta['CROSS_TYPE']) else None,
                "int_type": node_meta['INT_TYPE'],
                "daily_total_vehs": 0,           # 필요 시 계산
                "total_vehs": all_vehs_total,
                "total_delay": total_delay,
                "total_los": total_los,
                "signal_circle": 150,            # ✅ 중앙 신호주기(고정값)
                "hourly": hourly_items,
                "time_slices": []
            }

            # --- 구간별 time_slices(4개 고정) ---
            for slice_label in ORDERED_SLICES:
                df_time = df_node[df_node['TIMEINT'] == slice_label].copy()

                # (1) 원시 items: APPR_ID × DIRECTION
                items = []
                if not df_time.empty:
                    for (appr_id, direction), df_pair in df_time.groupby(['APPR_ID', 'DIRECTION']):
                        match = df_appr_meta[
                            (df_appr_meta['NODE_ID'] == node_id) &
                            (df_appr_meta['APPR_ID'] == appr_id)
                        ]
                        appr_name = match.iloc[0]['APPR_NAME'] if not match.empty else "미지정"

                        vehs = int(df_pair['VEHS'].sum(skipna=True) or 0)
                        delay_vals = df_pair['DELAY'].dropna().astype(float).tolist()
                        delay_avg = round(sum(delay_vals) / len(delay_vals), 1) if delay_vals else 0.0
                        los = get_los(delay_avg)

                        items.append({
                            "appr_id": int(appr_id) if pd.notna(appr_id) else None,
                            "appr_name": appr_name,
                            "direction": int(direction) if pd.notna(direction) else None,
                            "vehs": vehs,
                            "delay": delay_avg,
                            "los": los
                        })

                # (2) 접근로 합성: 세 방향 집계 (appr_id 단위)
                appr_summary = []
                if not df_time.empty:
                    for appr_id, df_ap in df_time.groupby('APPR_ID'):
                        match = df_appr_meta[
                            (df_appr_meta['NODE_ID'] == node_id) &
                            (df_appr_meta['APPR_ID'] == appr_id)
                        ]
                        appr_name = match.iloc[0]['APPR_NAME'] if not match.empty else "미지정"

                        vehs_sum = int(df_ap['VEHS'].sum(skipna=True) or 0)
                        dvals = df_ap['DELAY'].dropna().astype(float).tolist()
                        delay_avg3 = round(sum(dvals) / len(dvals), 1) if dvals else 0.0  # 단순 평균
                        los3 = get_los(delay_avg3)

                        appr_summary.append({
                            "appr_id": int(appr_id) if pd.notna(appr_id) else None,
                            "appr_name": appr_name,
                            "vehs_sum": vehs_sum,
                            "delay_avg": delay_avg3,
                            "los": los3
                        })

                # (3) 구간 총괄: 모든 접근·방향 합성
                if not df_time.empty:
                    total_vehs_slice = int(df_time['VEHS'].sum(skipna=True) or 0)
                    all_d = df_time['DELAY'].dropna().astype(float).tolist()
                    avg_delay_slice = round(sum(all_d) / len(all_d), 1) if all_d else 0.0
                    los_slice = get_los(avg_delay_slice)
                else:
                    total_vehs_slice = 0
                    avg_delay_slice = 0.0
                    los_slice = get_los(avg_delay_slice)

                result_obj["time_slices"].append({
                    "timeint": slice_label,        # '00-15', '15-30', '30-45', '45-00'
                    "items": items,                # 원시(APPR×DIRECTION)
                    "appr_summary": appr_summary,  # ✅ 세 방향 집계(접근로 단위)
                    "slice_summary": {             # ✅ 전체(우측 패널)
                        "total_vehs": total_vehs_slice,
                        "avg_delay": avg_delay_slice,
                        "los": los_slice
                    }
                })

            nodes.append(result_obj)

        if not nodes:
            return jsonify({
                "status": "fail",
                "message": f"{label}에 해당하는 데이터가 없습니다.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        # 최종 payload
        payload = {
            "status": "success",
            "label": label,
            "target_date": latest_date,
            "data": nodes
        }

        # ---- ETag / X-Next-Update / X-Dataset-Date ----
        body = json.dumps(payload, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'

        now_kst = datetime.now(KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/","").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = latest_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = latest_date
        resp.headers["X-Next-Update"] = x_next_update
        return resp

    except Exception as e:
        return jsonify({
            "status": "fail",
            "message": "노드 접근 결과 조회 중 오류 발생",
            "error": str(e)
        }), 500









# ========================================================= [ 교통관리 1 - 교통량 패턴비교 분석정보 ]

@app.route('/management/compare-traffic-vol', methods=['GET'])
def compare_traffic_vol():
    pass

# ========================================================= [ 교통관리 2 - Deep Learning Progress Overview ]

@app.route('/management/deep-learning-overview', methods=['GET'])
def deep_learning_overview():
    pass

# ========================================================= [ 교통관리 3 - SA(Sub Area) 그룹 관리정보 ]

@app.route('/management/sa-group-info', methods=['GET'])
def congested_info():
    pass

def sa_info():
    pass

# ========================================================= [ 교통관리 4 - 혼잡교차로 신호최적화 효과검증 ]

@app.route('/management/cross-optimize', methods=['GET'])
def cross_optimize():
    pass









# ========================================================= [ 서버실행 ]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101, debug=False)