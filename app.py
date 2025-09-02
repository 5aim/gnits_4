import pyodbc, os, subprocess, pathlib, json, hashlib
import pandas as pd
import numpy as np
import pytz

from pprint import pprint
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
    1: "교동",
    2: "송정",
    3: "도심",
    4: "경포"
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

# ========================================================= [ 메인페이지 ]

@app.route('/video_test', methods=['GET'])
def video_test():
    return render_template('video_test.html')

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
    now_kst = datetime.now(KST)  # 참고용
    # rule_date = resolve_dataset_date(now_kst)  # ▶ 배포 시 복원
    rule_date = "20250701"  # ▶ 테스트용

    # 공통 헤더 계산(다음 업데이트 시각: 매일 06:00 KST)
    next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
    if now_kst >= next_update:
        next_update += timedelta(days=1)
    x_next_update = next_update.isoformat()

    # 0) 쿼리 파라미터(hour) 검증 — 400 응답 분리
    ALLOWED_HOURS = {"08", "11", "14", "17", "24"}
    hour_param = request.args.get('hour', '').strip()
    if not hour_param:
        return jsonify({"error": "Missing 'hour' query parameter.",
                        "allowed": sorted(list(ALLOWED_HOURS))}), 400
    if hour_param not in ALLOWED_HOURS:
        return jsonify({"error": f"Invalid hour '{hour_param}'.",
                        "allowed": sorted(list(ALLOWED_HOURS))}), 400

    # ✅ make_etag가 정수 시퀀스를 기대하므로, int로 변환해 1-튜플로 전달
    try:
        hours_key_for_etag = (int(hour_param),)
    except ValueError:
        hours_key_for_etag = (24,) if hour_param == "24" else (0,)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # 1) 분기: 시간대 vs 일(day)
        if hour_param == "24":
            target_key = rule_date  # yyyymmdd
            sql_main = """
                SELECT LINK_ID, VC
                FROM TOMMS.DAY_LINK_RESULT
                WHERE STAT_DAY = ?
                ORDER BY LINK_ID
            """
            main_param = (rule_date,)

            # ── hour_label 계산: mm월 dd일 00시 ~ 24시
            date_source = rule_date + "00"  # yyyymmddhh 형태로 mm, dd 추출용
            mm = int(date_source[4:6])
            dd = int(date_source[6:8])
            hour_label = f"{mm}월 {dd}일 전일 평균"

        else:
            stat_hour = rule_date + hour_param  # 예: 2025070108
            target_key = stat_hour              # yyyymmddhh
            sql_main = """
                SELECT LINK_ID, VC
                FROM TOMMS.HOUR_LINK_RESULT
                WHERE STAT_HOUR = ?
                ORDER BY LINK_ID
            """
            main_param = (stat_hour,)

            # ── hour_label 계산: mm월 dd일 hh시 ~ (hh+1)시
            date_source = stat_hour             # yyyymmddhh
            mm = int(date_source[4:6])
            dd = int(date_source[6:8])
            start_h = int(hour_param)
            end_h = (start_h + 1) % 24
            hour_label = f"{mm}월 {dd}일 {start_h:02d}시 ~ {end_h:02d}시"

        cursor.execute(sql_main, main_param)
        rows = cursor.fetchall()

        # 2) 그룹/ID 수집 + per-id VC 누적
        groups = []
        all_ids_in_order = []
        from collections import defaultdict
        vc_list_by_id = defaultdict(list)

        for raw_link, vc_val in rows:
            if not raw_link:
                continue
            ids = [x.strip() for x in str(raw_link).split(",") if x.strip()]
            if not ids:
                continue

            try:
                vc_num = None if vc_val is None else float(vc_val)
            except Exception:
                vc_num = None

            groups.append({"raw": ",".join(ids), "ids": ids})
            all_ids_in_order.extend(ids)

            if vc_num is not None:
                for lid in ids:
                    vc_list_by_id[lid].append(vc_num)

        # 빈 결과 404 (정책상 ETag 없음)
        if not groups:
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        # --- ETag / If-None-Match (민감도: len(groups)만 반영) ---
        etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=len(groups))}"'
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/","").strip()

        if inm == etag.strip('"'):
            payload = {
                "status": "not_modified",
                "message": "Resource not modified. Use cached response.",
                "target_date": rule_date
            }
            body = json.dumps(payload, ensure_ascii=False)
            resp = Response(body, content_type="application/json; charset=utf-8", status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp
        # ---------------------------------------------------------

        # 3) LINK_VERTEX 일괄 조회(IN)
        unique_ids = list(dict.fromkeys(all_ids_in_order).keys())
        placeholders = ",".join(["?"] * len(unique_ids))
        sql_vertex = f"""
            SELECT LINK_ID, LINK_SEQ, WGS84_X, WGS84_Y
            FROM TOMMS.LINK_VERTEX
            WHERE LINK_ID IN ({placeholders})
            ORDER BY LINK_ID, LINK_SEQ
        """
        cursor.execute(sql_vertex, tuple(unique_ids))
        vrows = cursor.fetchall()
        if not vrows:
            return jsonify({"status": "fail", "message": "NODE_INFO 데이터가 없습니다."}), 404

        # 4) 개별 link_id → 좌표 목록 매핑
        coords_by_id = defaultdict(list)
        for link_id, link_seq, x, y in vrows:
            fx = float(x) if x is not None else None
            fy = float(y) if y is not None else None
            coords_by_id[str(link_id)].append([fx, fy])

        # 4-1) per-id VC 평균
        vc_by_id_avg = {}
        for lid, arr in vc_list_by_id.items():
            if arr:
                vc_by_id_avg[lid] = sum(arr) / len(arr)

        # 5) 문서 구성
        documents = []
        doc_vcs_for_avg = []
        for g in groups:
            merged_coords = []
            per_id_vcs = []
            for lid in g["ids"]:
                merged_coords.extend(coords_by_id.get(lid, []))
                if lid in vc_by_id_avg:
                    per_id_vcs.append(vc_by_id_avg[lid])

            if per_id_vcs:
                group_vc = round(sum(per_id_vcs) / len(per_id_vcs), 2)
                doc_vcs_for_avg.append(group_vc)
            else:
                group_vc = None

            documents.append({
                "link_id": g["raw"],
                "coordinates": merged_coords,
                "v/c": group_vc
            })

        # 6) 전체 평균 v/c
        avg_vc = round(sum(doc_vcs_for_avg) / len(doc_vcs_for_avg), 2) if doc_vcs_for_avg else 0.00

        final_json = {
            "v/c": avg_vc,
            "hour_label": hour_label,
            "traffic_vol": 9999,
            "documents": documents
        }

        # 본문 + 헤더
        body = json.dumps(final_json, ensure_ascii=False)
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
        finally:
            conn.close()

# 완료 ========================================================= [ 모니터링 2 - 교통존간 통행정보 ]

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

            # ---- ⬇️ from_zone 좌표 보정 로직: points.features 전체 기준 bounding-box의 중심 ----
            # None 값 제거 후 lat/lon 각각의 min/max 계산
            all_lats = []
            all_lons = []
            for feat in points["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [])
                if not coords or len(coords) < 2:
                    continue
                lat, lon = coords[0], coords[1]  # 현재 구조: [lat, lon]
                if lat is not None and lon is not None:
                    all_lats.append(lat)
                    all_lons.append(lon)

            if all_lats and all_lons:
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                mid_lat = round((min_lat + max_lat) / 2.0, 4)
                mid_lon = round((min_lon + max_lon) / 2.0, 4)
                from_coords = [mid_lat, mid_lon]     # [lat, lon]
            else:
                # 안전가드: 계산 불가 시 원래 좌표 사용
                from_coords = [
                    g["from_lat"] if g["from_lat"] is not None else 0.0,
                    g["from_lon"] if g["from_lon"] is not None else 0.0
                ]

            payload.append({
                "from_zone": {
                    "id": g["from_zone_id"],
                    "name": g["from_zone_name"],
                    "coordinates": from_coords  # 새로 계산된 센터 좌표 [lat, lon]
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

# ========================================================= [ 모니터링 4K - 분석지역별 교통흐름 통계정보 ]

@app.route('/monitoring/statistics-traffic-flow', methods=['GET'])
def statistics_traffic_flow():
    try:
        # --- 0) 날짜 고정(테스트) ---
        now_kst = datetime.now(KST)  # 참고용
        # rule_date = resolve_dataset_date(now_kst)  # ▶ 배포 시 복원
        rule_date = "20250701"  # ▶ 테스트용 (YYYYMMDD)

        # --- 공통 헤더: 다음 업데이트 시각(매일 06:00 KST) ---
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # --- 0-1) hour 쿼리 파라미터 파싱/검증 ---
        ALLOWED_HOURS = {"08", "11", "14", "17"}
        hours_raw = (request.args.get('hour') or '').strip()
        hours_filter = None
        if hours_raw:
            parts = [p.strip() for p in hours_raw.split(",") if p.strip()]
            invalid = [p for p in parts if p not in ALLOWED_HOURS]
            if invalid:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid hour value(s): {', '.join(invalid)}",
                    "allowed": sorted(list(ALLOWED_HOURS))
                }), 400
            hours_filter = set(parts)  # {"08","11"} 등

        # ✅ ETag용 키를 정수 튜플로 정규화 (예: {"08","11"} -> (8,11))
        if hours_filter:
            hours_key_for_etag = tuple(sorted(int(h) for h in hours_filter))
        else:
            hours_key_for_etag = None

        conn = get_connection()
        cursor = conn.cursor()

        # ============================================================
        # A) documents 구성: HOUR_LINK_RESULT + LINK_VERTEX (시간 필터 반영)
        # ============================================================
        cursor.execute("""
            SELECT STAT_HOUR, LINK_ID, DISTRICT, SA_NO, VC, VOLUME, SPEED
            FROM HOUR_LINK_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows] if rows else []
        if not rows:
            # 원자료 없음 → 404 (정책상 ETag 없음)
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        df_result = pd.DataFrame(
            rows,
            columns=["STAT_HOUR", "LINK_ID", "DISTRICT", "SA_NO", "VC", "VOLUME", "SPEED"]
        )

        # 시간 필터 적용(요청 시)
        if hours_filter:
            # STAT_HOUR: yyyymmddHH → 뒤 2자리(HH) 기준 필터
            df_result = df_result[df_result["STAT_HOUR"].str[-2:].isin(hours_filter)]

        # 빈 결과 204 + ETag
        if df_result.empty:
            etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=0)}"'
            resp = Response(status=204)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # ETag / If-None-Match
        etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=len(df_result))}"'
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
        # -----------------------------------------------------------

        # 숫자형 보정(문자열은 건드리지 않음)
        df_result["SPEED"] = pd.to_numeric(df_result["SPEED"], errors="coerce")
        df_result["VC"] = pd.to_numeric(df_result["VC"], errors="coerce")
        df_result["VOLUME"] = pd.to_numeric(df_result["VOLUME"], errors="coerce")

        # LINK_ID 목록 추출(순서 보존 중복제거)
        all_ids_in_order = [str(x).strip() for x in df_result["LINK_ID"].tolist() if str(x).strip()]
        unique_ids = list(dict.fromkeys(all_ids_in_order).keys())

        # LINK_VERTEX 일괄 조회(IN)
        placeholders = ",".join(["?"] * len(unique_ids))
        sql_vertex = f"""
            SELECT LINK_ID, LINK_SEQ, WGS84_X, WGS84_Y
            FROM LINK_VERTEX
            WHERE LINK_ID IN ({placeholders})
            ORDER BY LINK_ID, LINK_SEQ
        """
        cursor.execute(sql_vertex, tuple(unique_ids))
        vrows = cursor.fetchall()
        if not vrows:
            # 보조 테이블 없음 → 404 (정책상 ETag 없음)
            return jsonify({"status": "fail", "message": "LINK_VERTEX 데이터가 없습니다."}), 404

        # 좌표 매핑(link_id → [[x,y], ...])  [x=lon, y=lat]
        from collections import defaultdict
        coords_by_id = defaultdict(list)
        for link_id, link_seq, x, y in vrows:
            fx = float(x) if x is not None else None
            fy = float(y) if y is not None else None
            coords_by_id[str(link_id)].append([fx, fy])

        # --- 링크별 요약 ---
        agg_df = (
            df_result.groupby("LINK_ID").agg(
                vc_avg=("VC", "mean"),
                volume_sum=("VOLUME", "sum"),
                speed_avg=("SPEED", "mean")
            ).reset_index()
        )
        vc_by_id = {str(r["LINK_ID"]): (None if pd.isna(r["vc_avg"]) else round(float(r["vc_avg"]), 2))
                    for _, r in agg_df.iterrows()}
        vol_by_id = {str(r["LINK_ID"]): (None if pd.isna(r["volume_sum"]) else int(r["volume_sum"]))
                     for _, r in agg_df.iterrows()}
        spd_by_id = {str(r["LINK_ID"]): (None if pd.isna(r["speed_avg"]) else round(float(r["speed_avg"]), 2))
                     for _, r in agg_df.iterrows()}

        # --- LINK_ID → DISTRICT 매핑 (첫 등장값 기준) ---
        df_link_dist = df_result[["LINK_ID", "DISTRICT"]].copy()
        df_link_dist["DISTRICT"] = pd.to_numeric(df_link_dist["DISTRICT"], errors="coerce").astype("Int64")
        df_link_dist = df_link_dist.dropna(subset=["DISTRICT"])
        df_link_dist = df_link_dist.drop_duplicates(subset=["LINK_ID"], keep="first")
        link_to_district = {str(row["LINK_ID"]): int(row["DISTRICT"]) for _, row in df_link_dist.iterrows()}

        # --- 구역별 바스켓 4개 생성 ---
        order_codes = [1, 2, 3, 4]
        documents_grouped = [
            {
                "district_no": code,
                "district_name": district_mapping.get(code, str(code)),
                "data": []
            }
            for code in order_codes
        ]
        idx_by_code = {d["district_no"]: i for i, d in enumerate(documents_grouped)}

        # --- 링크 요약을 해당 DISTRICT 바스켓에 채우기 ---
        for lid in unique_ids:
            d_no = link_to_district.get(str(lid))
            if d_no not in idx_by_code:  # DISTRICT가 1~4가 아니면 스킵
                continue
            item = {
                "link_id": str(lid),
                "coordinates": coords_by_id.get(str(lid), []),
                "v/c": vc_by_id.get(str(lid)),
                "volume": vol_by_id.get(str(lid)),
                "speed": spd_by_id.get(str(lid))
            }
            documents_grouped[idx_by_code[d_no]]["data"].append(item)

        # ============================================================
        # B) 일 평균 속도 - daily_average_speed: DAY_LINK_RESULT에서 계산 (파라미터 무관)
        # ============================================================
        cursor.execute("""
            SELECT DISTRICT, SA_NO, SPEED
            FROM DAY_LINK_RESULT
            WHERE STAT_DAY = ?
        """, [rule_date])
        day_rows = cursor.fetchall()
        day_rows = [tuple(r) for r in day_rows] if day_rows else []

        daily_average_speed = []
        if day_rows:
            df_day = pd.DataFrame(day_rows, columns=["DISTRICT", "SA_NO", "SPEED"])
            df_day["SPEED"] = pd.to_numeric(df_day["SPEED"], errors="coerce")
            df_day["DISTRICT"] = pd.to_numeric(df_day["DISTRICT"], errors="coerce").astype("Int64")

            overall_avg_by_dist = df_day.groupby("DISTRICT")["SPEED"].mean()

            sa_present_mask = df_day["SA_NO"].notna() & (df_day["SA_NO"].astype(str).str.strip() != "")
            df_day_sa = df_day[sa_present_mask]
            sa_included_by_dist = (
                df_day_sa.groupby("DISTRICT")["SPEED"].mean()
                if not df_day_sa.empty else pd.Series(dtype=float)
            )

            for code in order_codes:
                name = district_mapping.get(code)
                overall = None
                if code in overall_avg_by_dist.index:
                    o = overall_avg_by_dist.loc[code]
                    overall = None if pd.isna(o) else round(float(o), 2)

                sa_included = None
                if code in sa_included_by_dist.index:
                    s = sa_included_by_dist.loc[code]
                    sa_included = None if pd.isna(s) else round(float(s), 2)

                daily_average_speed.append({
                    "district": name,
                    "daily_average_speed": {
                        "overall": overall,
                        "sa_included": sa_included
                    }
                })
        else:
            for code in order_codes:
                daily_average_speed.append({
                    "district": district_mapping.get(code),
                    "daily_average_speed": {"overall": None, "sa_included": None}
                })

        # ============================================================
        # C) hour_label 생성: 'mm월 dd일 hh시 ~ hh시'
        #    - 단일 시간: 문자열 1개
        #    - 다중/미지정: 해당 시간대 전체의 라벨 리스트
        # ============================================================
        mm = int(rule_date[4:6])
        dd = int(rule_date[6:8])

        def make_label(hh_str: str) -> str:
            start_h = int(hh_str)
            end_h = (start_h + 1) % 24
            return f"{mm}월 {dd}일 {start_h:02d}시 ~ {end_h:02d}시"

        if hours_filter and len(hours_filter) == 1:
            single_hour = sorted(list(hours_filter))[0]
            hour_label_value = make_label(single_hour)  # 문자열
        else:
            # 파라미터 없거나 다중 시간의 경우: 데이터에 실제 존재하는 시간대 기준으로 구성
            if hours_filter:
                candidate_hours = sorted(hours_filter)
            else:
                # df_result에서 실제 존재하는 시간대 추출 (ALLOWED_HOURS와 교집합)
                candidate_hours = sorted(set(df_result["STAT_HOUR"].str[-2:]).intersection(ALLOWED_HOURS))
            hour_label_value = [make_label(h) for h in candidate_hours]  # 리스트

        # ============================================================
        # D) 본문 + 헤더(정상 200에서도 ETag 포함)
        # ============================================================
        payload = {
            "status": "success",
            "hour_label": hour_label_value,
            "row_count": int(len(df_result)),
            "documents": documents_grouped,              # ✅ DISTRICT-그룹 형태
            "daily_average_speed": daily_average_speed,  # 그대로 유지
            "hourly_total_traffic_volume": [
                {"district": "교동", "traffic_volume": 8888},
                {"district": "송정", "traffic_volume": 7777},
                {"district": "도심", "traffic_volume": 9999},
                {"district": "경포", "traffic_volume": 6666}
            ],
            "map_center_coordinates": [
                {"district": "교동", "coordinates": [128.874273, 37.765208]},
                {"district": "송정", "coordinates": [128.924538, 37.771808]},
                {"district": "도심", "coordinates": [128.897176, 37.755575]},
                {"district": "경포", "coordinates": [128.891529, 37.787484]}
            ]
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
    finally:
        try:
            cursor.close()
        finally:
            conn.close()

# ========================================================= [ 모니터링 4 - 교차로별 통행정보 ]

@app.route('/monitoring/node-result', methods=['GET'])
def node_result_summary():
    now_kst = datetime.now(KST)  # 참고용
    # rule_date = resolve_dataset_date(now_kst)  # ▶ 배포 시 복원
    rule_date = "20250701"  # ▶ 테스트용 (YYYYMMDD)

    # 공통 헤더 계산(다음 업데이트 시각: 매일 06:00 KST)
    next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
    if now_kst >= next_update:
        next_update += timedelta(days=1)
    x_next_update = next_update.isoformat()

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 0) 테이블에 존재하는 최신 날짜(YYYYMMDD)
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

        # 1) rule_date 데이터 유무 확인 → 있으면 rule_date 사용, 없으면 latest_date로 폴백
        cursor.execute("""
            SELECT 1
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
              AND ROWNUM = 1
        """, [rule_date])
        has_rule_date = cursor.fetchone() is not None

        active_date = rule_date if has_rule_date else latest_date
        fallback_used = (active_date != rule_date)

        mm, dd = int(active_date[4:6]), int(active_date[6:8])

        # 2) 데이터 조회 (active_date 기준)
        cursor.execute("""
            SELECT STAT_HOUR, TIMEINT, NODE_ID, QLEN, VEHS, DELAY, STOPS
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [active_date])
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

        # 8) 시간 블록 데이터 구성 (active_date의 mm/dd로 라벨 생성)
        data_blocks = []
        for hour, group in df_merged.groupby("HOUR"):
            h = int(hour); h_next = (h + 1) % 24
            hour_label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"

            items = []
            for _, r in group.iterrows():
                qlen  = float(r["qlen"])  if pd.notna(r["qlen"])  else 0.0
                vehs  = int(r["vehs"])    if pd.notna(r["vehs"])  else 0.0
                delay = float(r["delay"]) if pd.notna(r["delay"]) else 0.0
                stops = float(r["stops"]) if pd.notna(r["stops"]) else 0.0
                los   = int(r["LOS_NUM"]) if pd.notna(r["LOS_NUM"]) else None

                items.append({
                    "node_name": str(r["node_name"]),
                    "qlen": f"{qlen:.1f}",
                    "vehs": f"{int(vehs)}",
                    "delay": f"{delay:.1f}",
                    "stops": f"{stops:.1f}",
                    "los": f"{los}",
                    "max_qlen": f"{qlen * 1.5:.1f}",
                    "max_vehs": f"{int(vehs * 1.5)}",
                    "max_delay": f"{delay * 1.5:.1f}",
                    "max_stops": "5",
                    "max_los": "6"
                })

            data_blocks.append({"hour_label": hour_label, "items": items})

        # 9) Payload & ETag
        payload = {
            # "requested_date": rule_date,     # 요청 의도
            # "active_date": active_date,      # 실제 조회에 사용된 날짜
            # "fallback_used": fallback_used,  # rule_date 데이터 없어서 최신으로 대체했는지 여부
            "target_date": latest_date,
            "data": data_blocks
        }
        body = json.dumps(payload, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'

        # 10) 다음 업데이트 시간 (매일 06:00 KST)
        now_kst = datetime.now(KST)
        next_update_kst = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update_kst:
            next_update_kst += timedelta(days=1)

        x_next_update_str = next_update_kst.isoformat()

        # 11) If-None-Match 처리
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/", "").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = active_date
            resp.headers["X-Next-Update"] = x_next_update_str
            return resp

        # 12) 정상 응답
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = active_date
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

# ========================================================= [ 모니터링 5 - 도로구간별 통행량 정보 ]

@app.route('/monitoring/road-traffic-info', methods=['GET'])
def road_traffic_info():
    pass









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

# ========================================================= [ 신호운영 4K - 권역별 시간대별 교통혼잡 정보 ]

@app.route('/signal/district-hourly-congested-info', methods=['GET'])
def hourly_congested_info_data():
    try:
        # --- 0) 날짜 고정(테스트) ---
        now_kst = datetime.now(KST)  # 참고용
        # rule_date = resolve_dataset_date(now_kst)  # ▶ 배포 시 복원
        rule_date = "20250701"  # ▶ 테스트용

        # --- 공통 헤더: 다음 업데이트 시각(매일 06:00 KST) ---
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # --- 0-1) hour 쿼리 파라미터 파싱/검증 ---
        ALLOWED_HOURS = {"08", "11", "14", "17"}
        hours_raw = (request.args.get('hour') or '').strip()
        hours_filter = None
        if hours_raw:
            parts = [p.strip() for p in hours_raw.split(",") if p.strip()]
            invalid = [p for p in parts if p not in ALLOWED_HOURS]
            if invalid:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid hour value(s): {', '.join(invalid)}",
                    "allowed": sorted(list(ALLOWED_HOURS))
                }), 400
            hours_filter = set(parts)  # {"08","11"} 등

        # If-None-Match 수신 (ETag 비교는 응답 직전에 수행)
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/","").strip()

        # ✅ ETag용 키 정규화
        hours_key_for_etag = tuple(sorted(int(h) for h in hours_filter)) if hours_filter else None

        conn = get_connection()
        cursor = conn.cursor()

        # ============================================================
        # A) 권역별 분석정보 : NP_RESULT (시간 필터 및 권역별 COST 합계)
        # ============================================================
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, VEHS, COST
            FROM TOMMS.NP_RESULT
            WHERE SUBSTR(TO_CHAR(STAT_HOUR), 1, 8) = ?
        """, [rule_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows] if rows else []
        if not rows:
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        df_np_result = pd.DataFrame(rows, columns=["DISTRICT", "STAT_HOUR", "VEHS", "COST"])

        # 시간 필터 적용(요청 시)
        if hours_filter:
            df_np_result = df_np_result[df_np_result["STAT_HOUR"].astype(str).str[-2:].isin(hours_filter)]

        # 빈 결과 204
        if df_np_result.empty:
            etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=0)}"'
            resp = Response(status=204)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # ---- 타입 보정 (A)
        df_np_result["DISTRICT"] = pd.to_numeric(df_np_result["DISTRICT"], errors="coerce")
        df_np_result["COST"] = pd.to_numeric(df_np_result["COST"], errors="coerce")
        df_np_result["STAT_HOUR"] = df_np_result["STAT_HOUR"].astype(str).str[:10]  # 'YYYYMMDDHH'
        df_np_result["HH"] = df_np_result["STAT_HOUR"].str[-2:]
        df_np_result["MM"] = df_np_result["STAT_HOUR"].str[4:6]
        df_np_result["DD"] = df_np_result["STAT_HOUR"].str[6:8]

        # 동일 키 중복 방지: 시간/권역별 COST 합
        agg_a = (
            df_np_result
            .groupby(["STAT_HOUR", "HH", "MM", "DD", "DISTRICT"], dropna=False, as_index=False)["COST"]
            .sum()
        )

        def build_hours_order(present_hours, hours_filter):
            if hours_filter:
                return [h for h in ["08", "11", "14", "17"] if h in hours_filter]
            return [h for h in ["08", "11", "14", "17"] if h in present_hours]

        hours_order_a = build_hours_order(set(agg_a["HH"].unique().tolist()), hours_filter)

        def make_hour_label(mm: str, dd: str, hh: str) -> str:
            hs = int(hh)
            he = (hs + 1) % 24
            return f"{mm}월 {dd}일 {hs:02d}시 ~ {he:02d}시"

        # ---- documents 기본 골격 생성(A: district_data)
        documents = []
        doc_by_hh = {}
        for hh in hours_order_a:
            sub = agg_a[agg_a["HH"] == hh]
            if sub.empty:
                continue
            mm = sub.iloc[0]["MM"]
            dd = sub.iloc[0]["DD"]
            hour_label = make_hour_label(mm, dd, hh)

            district_data = []
            for dno in [1, 2, 3, 4]:
                row = sub[sub["DISTRICT"] == dno]
                cost_val = float(row.iloc[0]["COST"]) if not row.empty and pd.notna(row.iloc[0]["COST"]) else None
                district_data.append({
                    "district_no": dno,
                    "district": district_mapping.get(dno, str(dno)),
                    "cost": cost_val
                })

            doc = {"hour_label": hour_label, "district_data": district_data}
            documents.append(doc)
            doc_by_hh[hh] = doc  # 이후 B 섹션에서 시간대 매칭 시 활용

        # ============================================================
        # B-1) 도로(ROAD_NAME) 기준 평균: HOUR_LINK_RESULT
        #      - AVG(VOLUME), AVG(VC)
        #      - UPDOWN 무시
        # ============================================================
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, ROAD_NAME,
                   AVG(VOLUME) AS VOLUME_AVG,
                   AVG(VC)     AS VC_AVG
            FROM TOMMS.HOUR_LINK_RESULT
            WHERE SUBSTR(TO_CHAR(STAT_HOUR), 1, 8) = ?
            GROUP BY DISTRICT, STAT_HOUR, ROAD_NAME
        """, [rule_date])
        link_rows = cursor.fetchall()
        link_rows = [tuple(r) for r in link_rows] if link_rows else []
        df_link = pd.DataFrame(
            link_rows,
            columns=["DISTRICT", "STAT_HOUR", "ROAD_NAME", "VOLUME_AVG", "VC_AVG"]
        ) if link_rows else pd.DataFrame(columns=["DISTRICT","STAT_HOUR","ROAD_NAME","VOLUME_AVG","VC_AVG"])

        if not df_link.empty:
            if hours_filter:
                df_link = df_link[df_link["STAT_HOUR"].astype(str).str[-2:].isin(hours_filter)]
            df_link["DISTRICT"]   = pd.to_numeric(df_link["DISTRICT"], errors="coerce")
            df_link["VOLUME_AVG"] = pd.to_numeric(df_link["VOLUME_AVG"], errors="coerce")
            df_link["VC_AVG"]     = pd.to_numeric(df_link["VC_AVG"], errors="coerce")
            df_link["STAT_HOUR"]  = df_link["STAT_HOUR"].astype(str).str[:10]
            df_link["HH"] = df_link["STAT_HOUR"].str[-2:]
            df_link["MM"] = df_link["STAT_HOUR"].str[4:6]
            df_link["DD"] = df_link["STAT_HOUR"].str[6:8]

            hours_order_b1 = build_hours_order(set(df_link["HH"].unique().tolist()), hours_filter)
            for hh in hours_order_b1:
                if hh not in doc_by_hh:
                    sub_any = df_link[df_link["HH"] == hh]
                    if sub_any.empty:
                        continue
                    mm = sub_any.iloc[0]["MM"]; dd = sub_any.iloc[0]["DD"]
                    doc = {"hour_label": make_hour_label(mm, dd, hh), "district_data": []}
                    documents.append(doc)
                    doc_by_hh[hh] = doc

            # 시간대별 road_data 생성/부착
            for hh, doc in doc_by_hh.items():
                sub = df_link[df_link["HH"] == hh]
                if sub.empty:
                    doc["road_data"] = []
                    continue
                road_data = []
                for dno in [1, 2, 3, 4]:
                    part = sub[sub["DISTRICT"] == dno].copy()
                    if part.empty:
                        road_data.append({
                            "district_no": dno,
                            "district": district_mapping.get(dno, str(dno)),
                            "roads": []
                        })
                        continue
                    part = part.sort_values(["ROAD_NAME"]).reset_index(drop=True)
                    roads = []
                    for _, r in part.iterrows():
                        roads.append({
                            "road_name": (str(r["ROAD_NAME"]) if pd.notna(r["ROAD_NAME"]) else None),
                            "volume_avg": (float(r["VOLUME_AVG"]) if pd.notna(r["VOLUME_AVG"]) else None),
                            "vc_avg":     (float(r["VC_AVG"]) if pd.notna(r["VC_AVG"]) else None),
                        })
                    road_data.append({
                        "district_no": dno,
                        "district": district_mapping.get(dno, str(dno)),
                        "roads": roads
                    })
                doc["road_data"] = road_data

        # ============================================================
        # B-2) 축(SA_NO) 기준 평균: HOUR_LINK_RESULT
        #      - AVG(VC) -> 소수점 2자리
        #      - AVG(SPEED) -> 소수점 1자리
        #      - UPDOWN 무시
        # ============================================================
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, SA_NO,
                   AVG(VC)    AS VC_AVG,
                   AVG(SPEED) AS SPEED_AVG
            FROM TOMMS.HOUR_LINK_RESULT
            WHERE SUBSTR(TO_CHAR(STAT_HOUR), 1, 8) = ?
            GROUP BY DISTRICT, STAT_HOUR, SA_NO
        """, [rule_date])
        sa_rows = cursor.fetchall()
        sa_rows = [tuple(r) for r in sa_rows] if sa_rows else []
        df_sa = pd.DataFrame(
            sa_rows,
            columns=["DISTRICT", "STAT_HOUR", "SA_NO", "VC_AVG", "SPEED_AVG"]
        ) if sa_rows else pd.DataFrame(columns=["DISTRICT","STAT_HOUR","SA_NO","VC_AVG","SPEED_AVG"])

        if not df_sa.empty:
            if hours_filter:
                df_sa = df_sa[df_sa["STAT_HOUR"].astype(str).str[-2:].isin(hours_filter)]
            df_sa["DISTRICT"]  = pd.to_numeric(df_sa["DISTRICT"], errors="coerce")
            df_sa["VC_AVG"]    = pd.to_numeric(df_sa["VC_AVG"], errors="coerce")
            df_sa["SPEED_AVG"] = pd.to_numeric(df_sa["SPEED_AVG"], errors="coerce")
            df_sa["STAT_HOUR"] = df_sa["STAT_HOUR"].astype(str).str[:10]
            df_sa["HH"] = df_sa["STAT_HOUR"].str[-2:]
            df_sa["MM"] = df_sa["STAT_HOUR"].str[4:6]
            df_sa["DD"] = df_sa["STAT_HOUR"].str[6:8]

            hours_order_b2 = build_hours_order(set(df_sa["HH"].unique().tolist()), hours_filter)
            for hh in hours_order_b2:
                if hh not in doc_by_hh:
                    sub_any = df_sa[df_sa["HH"] == hh]
                    if sub_any.empty:
                        continue
                    mm = sub_any.iloc[0]["MM"]; dd = sub_any.iloc[0]["DD"]
                    doc = {"hour_label": make_hour_label(mm, dd, hh), "district_data": []}
                    documents.append(doc)
                    doc_by_hh[hh] = doc

            # 반올림 유틸
            def round_or_none(x, nd):
                return round(float(x), nd) if (x is not None and pd.notna(x)) else None

            # 시간대별 sa_data 생성/부착
            for hh, doc in doc_by_hh.items():
                sub = df_sa[df_sa["HH"] == hh]
                if sub.empty:
                    doc["sa_data"] = []
                    continue
                sa_data = []
                for dno in [1, 2, 3, 4]:
                    part = sub[sub["DISTRICT"] == dno].copy()
                    if part.empty:
                        sa_data.append({
                            "district_no": dno,
                            "district": district_mapping.get(dno, str(dno)),
                            "segments": []
                        })
                        continue
                    # SA_NO 정렬
                    part = part.sort_values(["SA_NO"]).reset_index(drop=True)
                    segments = []
                    for _, r in part.iterrows():
                        segments.append({
                            "sa_no": (str(r["SA_NO"]) if pd.notna(r["SA_NO"]) else None),
                            "vc_avg":   round_or_none(r["VC_AVG"], 2),  # 소수점 2자리
                            "speed_avg": round_or_none(r["SPEED_AVG"], 1)  # 소수점 1자리
                        })
                    sa_data.append({
                        "district_no": dno,
                        "district": district_mapping.get(dno, str(dno)),
                        "segments": segments
                    })
                doc["sa_data"] = sa_data

        # ---- ETag 계산(A+B1+B2 합산)
        total_rows = len(agg_a) + (len(df_link) if not df_link.empty else 0) + (len(df_sa) if not df_sa.empty else 0)
        etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=total_rows)}"'
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        payload = {
            "status": "success",
            "rule_date": rule_date,     # 'YYYYMMDD'
            # documents[*] = {
            #   hour_label,
            #   district_data: [{district_no, district, cost} ×4],
            #   road_data: [{district_no, district, roads: [{road_name, volume_avg, vc_avg}...]} ×4],
            #   sa_data:   [{district_no, district, segments: [{sa_no, vc_avg(2), speed_avg(1)}...]} ×4]
            # }
            "documents": documents
        }

        resp = jsonify(payload)
        resp.headers["ETag"] = etag
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp, 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            cursor.close()
        finally:
            conn.close()

# ========================================================= [ 신호운영 4 - 교차로별 효과지표 분석정보 ]

def _run_sql(cursor, sql, params=None, step=""):
    params = params or []
    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except Exception as e:
        # 서버 로그에 상세 남기기
        print(f"[DB-ERROR] step={step}\nSQL=\n{sql}\nparams={params}\nexc={repr(e)}")
        # 호출자에게 에러 전달
        raise

@app.route('/signal/node-approach-result', methods=['GET'])
def node_approach_result():
    hour_filter = (request.args.get('hour') or '').strip()  # '08','11','14','17' 등
    if not (len(hour_filter) == 2 and hour_filter.isdigit() and 0 <= int(hour_filter) <= 23):
        return jsonify({
            "status": "fail",
            "message": f"유효하지 않은 hour 파라미터입니다: {hour_filter}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 400

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # -------------------------------------------- 1) 최신 일자 (단순화: MAX 사용)
        sql_latest = """
            SELECT MAX(SUBSTR(STAT_HOUR, 1, 8)) AS STAT_DATE
            FROM NODE_DIR_RESULT
        """
        latest_rows = _run_sql(cursor, sql_latest, step="latest_date")
        latest_date_row = latest_rows[0] if latest_rows else None
        if not latest_date_row or not latest_date_row[0]:
            return jsonify({
                "status": "fail",
                "message": "NODE_DIR_RESULT에 STAT_HOUR 데이터가 없습니다.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        latest_date = latest_date_row[0]  # 'YYYYMMDD'
        mm, dd = int(latest_date[4:6]), int(latest_date[6:8])
        h = int(hour_filter); h_next = (h + 1) % 24
        label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"

        # -------------------------------------------- 2) 최신 일자 + 해당 시(hour)만 DB에서 바로 필터
        sql_rows = """
            SELECT STAT_HOUR, TIMEINT, NODE_ID, SA_NO,
                APPR_ID, DIRECTION, QLEN, VEHS, DELAY, STOPS
            FROM TOMMS.NODE_DIR_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
            AND SUBSTR(STAT_HOUR, -2, 2) = ?
        """
        rows = [tuple(r) for r in cursor.execute(sql_rows, [latest_date, hour_filter]).fetchall()]
        cols = ['STAT_HOUR','TIMEINT','NODE_ID','SA_NO','APPR_ID','DIRECTION','QLEN','VEHS','DELAY','STOPS']
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return jsonify({
                "status": "fail",
                "message": f"{label}에 해당하는 데이터가 없습니다.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        # 숫자 변환
        for col in ["APPR_ID","DIRECTION","QLEN","VEHS","DELAY","STOPS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        ORDERED_SLICES = ['00-15','15-30','30-45','45-00']
        df = df[df['TIMEINT'].isin(ORDERED_SLICES)].copy()

        # -------------------------------------------- 3) NODE_DIR_INFO 조회
        sql_info = """
            SELECT CROSS_ID, DISTRICT, NODE_ID, NODE_NAME, CROSS_TYPE, INT_TYPE,
                APPR_ID, DIRECTION, APPR_NAME
            FROM TOMMS.NODE_DIR_INFO
        """
        info_rows = [tuple(r) for r in cursor.execute(sql_info).fetchall()]
        info_cols = ['CROSS_ID','DISTRICT','NODE_ID','NODE_NAME','CROSS_TYPE','INT_TYPE',
                    'APPR_ID','DIRECTION','APPR_NAME']
        df_info = pd.DataFrame(info_rows, columns=info_cols)
        for col in ["CROSS_ID","APPR_ID","DIRECTION","CROSS_TYPE"]:
            if col in df_info.columns:
                df_info[col] = pd.to_numeric(df_info[col], errors="coerce")

        # ✅ 여기서 merge하여 DISTRICT, CROSS_ID를 부여
        df = df.merge(df_info[['NODE_ID','CROSS_ID','DISTRICT']], on='NODE_ID', how='left')

        # 메타 프레임(기존 로직 유지)
        df_node_meta = df_info.drop_duplicates(subset=['NODE_ID'])[['NODE_ID','NODE_NAME','CROSS_TYPE','INT_TYPE']].set_index('NODE_ID')
        df_appr_meta = df_info[['NODE_ID','APPR_ID','DIRECTION','APPR_NAME']].dropna()
        if 'APPR_ID' in df_appr_meta.columns:
            df_appr_meta['APPR_ID'] = pd.to_numeric(df_appr_meta['APPR_ID'], errors="coerce")

        # -------------------------------------------- 4) 일일 총 교통량 맵
        
        unique_cross_ids = sorted({int(c) for c in df['CROSS_ID'].dropna().astype(int).tolist()})
        daily_volume_map = {}
        if unique_cross_ids:
            placeholders = ",".join(["?"] * len(unique_cross_ids))
            query = f"""
                SELECT CROSS_ID, VOL
                FROM TOMMS.STAT_DAY_CROSS
                WHERE STAT_DAY = ?
                AND INFRA_TYPE = 'SMT'
                AND CROSS_ID IN ({placeholders})
            """
            params = [latest_date] + unique_cross_ids
            cursor.execute(query, params)
            for cross_id_val, vol_val in cursor.fetchall():
                c = int(cross_id_val)
                v = 0
                if vol_val is not None:
                    try:
                        v = int(vol_val)
                    except Exception:
                        try:
                            v = int(float(vol_val))
                        except Exception:
                            v = 0
                daily_volume_map[c] = v
        
        # -------------------------------------------- 5) 최신 신호주기 매핑

        signal_cycle_map = {}

        # df에는 이미 NODE_DIR_RESULT + NODE_DIR_INFO merge로 CROSS_ID가 있음
        try:
            unique_cross_ids = sorted({int(c) for c in df['CROSS_ID'].dropna().astype(int).tolist()})
        except Exception:
            unique_cross_ids = []

        if unique_cross_ids:
            placeholders = ",".join(["?"] * len(unique_cross_ids))
            # ROW_NUMBER()로 cross_id마다 최신(INT_CREDATE DESC) 1건만 선택
            sql_cycle = f"""
                SELECT INT_LCNO, INT_CYCLE
                FROM (
                    SELECT INT_LCNO, INT_CYCLE, INT_CREDATE,
                        ROW_NUMBER() OVER (PARTITION BY INT_LCNO ORDER BY INT_CREDATE DESC) AS RN
                    FROM ITS_SCS_L_OPER
                    WHERE INT_LCNO IN ({placeholders})
                ) T
                WHERE T.RN = 1
            """
            params = unique_cross_ids
            cursor.execute(sql_cycle, params)
            for lcno, cycle in cursor.fetchall():
                # 키: cross_id(INT_LCNO), 값: 신호주기(INT_CYCLE), 안전 변환
                try:
                    key = int(lcno)
                except Exception:
                    continue
                val = None
                if cycle is not None:
                    try:
                        val = int(cycle)
                    except Exception:
                        try:
                            val = int(float(cycle))
                        except Exception:
                            val = None
                if val is not None and val > 0:
                    signal_cycle_map[key] = val

        # -------------------------------------------- 6) 결과 가공
        
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

            # [변경] daily_total_vehs 할당
            
            daily_total_val = 0
            if pd.notna(cross_id):
                try:
                    daily_total_val = daily_volume_map.get(int(cross_id), 0)
                except Exception:
                    daily_total_val = 0
            
            # -------------------------------------------- 최신 신호주기 반영
            
            signal_cycle_val = 150
            if pd.notna(cross_id):
                try:
                    signal_cycle_val = signal_cycle_map.get(int(cross_id), 150)
                except Exception:
                    signal_cycle_val = 150

            result_obj = {
                "node_name": node_name,
                "cross_id": int(cross_id) if pd.notna(cross_id) else None,
                "sa_no": sa_no,
                "cross_type": int(node_meta['CROSS_TYPE']) if pd.notna(node_meta['CROSS_TYPE']) else None,
                "int_type": node_meta['INT_TYPE'],
                "daily_total_vehs": daily_total_val,
                "total_vehs": all_vehs_total,
                "total_delay": total_delay,
                "total_los": total_los,
                "signal_circle": signal_cycle_val,
                "hourly": hourly_items,
                "time_slices": []
            }

            # -------------------------------------------- 구간별 time_slices(4개 고정)
            for slice_label in ORDERED_SLICES:
                df_time = df_node[df_node['TIMEINT'] == slice_label].copy()

                # -------------------------------------------- (1) 원시 items: APPR_ID × DIRECTION
                
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

                # -------------------------------------------- (2) 접근로 합성
                
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
                        delay_avg3 = round(sum(dvals) / len(dvals), 1) if dvals else 0.0
                        los3 = get_los(delay_avg3)

                        appr_summary.append({
                            "appr_id": int(appr_id) if pd.notna(appr_id) else None,
                            "appr_name": appr_name,
                            "vehs_sum": vehs_sum,
                            "delay_avg": delay_avg3,
                            "los": los3
                        })

                # -------------------------------------------- (3) 구간 총괄
                
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
                    "timeint": slice_label,
                    "items": items,
                    "appr_summary": appr_summary,
                    "slice_summary": {
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

        # -------------------------------------------- 최종 payload
        
        body = json.dumps({
            "status": "success",
            "label": label,
            "target_date": latest_date,
            "data": nodes
        }, ensure_ascii=False)
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

    # -------------------------------------------- 종료

    except Exception as e:
        # 클라이언트에는 고정 메시지 + 간단한 에러 문자열만
        return jsonify({
            "status": "fail",
            "message": "노드 접근 결과 조회 중 오류 발생",
            "error": str(e)
        }), 500
    finally:
        try:
            if conn:
                conn.close()
        except:
            pass









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

@app.route('/management/signal-optimize', methods=['GET'])
def cross_optimize():
    pass









# ========================================================= [ 서버실행 ]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101, debug=False)