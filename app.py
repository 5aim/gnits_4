import pyodbc, os, subprocess, pathlib, json, hashlib, re
import pandas as pd
import numpy as np
import pytz

from pprint import pprint
from decimal import Decimal
from flask_cors import CORS
from dotenv import load_dotenv
from collections import defaultdict
from flask_compress import Compress
from windows import set_dpi_awareness
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, make_response, flash, session









set_dpi_awareness()
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = "change-this"  # flash 메시지용
ADMIN_ID_RE = re.compile(r'^(?=(?:.*[A-Za-z]){5,})[A-Za-z0-9]+$')   # 영문 5자 이상 포함, 영문/숫자만
KOREAN_NAME_RE = re.compile(r'^[가-힣]{2,50}$')                      # 한글 2~50자
CORS(app)
Compress(app)








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

# ========================================================= If-None-Match 값 정규화 (따옴표, W/, :br 등 제거)

def _normalize_inm(value: str) -> str | None:
    if not value:
        return None
    token = value.split(",")[0].strip()
    if token.startswith("W/"):
        token = token[2:].strip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        token = token[1:-1]
    if ":" in token:  # ← ':br', ':gzip' 등 제거
        token = token.split(":", 1)[0]
    return token or None

# =========================================================  모든 응답의 ETag에서 ':br' 같은 접미사를 차단하고 Vary 보장

def _strip_compression_suffix_from_etag(response):
    et = response.headers.get("ETag")
    if not et:
        return response
    # 따옴표 포함 형태: "hash[:suffix]"
    et = et.strip()
    # 따옴표 벗겨서 순수 값만 비교/정리
    if len(et) >= 2 and et[0] == '"' and et[-1] == '"':
        core = et[1:-1]
        if ":" in core:
            core = core.split(":", 1)[0]  # ← ':br' 같은 접미사 제거
        response.headers["ETag"] = f'"{core}"'
    else:
        # 혹시 따옴표 없이 내려오는 비표준 케이스도 방어
        if ":" in et:
            et = et.split(":", 1)[0]
        response.headers["ETag"] = f'"{et}"'
    # 인코딩별 표현 차이는 캐시가 분리되도록
    response.headers.setdefault("Vary", "Accept-Encoding")
    return response

app.after_request_funcs.setdefault(None, [])
app.after_request_funcs[None].insert(0, _strip_compression_suffix_from_etag)



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
    2: "송정지구",
    3: "중심지구",
    4: "경포지구"
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









#  [ 로그인 ]  =========================================================

@app.route('/')
def index():
    if "admin_id" in session:
        return redirect(url_for("home_dashboard"))
    return redirect(url_for("login"))

def require_login():
    if "admin_id" not in session:
        flash("로그인이 필요합니다.")
        return False
    return True

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        # 이미 로그인된 상태라면 홈으로 리다이렉트
        if "admin_id" in session:
            return redirect(url_for("home_dashboard"))
        return render_template("login.html")

    # --- POST ---
    admin_id = (request.form.get("admin_id") or "").strip()
    password = (request.form.get("password") or "").strip()

    if not admin_id or not password:
        flash("아이디와 비밀번호를 모두 입력하세요.")
        return redirect(url_for("login"))

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT ADMIN_ID, PASSWORD, NAME FROM TOMMS.ADMIN_USER WHERE ADMIN_ID = ?",
            (admin_id,)
        )
        row = cur.fetchone()
        if not row:
            flash("존재하지 않는 아이디입니다.")
            return redirect(url_for("login"))

        db_admin_id, db_password_hash, db_name = row
        if check_password_hash(db_password_hash, password):
            session["admin_id"] = db_admin_id
            session["admin_name"] = db_name
            flash(f"{db_name}님, 로그인 성공!")
            return redirect(url_for("home_dashboard"))  # ← 여기 수정 포인트
        else:
            flash("비밀번호가 올바르지 않습니다.")
            return redirect(url_for("login"))
    except Exception:
        flash("로그인 처리 중 문제가 발생했습니다.")
        return redirect(url_for("login"))
    finally:
        if conn:
            conn.close()

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("로그아웃되었습니다.")
    return redirect(url_for("login"))

#  [ 회원가입 ]  =========================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("sign_up.html")

    # --- POST: 입력값 수집 ---
    admin_id = (request.form.get("admin_id") or "").strip()
    password = (request.form.get("password") or "").strip()
    name = (request.form.get("name") or "").strip()

    # --- 1) 서버측 검증 ---
    if not ADMIN_ID_RE.match(admin_id):
        flash("ADMIN_ID는 영문 5자 이상을 포함하고, 영문/숫자만 사용할 수 있습니다.")
        return redirect(url_for("signup"))

    if len(password) < 8:
        flash("비밀번호는 최소 8자 이상이어야 합니다.")
        return redirect(url_for("signup"))

    if not KOREAN_NAME_RE.match(name):
        flash("성함은 한글 2~50자만 가능합니다.")
        return redirect(url_for("signup"))

    # --- 2) 비밀번호 해시 (PBKDF2-SHA256 + salt) ---
    password_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

    # --- 3) 생성시각(YYYYMMDDHH24MISS) ---
    created_at = datetime.now().strftime("%Y%m%d%H%M%S")

    # --- 4) DB 사전 중복 체크 + INSERT + PK 충돌 예외 처리 ---
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # (UX) 사전 중복 체크
        cur.execute("SELECT 1 FROM TOMMS.ADMIN_USER WHERE ADMIN_ID = ?", (admin_id,))
        if cur.fetchone():
            flash("이미 존재하는 ADMIN_ID 입니다.")
            return redirect(url_for("signup"))

        # 실제 INSERT (경쟁 조건 대비 try/except)
        cur.execute(
            """
            INSERT INTO TOMMS.ADMIN_USER (ADMIN_ID, PASSWORD, NAME, CREATED_AT)
            VALUES (?, ?, ?, ?)
            """,
            (admin_id, password_hash, name, created_at)
        )
        conn.commit()

        flash("관리자 계정이 생성되었습니다. 로그인해주세요.")
        return redirect(url_for("login"))

    except pyodbc.IntegrityError as e:
        # Tibero PK/UNIQUE 위반: SQLSTATE '23000' or message contains 'UNIQUE constraint violation'
        if conn:
            conn.rollback()
        msg = str(e)
        if "23000" in msg or "UNIQUE constraint violation" in msg:
            flash("이미 존재하는 ADMIN_ID 입니다.")
        else:
            flash("회원가입 처리 중 제약 조건 위반이 발생했습니다.")
        return redirect(url_for("signup"))

    except Exception as e:
        if conn:
            conn.rollback()
        # 필요 시 서버 로그로 e 출력/기록
        flash("회원가입 처리 중 문제가 발생했습니다.")
        return redirect(url_for("signup"))

    finally:
        if conn:
            conn.close()

#  [ 메인페이지 ]  =========================================================

@app.route("/home")
def home_dashboard():
    if not require_login():
        return redirect(url_for("login"))
    return render_template(
        "home_dashboard.html",
        name=session.get("admin_name"),
        active_page="dashboard",
        last_checked="2025-09-05 02:52 KST"
    )

#  [ DB 테이블 스페이스 조회 ]  =========================================================

@app.route("/home/db-space")
def home_db_space():
    if not require_login():
        return redirect(url_for("login"))
    return render_template(
        "home_db_space.html",
        name=session.get("admin_name"),
        active_page="dbspace"
    )

#  [ 시뮬레이션 교통 분석 데이터 검색 ]  =========================================================

@app.route("/home/sim-search")
def home_sim_search():
    if not require_login():
        return redirect(url_for("login"))
    return render_template(
        "home_sim_search.html",
        name=session.get("admin_name"),
        active_page="simsearch"
    )

#  [ 교차로별 신호최적화 ]  =========================================================

@app.route("/home/signal-opt")
def home_signal_opt():
    if not require_login():
        return redirect(url_for("login"))
    return render_template(
        "home_signal_opt.html",
        name=session.get("admin_name"),
        active_page="signalopt"
    )















#  [ 모니터링 1 - 시간대별 교통수요 분석정보 ]  =========================================================

@app.route('/monitoring/visum-hourly-vc', methods=['GET'])
def visum_hourly_vc():
    
    # 기준 시간
    now_kst = datetime.now(KST)
    rule_date = "20250701"  # ▶ 테스트용
    # rule_date = resolve_dataset_date(now_kst)

    # 다음 업데이트 시각
    next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
    if now_kst >= next_update:
        next_update += timedelta(days=1)
    x_next_update = next_update.isoformat()

    ALLOWED_HOURS = {"08", "11", "14", "17", "24"}
    hour_param = (request.args.get('hour') or '').strip()
    if not hour_param or hour_param not in ALLOWED_HOURS:
        return jsonify({"error": "Missing/invalid 'hour'.",
                        "allowed": sorted(list(ALLOWED_HOURS))}), 400

    geometry_param = (request.args.get('geometry') or '0').strip()
    include_geometry = (geometry_param == '1')

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # --------------------------------------------------------
        # [변경 1] 가벼운 COUNT(*)로 ETag를 "먼저" 계산 → 조기 304
        # --------------------------------------------------------
        if hour_param == "24":
            sql_count = """
                SELECT COUNT(*)
                FROM TOMMS.TDA_LINK_DAY_RESULT
                WHERE STAT_DAY = ?
            """
            count_param = (rule_date,)
            display_mm = int(rule_date[4:6]); display_dd = int(rule_date[6:8])
            hour_label = f"{display_mm}월 {display_dd}일 전일 평균"
        else:
            stat_hour = rule_date + hour_param  # yyyyMMddHH
            sql_count = """
                SELECT COUNT(*)
                FROM TOMMS.TDA_LINK_HOUR_RESULT
                WHERE STAT_HOUR = ?
            """
            count_param = (stat_hour,)
            display_mm = int(stat_hour[4:6]); display_dd = int(stat_hour[6:8])
            sh = int(hour_param); eh = (sh + 1) % 24
            hour_label = f"{display_mm}월 {display_dd}일 {sh:02d}시 ~ {eh:02d}시"

        cursor.execute(sql_count, count_param)
        total_rows = int(cursor.fetchone()[0])

        # 시간 필터 집합(요청이 단일 시간대이므로 {hour})
        hours_filter = {int(hour_param)} if hour_param != "24" else {24}
        etag_val = make_etag(rule_date, hours_filter, total_rows)
        current_etag = f'"{etag_val}"'

        inm = _normalize_inm(request.headers.get("If-None-Match", ""))
        if inm == etag_val:
            # ▶ 조기 304: 무거운 SELECT/조인, JSON 가공 전 단계에서 바로 반환
            resp = Response(status=304)
            resp.headers["ETag"] = current_etag
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["X-Geometry-Included"] = '1' if include_geometry else '0'
            resp.headers["Vary"] = "Accept-Encoding"
            return resp

        # --------------------------------------------------------
        # [변경 2] 304가 아니면 그때만 "무거운" 본문 조회 수행
        # --------------------------------------------------------
        if hour_param == "24":
            sql_main = """
                SELECT LINK_ID, VC
                FROM TOMMS.TDA_LINK_DAY_RESULT
                WHERE STAT_DAY = ?
                ORDER BY LINK_ID
            """
            main_param = (rule_date,)
        else:
            sql_main = """
                SELECT LINK_ID, VC
                FROM TOMMS.TDA_LINK_HOUR_RESULT
                WHERE STAT_HOUR = ?
                ORDER BY LINK_ID
            """
            main_param = (stat_hour,)

        cursor.execute(sql_main, main_param)
        rows = cursor.fetchall()

        groups = []
        all_ids_in_order = []
        vc_list_by_id = defaultdict(list)

        for raw_link, vc_val in rows:
            if not raw_link:
                continue
            ids = [x.strip() for x in str(raw_link).split(",") if x.strip()]
            if not ids:
                continue
            try:
                vc_num = None if vc_val is None else float(vc_val)
            except:
                vc_num = None
            groups.append({"raw": ",".join(ids), "ids": ids})
            all_ids_in_order.extend(ids)
            if vc_num is not None:
                for lid in ids:
                    vc_list_by_id[lid].append(vc_num)

        if not groups:
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        coords_by_id = {}
        if include_geometry:
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
                return jsonify({"status": "fail", "message": "LINK_VERTEX 데이터가 없습니다."}), 404
            from collections import defaultdict as dd2
            coords_by_id = dd2(list)
            for link_id, link_seq, x, y in vrows:
                coords_by_id[str(link_id)].append([float(x), float(y)])

        vc_by_id_avg = {lid: (sum(arr) / len(arr)) for lid, arr in vc_list_by_id.items() if arr}

        documents = []
        doc_vcs_for_avg = []
        for g in groups:
            per_id_vcs = [vc_by_id_avg[lid] for lid in g["ids"] if lid in vc_by_id_avg]
            group_vc = round(sum(per_id_vcs) / len(per_id_vcs), 2) if per_id_vcs else None
            if group_vc is not None:
                doc_vcs_for_avg.append(group_vc)

            if include_geometry:
                merged_coords = []
                for lid in g["ids"]:
                    merged_coords.extend(coords_by_id.get(lid, []))
                documents.append({"link_id": g["raw"], "v/c": group_vc, "coordinates": merged_coords})
            else:
                documents.append({"link_id": g["raw"], "v/c": group_vc})

        # 참고 지표(집계)
        try:
            if hour_param == "24":
                sql_vol = """
                    SELECT NVL(SUM(TO_NUMBER(VOL)), 0)
                    FROM TOMMS.STAT_DAY_CROSS
                    WHERE STAT_DAY = ?
                      AND TRIM(INFRA_TYPE) = 'SMT'
                """
                vol_param = (rule_date,)
            else:
                stat_hour_for_vol = rule_date + hour_param
                sql_vol = """
                    SELECT NVL(SUM(TO_NUMBER(VOL)), 0)
                    FROM TOMMS.STAT_HOUR_CROSS
                    WHERE STAT_HOUR = ?
                      AND TRIM(INFRA_TYPE) = 'SMT'
                """
                vol_param = (stat_hour_for_vol,)
            cursor.execute(sql_vol, vol_param)
            row = cursor.fetchone()
            raw_traffic_vol = int(row[0]) if row and row[0] is not None else 0
            traffic_vol = int(round(raw_traffic_vol / 3))
        except Exception as e:
            print(f"[traffic_vol-ERROR] sql={sql_vol}\nparam={vol_param}\nexc={repr(e)}")
            traffic_vol = 0

        avg_vc = round(sum(doc_vcs_for_avg) / len(doc_vcs_for_avg), 2) if doc_vcs_for_avg else 0.00

        final_json = {
            "v/c": avg_vc,
            "hour_label": hour_label,
            "traffic_vol": traffic_vol,
            "documents": documents
        }

        body = json.dumps(final_json, ensure_ascii=False, separators=(',', ':'))
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)

        # 헤더
        resp.headers["ETag"] = current_etag                 # ← 위에서 만든 순수 해시 ETag 재사용
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        resp.headers["X-Geometry-Included"] = '1' if include_geometry else '0'
        resp.headers["Vary"] = "Accept-Encoding"
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
        finally:
            conn.close()

#  [ 모니터링 2 - 교통존간 통행정보 ]  =========================================================

@app.route('/monitoring/visum-zone-od', methods=['GET'])
def visum_zone_od():
    try:
        # ---------------------------
        # 파라미터
        # ---------------------------
        from_filter = request.args.get('from')   # ex) '810011'
        top_param   = request.args.get('top', default='10')
        try:
            top_n = max(1, int(top_param))
        except:
            top_n = 10

        conn = get_connection()
        cursor = conn.cursor()

        # ---------------------------
        # 실제 본문 조회 실행
        # ---------------------------
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
                    f.ZONE_NAME AS FROM_ZONE_NAME,
                    o.TO_ZONE_ID,
                    t.ZONE_NAME AS TO_ZONE_NAME,
                    NVL(o.AUTO_MATRIX_VALUE,0)
                  + NVL(o.BUS_MATRIX_VALUE,0)
                  + NVL(o.HGV_MATRIX_VALUE,0)  AS OD_MATRIX_VALUE,
                    f.LON AS FROM_LON, f.LAT AS FROM_LAT,
                    t.LON AS TO_LON,   t.LAT AS TO_LAT,
                    ROW_NUMBER() OVER (
                        PARTITION BY o.FROM_ZONE_ID
                        ORDER BY NVL(o.AUTO_MATRIX_VALUE,0)
                              + NVL(o.BUS_MATRIX_VALUE,0)
                              + NVL(o.HGV_MATRIX_VALUE,0) DESC
                    ) AS RN
                FROM TOMMS.TDA_ZONE_OD_RESULT o
                LEFT JOIN TOMMS.TDA_ZONE_INFO f ON f.ZONE_ID = o.FROM_ZONE_ID
                LEFT JOIN TOMMS.TDA_ZONE_INFO t ON t.ZONE_ID = o.TO_ZONE_ID
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

        # ---------------------------
        # 그룹 빌드
        # ---------------------------
        from collections import defaultdict
        groups = defaultdict(lambda: {
            "from_zone_id": None,
            "from_zone_name": None,
            "from_lon": None, "from_lat": None,
            "items": []
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

        # ---------------------------
        # GeoJSON 반환 (좌표 순서: [lon, lat])
        # ---------------------------
        payload = []
        for _, g in groups.items():
            points = {"type": "FeatureCollection", "features": []}

            # from point (GeoJSON 표준: [lon, lat])
            points["features"].append({
                "type": "Feature",
                "properties": {
                    "role": "from",
                    "zone_id": g["from_zone_id"],
                    "zone_name": g["from_zone_name"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [g["from_lon"], g["from_lat"]]
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
                        "coordinates": [it["to_lon"], it["to_lat"]]
                    }
                })

            # Lines from→to
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
                            [g["from_lon"], g["from_lat"]],
                            [it["to_lon"], it["to_lat"]]
                        ]
                    }
                })

            # 중심좌표 계산도 [lon, lat]로
            all_lons, all_lats = [], []
            for feat in points["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                    all_lons.append(coords[0]); all_lats.append(coords[1])
            if all_lons and all_lats:
                mid_lon = round((min(all_lons) + max(all_lons)) / 2.0, 4)
                mid_lat = round((min(all_lats) + max(all_lats)) / 2.0, 4)
                from_coords = [mid_lon, mid_lat]
            else:
                from_coords = [
                    g["from_lon"] if g["from_lon"] is not None else 0.0,
                    g["from_lat"] if g["from_lat"] is not None else 0.0
                ]

            payload.append({
                "from_zone": {
                    "id": g["from_zone_id"],
                    "name": g["from_zone_name"],
                    "coordinates": from_coords  # [lon, lat]
                },
                "points": points,
                "lines": lines,
                "meta": {"top": top_n, "format": "geojson", "coord_order": "lonlat"}
            })

        result = payload[0] if from_filter and len(payload) == 1 else payload

        # ---------------------------
        # 응답
        # ---------------------------
        body = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        return Response(body, status=200, content_type="application/json; charset=utf-8")

    except Exception as e:
        payload = {
            "status": "fail",
            "message": "OD 분석 중 에러 발생",
            "error": str(e),
            "timestamp": get_current_time(),
        }
        return Response(json.dumps(payload, ensure_ascii=False),
                        status=500,
                        content_type="application/json; charset=utf-8")

#  [ 모니터링 4K - 분석지역별 교통흐름 통계정보 ]  =========================================================

@app.route('/monitoring/statistics-traffic-flow', methods=['GET'])
def statistics_traffic_flow():
    try:
        # =========================================================
        # ✅   기준 시간/헤더용 현재 시간 계산
        # =========================================================
        now_kst = datetime.now(KST)

        # ▶ 테스트용 (배포 시 resolve_dataset_date(now_kst)로 대체)
        rule_date = "20250701"
        # rule_date = resolve_dataset_date(now_kst)

        # ▶ 다음 업데이트 시각(매일 06:00 KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # =========================================================
        # ✅   hour / geometry 쿼리 파라미터 파싱 / 검증
        # =========================================================
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

        # ▶ geometry 파라미터 (기본: 0 = 좌표 생략 / 1 = 좌표 포함)
        geometry_param = (request.args.get('geometry') or '0').strip()
        include_geometry = (geometry_param == '1')

        # ▶ ETag용 키를 정수 튜플로 정규화 (예: {"08","11"} -> (8,11))
        if hours_filter:
            hours_key_for_etag = tuple(sorted(int(h) for h in hours_filter))
        else:
            hours_key_for_etag = None

        conn = get_connection()
        cursor = conn.cursor()

        # ============================================================
        # ✅    데이터 조회 : 시간별 링크 결과 HOUR_LINK_RESULT
        # ============================================================
        cursor.execute("""
            SELECT
                h.STAT_HOUR,
                h.LINK_ID,
                i.DISTRICT_ID AS DISTRICT,
                i.SA_NO       AS SA_NO,
                h.VC,
                h.VEHS AS VOLUME,
                h.SPEED
            FROM TOMMS.TDA_LINK_HOUR_RESULT h
            LEFT JOIN TOMMS.TDA_LINK_INFO i
                ON i.LINK_ID = h.LINK_ID
            WHERE SUBSTR(h.STAT_HOUR, 1, 8) = ?
        """, [rule_date])

        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows] if rows else []
        if not rows:
            return jsonify({"status": "fail", "message": f"{rule_date} 데이터가 없습니다."}), 404

        df_result = pd.DataFrame(
            rows,
            columns=["STAT_HOUR", "LINK_ID", "DISTRICT", "SA_NO", "VC", "VOLUME", "SPEED"]
        )

        # ▶ 시간 필터 적용(요청 시)
        if hours_filter:
            # STAT_HOUR: yyyymmddHH → 뒤 2자리(HH) 기준 필터
            df_result = df_result[df_result["STAT_HOUR"].str[-2:].isin(hours_filter)]

        # ▶ 빈 결과 204 + ETag
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

        # ▶ ETag / If-None-Match  (요청 기준 유지 — etag 로직은 그대로)
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

        # ============================================================
        # ✅ 4k 지도 // 권역별 시간별 평균 속도 - 결과값 가공
        #     * 변경점:
        #       - VC, VOLUME은 사용하지 않으므로 변환/집계 제거
        #       - 좌표는 geometry=1일 때만 조회/첨부
        # ============================================================

        # 숫자형 보정: SPEED만 사용
        df_result["SPEED"] = pd.to_numeric(df_result["SPEED"], errors="coerce")

        # ▶ LINK_ID 목록 추출(순서 보존 중복제거)
        all_ids_in_order = [str(x).strip() for x in df_result["LINK_ID"].tolist() if str(x).strip()]
        unique_ids = list(dict.fromkeys(all_ids_in_order).keys())

        if not unique_ids:
            return jsonify({"status": "fail", "message": "조회된 LINK_ID가 없습니다."}), 404

        # ▶ LINK_VERTEX 일괄 조회(IN) — geometry=1일 때만
        coords_by_id = {}
        if include_geometry:
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
                return jsonify({"status": "fail", "message": "LINK_VERTEX 데이터가 없습니다."}), 404

            from collections import defaultdict as dd2
            coords_by_id = dd2(list)
            # [x=lon, y=lat]
            for link_id, link_seq, x, y in vrows:
                fx = float(x) if x is not None else None
                fy = float(y) if y is not None else None
                coords_by_id[str(link_id)].append([fx, fy])

        # ▶ 링크별 요약: speed 평균만
        agg_df = (
            df_result.groupby("LINK_ID").agg(
                speed_avg=("SPEED", "mean")
            ).reset_index()
        )
        spd_by_id = {
            str(r["LINK_ID"]): (None if pd.isna(r["speed_avg"]) else round(float(r["speed_avg"]), 2))
            for _, r in agg_df.iterrows()
        }

        # ▶ LINK_ID → DISTRICT 매핑 (첫 등장값 기준)
        df_link_dist = df_result[["LINK_ID", "DISTRICT"]].copy()
        df_link_dist["DISTRICT"] = pd.to_numeric(df_link_dist["DISTRICT"], errors="coerce").astype("Int64")
        df_link_dist = df_link_dist.dropna(subset=["DISTRICT"])
        df_link_dist = df_link_dist.drop_duplicates(subset=["LINK_ID"], keep="first")
        link_to_district = {str(row["LINK_ID"]): int(row["DISTRICT"]) for _, row in df_link_dist.iterrows()}

        # ▶ 구역별 바스켓 4개 생성
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

        # ▶ 링크 요약을 해당 DISTRICT 바스켓에 채우기
        for lid in unique_ids:
            d_no = link_to_district.get(str(lid))
            if d_no not in idx_by_code:  # DISTRICT가 1~4가 아니면 스킵
                continue
            item = {
                "link_id": str(lid),
                "speed": spd_by_id.get(str(lid))  # ✅ v/c, volume 제거
            }
            if include_geometry:
                item["coordinates"] = coords_by_id.get(str(lid), [])  # ✅ geometry=1일 때만 첨부
            documents_grouped[idx_by_code[d_no]]["data"].append(item)

        # ============================================================
        # ✅ 우측 결과값 // 일 평균 속도 - daily_average_speed (기존 유지)
        # ============================================================
        cursor.execute("""
            SELECT
                i.DISTRICT_ID AS DISTRICT,
                i.SA_NO       AS SA_NO,
                d.SPEED
            FROM TOMMS.TDA_LINK_DAY_RESULT d
            LEFT JOIN TOMMS.TDA_LINK_INFO i
                ON i.LINK_ID = d.LINK_ID
            WHERE d.STAT_DAY = ?
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
        # ✅     hour_label 생성: 'mm월 dd일 hh시 ~ hh시'
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
            if hours_filter:
                candidate_hours = sorted(hours_filter)
            else:
                candidate_hours = sorted(set(df_result["STAT_HOUR"].str[-2:]).intersection(ALLOWED_HOURS))
            hour_label_value = [make_label(h) for h in candidate_hours]

        # ============================================================
        # ✅ hourly_total_traffic_volume 값 조회 (기존 유지)
        # ============================================================
        if hours_filter and len(hours_filter) == 1:
            hh = next(iter(hours_filter))
        else:
            if hours_filter:
                candidate_hours = sorted(hours_filter)
            else:
                candidate_hours = sorted(set(df_result["STAT_HOUR"].str[-2:]).intersection(ALLOWED_HOURS))
                if not candidate_hours:
                    candidate_hours = sorted(list(ALLOWED_HOURS))
            hh = candidate_hours[0]

        stat_hour_key = rule_date + hh  # 'yyyymmddhh'

        sql_tv = """
            SELECT n.DISTRICT_ID,
                    NVL(SUM(TO_NUMBER(c.VOL)), 0) AS SUM_VOL
            FROM TOMMS.STAT_HOUR_CROSS c
            JOIN TOMMS.TFA_NODE_INFO n
                ON n.CROSS_ID = c.CROSS_ID
            WHERE c.STAT_HOUR = ?
                AND TRIM(c.INFRA_TYPE) = 'SMT'
                AND n.DISTRICT_ID IN (1,2,3,4)
            GROUP BY n.DISTRICT_ID
        """
        cursor.execute(sql_tv, (stat_hour_key,))
        tv_rows = cursor.fetchall()

        dist_sum = {1: 0, 2: 0, 3: 0, 4: 0}
        for d_id, s in tv_rows or []:
            try:
                dist_sum[int(d_id)] = int(s)
            except Exception:
                pass

        hourly_total_traffic_volume = [
            {"district": "교동지구", "traffic_volume": int(dist_sum[1] / 3)},
            {"district": "송정지구", "traffic_volume": int(dist_sum[2] / 3)},
            {"district": "중심지구", "traffic_volume": int(dist_sum[3] / 3)},
            {"district": "경포지구", "traffic_volume": int(dist_sum[4] / 3)},
        ]

        # ============================================================
        # ✅     본문 + 헤더(정상 200에서도 ETag 포함)
        # ============================================================
        payload = {
            "status": "success",
            "hour_label": hour_label_value,
            "row_count": int(len(df_result)),
            "documents": documents_grouped,              # DISTRICT-그룹 형태
            "daily_average_speed": daily_average_speed,
            "hourly_total_traffic_volume": hourly_total_traffic_volume,
            "map_center_coordinates": [
                {"district": "교동지구", "coordinates": [128.874273, 37.765208]},
                {"district": "송정지구", "coordinates": [128.924538, 37.771808]},
                {"district": "중심지구", "coordinates": [128.897176, 37.755575]},
                {"district": "경포지구", "coordinates": [128.891529, 37.787484]}
            ]
        }

        body = json.dumps(payload, ensure_ascii=False)
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = etag  # etag 로직은 요청대로 유지
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        # 가독/디버그용: 좌표 포함 여부 노출
        resp.headers["X-Geometry-Included"] = '1' if include_geometry else '0'
        return resp

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            cursor.close()
        finally:
            conn.close()

#  [ 모니터링 4 - 교차로별 통행정보 ]  =========================================================

@app.route('/monitoring/node-result', methods=['GET'])
def node_result_summary():
    now_kst = datetime.now(KST)

    # 테스트용
    rule_date = "20250701"
    mm, dd = int(rule_date[4:6]), int(rule_date[6:8])

    # 다음 업데이트 시각(매일 06:00 KST)
    next_update_kst = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
    if now_kst >= next_update_kst:
        next_update_kst += timedelta(days=1)
    x_next_update_str = next_update_kst.isoformat()

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # =========================================================
        # ✅ 조기 ETag 계산 (가벼운 COUNT(*) 기반) + 304 선판단
        #    - 유틸: make_etag(dataset_date, hours_filter_set, total_rows)
        #    - node-result는 하루 단일 뷰이므로 hours_filter_set=None
        # =========================================================
        cursor.execute("""
            SELECT COUNT(*)
            FROM TOMMS.TFA_NODE_15MIN_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        row_count = int(cursor.fetchone()[0] or 0)

        etag_val = make_etag(rule_date, None, row_count)  # 유틸 사용
        current_etag = f'"{etag_val}"'

        # If-None-Match 정규화 비교(유틸 사용)
        inm = _normalize_inm(request.headers.get("If-None-Match", ""))
        if inm == etag_val:
            resp = Response(status=304)
            resp.headers["ETag"] = current_etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update_str
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            return resp

        # =========================================================
        # ✅ 데이터 조회 (304가 아닐 때만 본문 생성)
        # =========================================================
        cursor.execute("""
            SELECT STAT_HOUR, TIMEINT, NODE_ID, QLEN, VEHS, DELAY, STOPS
            FROM TOMMS.TFA_NODE_15MIN_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        rows = cursor.fetchall()
        rows = [tuple(r) for r in rows]

        if not rows:
            return jsonify({"status": "fail", "message": "해당 날짜에 대한 교차로 데이터가 없습니다."}), 404

        # 가공
        df = pd.DataFrame(rows, columns=["STAT_HOUR", "TIMEINT", "NODE_ID", "QLEN", "VEHS", "DELAY", "STOPS"])
        df[["QLEN", "VEHS", "DELAY", "STOPS"]] = df[["QLEN", "VEHS", "DELAY", "STOPS"]].apply(pd.to_numeric, errors="coerce")
        df["DATE"] = df["STAT_HOUR"].str[:8]
        df["HOUR"] = df["STAT_HOUR"].str[8:10]

        df_avg = df.groupby(["DATE", "HOUR", "NODE_ID"], as_index=False).agg({
            "QLEN": "mean",
            "VEHS": "mean",
            "DELAY": "mean",
            "STOPS": "mean"
        }).round(2)

        cursor.execute("SELECT NODE_ID, CROSS_NAME FROM TOMMS.TFA_NODE_INFO")
        node_info_rows = cursor.fetchall()
        node_info = [tuple(r) for r in node_info_rows]
        df_node_info = pd.DataFrame(node_info, columns=["NODE_ID", "NODE_NAME"]).drop_duplicates(subset="NODE_ID")

        df_merged = df_avg.merge(df_node_info, on="NODE_ID", how="left")
        df_merged = df_merged[df_merged["NODE_NAME"].notna()].copy()

        def los_alpha(delay):
            if delay < 15: return "A"
            elif delay < 30: return "B"
            elif delay < 50: return "C"
            elif delay < 70: return "D"
            elif delay < 100: return "E"
            elif delay < 220: return "F"
            elif delay < 340: return "FF"
            else: return "FFF"
        los_map = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "FF": "F", "FFF": "F"}
        df_merged["LOS_STR"] = df_merged["DELAY"].apply(lambda d: los_map[los_alpha(d)])

        df_merged.rename(columns={
            "NODE_NAME": "node_name",
            "QLEN": "qlen",
            "VEHS": "vehs",
            "DELAY": "delay",
            "STOPS": "stops"
        }, inplace=True)

        # 시간 블록
        data_blocks = []
        for hour, group in df_merged.groupby("HOUR"):
            h = int(hour); h_next = (h + 1) % 24
            hour_label = f"{mm}월 {dd}일 {h:02d}시 ~ {h_next:02d}시"
            items = []
            for _, r in group.iterrows():
                qlen  = float(r["qlen"])  if pd.notna(r["qlen"])  else 0.0
                vehs  = int(r["vehs"])    if pd.notna(r["vehs"])  else 0
                delay = float(r["delay"]) if pd.notna(r["delay"]) else 0.0
                stops = float(r["stops"]) if pd.notna(r["stops"]) else 0.0
                los   = str(r["LOS_STR"]) if pd.notna(r["LOS_STR"]) else None

                items.append({
                    "node_name": str(r["node_name"]),
                    "qlen": round(qlen, 1),
                    "vehs": int(vehs),
                    "delay": round(delay, 1),
                    "stops": round(stops, 1),
                    "los": los,
                    "max_qlen": round(qlen * 1.5, 1),
                    "max_vehs": int(vehs * 1.5),
                    "max_delay": round(delay * 1.5, 1),
                    "max_stops": round(stops * 1.5, 1),
                    "max_los": "F"
                })
            data_blocks.append({"hour_label": hour_label, "items": items})

        # ===== Payload (본문 생성) =====
        payload = {
            "target_date": rule_date,
            "data": data_blocks
        }
        body = json.dumps(payload, ensure_ascii=False)

        # 200 OK — 조기 계산한 ETag 재사용
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["ETag"] = current_etag      # 본문 MD5 대신 '조기 ETag' 사용
        resp.headers["X-Dataset-Date"] = rule_date
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

#  [ 모니터링 5 - 도로구간별 통행량 정보 ]  =========================================================

@app.route('/monitoring/road-traffic-info', methods=['GET'])
def road_traffic_info():
    try:
        # =========================================================
        # ✅   기준 시간/헤더용 현재 시간 계산
        # =========================================================
        now_kst = datetime.now(KST)

        # ▶ 테스트용 (배포 시 resolve_dataset_date(now_kst)로 대체)
        rule_date = "20250701"
        # rule_date = resolve_dataset_date(now_kst)

        # ▶ 다음 업데이트 시각(매일 06:00 KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # =========================================================
        # ✅   쿼리 파라미터 파싱 / 검증
        # =========================================================
        ALLOWED_HOURS = {"08", "11", "14", "17"}

        hour_param = (request.args.get('hour') or '').strip()
        if hour_param not in ALLOWED_HOURS:
            return jsonify({
                "status": "error",
                "message": "Missing/invalid 'hour'.",
                "allowed": sorted(list(ALLOWED_HOURS))
            }), 400

        road_id = (request.args.get('road_id') or '').strip()
        if not road_id:
            return jsonify({"status": "error", "message": "Missing 'road_id'."}), 400

        geometry_param = (request.args.get('geometry') or '0').strip()
        include_geometry = (geometry_param == '1')

        stat_hour = rule_date + hour_param  # 'yyyymmddhh'
        mm, dd = int(rule_date[4:6]), int(rule_date[6:8])
        sh = int(hour_param); eh = (sh + 1) % 24
        hour_label = f"{mm}월 {dd}일 {sh:02d}시 ~ {eh:02d}시"

        # =========================================================
        # ✅   DB 연결
        # =========================================================
        conn = get_connection()
        cursor = conn.cursor()

        # =========================================================
        # ✅   A안: ETag는 'hour' 단일 기준 (road_id 무시)
        #      - COUNT + SUM(FB_VEHS)만으로 가벼운 지문 생성
        #      - If-None-Match 선판단 → 곧바로 304
        # =========================================================
        cursor.execute("""
            SELECT COUNT(*), NVL(SUM(FB_VEHS), 0)
            FROM TOMMS.TDA_ROAD_VOL_HOUR_RESULT
            WHERE STAT_HOUR = ?
        """, (stat_hour,))
        hour_cnt, hour_sum = cursor.fetchone()
        hour_cnt = int(hour_cnt or 0)
        hour_sum = round(float(hour_sum or 0.0), 2)

        # hour 스냅샷 공통 ETag (모든 road_id 요청에 동일)
        etag_base = f"{rule_date}|{hour_param}|cnt={hour_cnt}|sum={hour_sum}"
        etag_val_hour = hashlib.md5(etag_base.encode("utf-8")).hexdigest()
        etag_hdr = f'"{etag_val_hour}"'

        inm = _normalize_inm(request.headers.get("If-None-Match", ""))
        if inm == etag_val_hour:
            resp = Response(status=304)
            resp.headers["ETag"] = etag_hdr
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["X-Geometry-Included"] = '1' if include_geometry else '0'
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            return resp

        # =========================================================
        # ✅   본문 조회/가공: road_id 단일 결과
        #      - ROAD_ID + LINK_ID별 합산
        #      - ROAD_NAME 보강
        #      - geometry=1일 때만 LINK_VERTEX 조회
        # =========================================================
        # ROAD_ID + LINK_ID별 FB_VEHS 합산
        cursor.execute("""
            SELECT ROAD_ID, LINK_ID, SUM(FB_VEHS) AS FB_SUM
            FROM TOMMS.TDA_ROAD_VOL_HOUR_RESULT
            WHERE STAT_HOUR = ? AND ROAD_ID = ?
            GROUP BY ROAD_ID, LINK_ID
            HAVING SUM(FB_VEHS) > 0          -- 🔽 vehs 값이 0인 경우 제외
            ORDER BY LINK_ID
        """, (stat_hour, road_id))
        rows = cursor.fetchall()
        if not rows:
            return jsonify({"status": "fail", "message": f"{stat_hour} / {road_id} 데이터가 없습니다."}), 404

        # ROAD_NAME 조회
        cursor.execute("""
            SELECT ROAD_NAME, UPDOWN, VL_LINKS_IN, VL_LINKS_OUT
            FROM TOMMS.TDA_ROAD_VOL_INFO
            WHERE ROAD_ID = ?
        """, (road_id,))
        road_name_row = cursor.fetchone()

        # 상/하행 라벨링
        road_name = None
        vl_in_raw = None
        vl_out_raw = None
        if road_name_row:
            base_name, updown, vl_in_raw, vl_out_raw = road_name_row[0], road_name_row[1], road_name_row[2], road_name_row[3]
            if updown == 1:
                road_name = f"{base_name} 상행"
            elif updown == 0:
                road_name = f"{base_name} 하행"
            else:
                road_name = base_name  # 예외 처리
        def parse_vl_links(s: str):
            """
            '2520120800_2520114000_...' 형태 문자열을 '_'로 분리하여
            10자리 숫자 링크아이디 리스트로 반환
            """
            if not s:
                return []
            parts = [p.strip() for p in str(s).split('_') if p.strip()]
            return [p for p in parts if p.isdigit() and len(p) == 10]

        vl_links_in  = parse_vl_links(vl_in_raw)
        vl_links_out = parse_vl_links(vl_out_raw)

        # 링크 모음 및 값 맵
        link_ids = []
        fb_map = {}
        for r in rows:
            lid = str(r[1]) if r[1] is not None else None
            if not lid:
                continue
            link_ids.append(lid)
            try:
                fb_map[lid] = int(round(float(r[2] or 0.0)))
            except:
                fb_map[lid] = 0

        # 좌표 (geometry=1일 때만)
        coords_by_link = {}
        if include_geometry and link_ids:
            # 🔽 fb_vehs 값이 0보다 큰 LINK_ID만 geometry 조회 대상에 포함
            positive_link_ids = [lid for lid in link_ids if fb_map.get(lid, 0) > 0]
            if positive_link_ids:
                unique_link_ids = list(dict.fromkeys(positive_link_ids))
                placeholders = ",".join(["?"] * len(unique_link_ids))
                sql_vertex = f"""
                    SELECT LINK_ID, LINK_SEQ, WGS84_X, WGS84_Y
                    FROM TOMMS.LINK_VERTEX
                    WHERE LINK_ID IN ({placeholders})
                    ORDER BY LINK_ID, LINK_SEQ
                """
                cursor.execute(sql_vertex, tuple(unique_link_ids))
                vrows = cursor.fetchall() or []
                from collections import defaultdict as dd2
                tmp = dd2(list)
                for link_id, link_seq, x, y in vrows:
                    try:
                        fx = float(x) if x is not None else None
                        fy = float(y) if y is not None else None
                    except:
                        fx, fy = None, None
                    tmp[str(link_id)].append([fx, fy])  # [lon, lat]
                coords_by_link = dict(tmp)
            
        vals = list(fb_map.values())
        if vals:
            min_fb = min(vals)
            max_fb = max(vals)
        else:
            min_fb = max_fb = 0

        def to_width_bin(v: int, vmin: int, vmax: int) -> int:
            """
            fb_vehs 값을 4단계 범주(width 2/5/10/25)로 매핑.
            - 구간: [vmin, t1], (t1, t2], (t2, t3], (t3, vmax]
              where t1 = vmin + 1/4*(vmax-vmin), ... t3 = vmin + 3/4*(vmax-vmin)
            - 모든 값이 동일하면: v>0 -> 25, 그 외 -> 2
            """
            if vmax <= vmin:
                return 25 if v > 0 else 2

            span = (vmax - vmin) / 4.0
            t1 = vmin + span
            t2 = vmin + 2 * span
            t3 = vmin + 3 * span

            if v <= t1:
                return 2
            elif v <= t2:
                return 5
            elif v <= t3:
                return 10
            else:
                return 25

        # 응답 data 구성
        data = []
        for lid in link_ids:
            fbv = fb_map.get(lid, 0)
            item = {
                "link_id": lid,
                "vehs": fbv,
                "width": str(to_width_bin(fbv, min_fb, max_fb))
            }
            if include_geometry:
                item["coordinates"] = coords_by_link.get(lid, [])
            data.append(item)
        
        # 🔽 fb_vehs 기준 내림차순 정렬
        data.sort(key=lambda x: x["vehs"], reverse=True)

        payload = {
            "status": "success",
            "hour_label": hour_label,
            "road_id": road_id,
            "road_name": road_name,
            "row_count": len(data),
            # 디버깅/뷰 튜닝용으로 min/max를 헤더로 참고할 수 있게 포함(원하면 제거해도 됨)
            "min_fb_vehs": min_fb,
            "max_fb_vehs": max_fb,
            "vl_links_in": vl_links_in,
            "vl_links_out": vl_links_out,
            "data": data
        }

        body = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        resp.headers["ETag"] = etag_hdr                 # ← hour 공통 ETag 재사용
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update
        resp.headers["X-Geometry-Included"] = '1' if include_geometry else '0'
        return resp

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            cursor.close()
        finally:
            conn.close()










#  [ 신호운영 1 - 도로구간 및 지점별 통계정보 ]  =========================================================

@app.route('/signal/vttm-result', methods=['GET'])
def vttm_result_summary():
    # =========================================================
    # ✅   기준 시간/헤더용 현재 시간 계산
    # =========================================================
    now_kst = datetime.now(KST)

    # ▶ 테스트용 (배포 시 resolve_dataset_date(now_kst)로 대체)
    rule_date = "20250701"
    # rule_date = resolve_dataset_date(now_kst)
    mm, dd = int(rule_date[4:6]), int(rule_date[6:8])

    # ▶ 다음 업데이트 시각(매일 06:00 KST)
    next_update_kst = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
    if now_kst >= next_update_kst:
        next_update_kst += timedelta(days=1)
    x_next_update_str = next_update_kst.isoformat()

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # =========================================================
        # ✅ 1) 구간(VTTM) 시간별 결과 조회 + 기본정보 JOIN
        #     - rule_date(YYYYMMDD) 기준
        # =========================================================
        cursor.execute("""
            SELECT
                v.STAT_HOUR,          -- 'YYYYMMDDHH'
                v.VTTM_ID,
                v.DISTANCE,
                v.VEHS,
                v.TRAVEL_TIME,
                i.DISTRICT_ID,
                i.FROM_NODE_NAME,
                i.TO_NODE_NAME,
                i.UPDOWN
            FROM TOMMS.TFA_VTTM_HOUR_RESULT v
            JOIN TOMMS.TFA_VTTM_INFO i
            ON i.VTTM_ID = v.VTTM_ID
            WHERE SUBSTR(v.STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        vttm_rows = [tuple(r) for r in (cursor.fetchall() or [])]
        if not vttm_rows:
            return jsonify({"status": "fail", "message": f"{rule_date} VTTM 데이터가 없습니다."}), 404

        cols_v = ["STAT_HOUR","VTTM_ID","DISTANCE","VEHS","TRAVEL_TIME", "DISTRICT_ID","FROM_NODE_NAME","TO_NODE_NAME","UPDOWN"]
        df_v = pd.DataFrame(vttm_rows, columns=cols_v)

        # 숫자형 안전 변환
        for c in ["DISTANCE","VEHS","TRAVEL_TIME","DISTRICT_ID","UPDOWN"]:
            df_v[c] = pd.to_numeric(df_v[c], errors="coerce")

        # =========================================================
        # ✅ 2) 지점(DC) 시간별 결과 조회 + VTTM 매핑 JOIN
        #     - rule_date(YYYYMMDD) 기준
        # =========================================================
        cursor.execute("""
            SELECT
                d.STAT_HOUR,          -- 'YYYYMMDDHH'
                m.VTTM_ID,
                d.DC_ID,
                d.TRAVEL_TIME,
                d.VEHS,
                d.SPEED
            FROM TOMMS.TFA_DC_HOUR_RESULT d
            JOIN TOMMS.TFA_VTTM_DC_INFO m
            ON m.DC_ID = d.DC_ID
            WHERE SUBSTR(d.STAT_HOUR, 1, 8) = ?
        """, [rule_date])
        dc_rows = [tuple(r) for r in (cursor.fetchall() or [])]

        cols_d = ["STAT_HOUR","VTTM_ID","DC_ID","TRAVEL_TIME","VEHS","SPEED"]
        df_dc = pd.DataFrame(dc_rows, columns=cols_d) if dc_rows else pd.DataFrame(columns=cols_d)

        # 숫자형 변환
        if not df_dc.empty:
            df_dc["TRAVEL_TIME"] = pd.to_numeric(df_dc["TRAVEL_TIME"], errors="coerce")
            df_dc["VEHS"]        = pd.to_numeric(df_dc["VEHS"], errors="coerce")
            df_dc["SPEED"]       = pd.to_numeric(df_dc["SPEED"], errors="coerce")
            df_dc["VTTM_ID"]     = df_dc["VTTM_ID"].astype(str)   # ← 핵심

            # VTTM_ID 단위로 집계(같은 시각/같은 VTTM에 여러 DC가 매핑될 수 있으므로)
            # - traffic_vol: SUM(VEHS)
            # - travel_speed: AVG(SPEED)
            # - travel_time:  AVG(TRAVEL_TIME)
            df_dc_agg = (
                df_dc.groupby(["STAT_HOUR","VTTM_ID"], as_index=False)
                    .agg(traffic_vol=("VEHS","sum"),
                        travel_speed=("SPEED","mean"),
                        travel_time=("TRAVEL_TIME","mean"))
            )
        else:
            df_dc_agg = pd.DataFrame(columns=["STAT_HOUR","VTTM_ID","traffic_vol","travel_speed","travel_time"])

        # =========================================================
        # ✅ 3) 구간+지점 매칭을 위한 키 구성
        #     - pair_buffer[(district_name, hour_label, segment_key)][updown] = {..}
        #     - segment_key = tuple(sorted([from_node, to_node]))
        # =========================================================
        district_mapping_local = {
            1: "교동", 2: "송정", 3: "도심", 4: "경포"
        }

        def make_hour_label(stat_hour: str) -> str:
            # 'YYYYMMDDHH' → 'mm월 dd일 HH시 ~ HH+1시'
            hh = int(stat_hour[-2:])
            return f"{mm}월 {dd}일 {hh:02d}시 ~ {(hh + 1) % 24:02d}시"

        # VTTM_ID→(DISTRICT_ID, FROM, TO, UPDOWN, DIST, STAT_HOUR, TRAVEL_TIME)
        # 방향쌍을 만들기 위해 pair buffer 생성
        from collections import defaultdict
        pair_buffer = defaultdict(dict)  # key: (district_name, hour_label, segment_key) -> { '0': {...}, '1': {...} }
        distance_pick = {}               # key: (district_name, hour_label, segment_key) -> distance (첫 값 고정)

        # DC agg lookup dict
        dc_key = {
            (r["STAT_HOUR"], r["VTTM_ID"]): (
                int(r["traffic_vol"]) if not pd.isna(r["traffic_vol"]) else 0,
                float(r["travel_speed"]) if not pd.isna(r["travel_speed"]) else 0.0,
                float(r["travel_time"]) if not pd.isna(r["travel_time"]) else 0.0
            )
            for _, r in df_dc_agg.iterrows()
        }

        for _, r in df_v.iterrows():
            stat_hour = str(r["STAT_HOUR"])
            vttm_id   = str(r["VTTM_ID"])
            dist_id   = int(r["DISTRICT_ID"]) if not pd.isna(r["DISTRICT_ID"]) else None
            from_node = str(r["FROM_NODE_NAME"]) if pd.notna(r["FROM_NODE_NAME"]) else None
            to_node   = str(r["TO_NODE_NAME"])   if pd.notna(r["TO_NODE_NAME"])   else None
            updown    = int(r["UPDOWN"]) if not pd.isna(r["UPDOWN"]) else None
            distance  = float(r["DISTANCE"]) if not pd.isna(r["DISTANCE"]) else 0.0
            ttime     = float(r["TRAVEL_TIME"]) if not pd.isna(r["TRAVEL_TIME"]) else 0.0

            if dist_id not in (1,2,3,4) or from_node is None or to_node is None or updown is None:
                continue

            district_name = district_mapping_local.get(dist_id, f"기타지역-{dist_id}")
            hour_label    = make_hour_label(stat_hour)
            segment_key   = tuple(sorted([from_node, to_node]))
            key           = (district_name, hour_label, segment_key)

            # 속도 계산: km/h 가정 (distance[m]이면 3.6, km면 3600… 현재 로직 유지)
            travel_time  = round(ttime, 1) if ttime > 0 else 0.0
            travel_speed = round((distance / ttime) * 3.6, 1) if ttime > 0 else 0.0

            # 지점(DC) 집계 매칭 (없으면 0/0.0)
            dc_tuple = dc_key.get((stat_hour, vttm_id), (0, 0.0, 0.0))
            dc_traffic_vol, dc_travel_speed, dc_travel_time = dc_tuple

            pair_buffer[key][str(updown)] = {
                "from_node": from_node,
                "to_node": to_node,
                "travel_time": travel_time,
                "travel_speed": travel_speed
            }

            # distance는 처음 본 값으로 고정(방향별 상이 시 임의 일치)
            if key not in distance_pick:
                distance_pick[key] = float(distance)

            # DC 매칭 결과를 key별로 기억(마지막에 items 만들 때 사용하기 위해)
            # updown 별이 아니라 segment 묶음에 하나만 넣기 위해 pair_buffer에 저장
            pair_buffer[key]["__dc__"] = {
                "traffic_vol": int(dc_traffic_vol),
                "travel_speed": round(float(dc_travel_speed), 1) if dc_travel_speed else 0.0,
                "travel_time": round(float(dc_travel_time), 1) if dc_travel_time else 0.0
            }

        # =========================================================
        # ✅ 4) hour_label -> district_name -> items[] 구성
        #     items: { from_node, to_node, directions[], data_collection{} }
        # =========================================================
        hour_district_map = defaultdict(lambda: defaultdict(list))

        for (district_name, hour_label, segment_key), directions in pair_buffer.items():
            # directions: {'0': {...}, '1': {...}, '__dc__': {...}}
            dir_list = []
            if '0' in directions:
                dir_list.append({
                    "updown": 0,
                    "travel_time": directions['0']["travel_time"],
                    "travel_speed": directions['0']["travel_speed"],
                    "distance": distance_pick.get((district_name, hour_label, segment_key), 0.0)
                })
            if '1' in directions:
                dir_list.append({
                    "updown": 1,
                    "travel_time": directions['1']["travel_time"],
                    "travel_speed": directions['1']["travel_speed"],
                    "distance": distance_pick.get((district_name, hour_label, segment_key), 0.0)
                })

            # from/to는 정렬된 키의 0,1순서가 아니라, 실제 한쪽(updown=0)의 from/to를 쓰고 싶다면 교체 가능
            from_node_out, to_node_out = segment_key[0], segment_key[1]

            dc_info = directions.get("__dc__", {"traffic_vol": 0, "travel_speed": 0.0, "travel_time": 0.0})

            hour_district_map[hour_label][district_name].append({
                "from_node": from_node_out,
                "to_node": to_node_out,
                "directions": dir_list,          # 구간(양방향) 결과
                "data_collection": dc_info       # 지점 결과(매칭 집계)
            })

        # 배열로 변환
        data_blocks = []
        for hour_label, districts in hour_district_map.items():
            for district_name, items in districts.items():
                data_blocks.append({
                    "hour_label": hour_label,
                    "district": district_name,
                    "items": items
                })

        # =========================================================
        # ✅   Payload & Etag / If-None-Match / Response
        # =========================================================
        payload = {
            "status": "success",
            "target_date": rule_date,
            "data": data_blocks
        }

        body = json.dumps(payload, ensure_ascii=False)
        etag = f'"{hashlib.md5(body.encode("utf-8")).hexdigest()}"'

        # If-None-Match 처리
        inm_raw = request.headers.get("If-None-Match", "")
        inm = inm_raw.strip().strip('"').replace("W/", "").strip()
        if inm == etag.strip('"'):
            resp = Response(status=304)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update_str
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
        resp.headers["X-Dataset-Date"] = rule_date
        resp.headers["X-Next-Update"] = x_next_update_str
        return resp

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "VTTM 결과 조회 중 오류 발생",
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

#  [ 신호운영 4K - 분석지역 및 세부분석단위별 통행정보 : {DISTRICT_NAME} ]  =========================================================

@app.route('/signal/district-hourly-congested-info', methods=['GET'])
def hourly_congested_info_data():
    try:
        now_kst = datetime.now(KST)
        # rule_date = resolve_dataset_date(now_kst)  # 배포 시 복원
        rule_date = "20250701"                       # 테스트용

        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # --- hour 파라미터 ---
        ALLOWED_HOURS = {"08", "11", "14", "17"}
        hours_raw = (request.args.get('hour') or '').strip()
        hours_filter_list = []
        if hours_raw:
            parts = [p.strip() for p in hours_raw.split(",") if p.strip()]
            invalid = [p for p in parts if p not in ALLOWED_HOURS]
            if invalid:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid hour value(s): {', '.join(invalid)}",
                    "allowed": sorted(list(ALLOWED_HOURS))
                }), 400
            hours_filter_list = sorted(set(parts))  # 예: ['08','11']

        hours_key_for_etag = tuple(int(h) for h in hours_filter_list) if hours_filter_list else None
        district_mapping = {1: "교동", 2: "송정", 3: "도심", 4: "경포"}

        conn = get_connection()
        cursor = conn.cursor()

        # ============================================================
        # A) travel_cost — TFA_DISTRICT_HOUR_RESULT (권역×시간대 합계)
        #    3컬럼 고정: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), COST
        # ============================================================
        sql_cost = """
            SELECT
                DISTRICT_ID,
                SUBSTR(STAT_HOUR, 1, 10) AS STAT_HOUR,
                SUM(COST)                AS COST
            FROM TOMMS.TFA_DISTRICT_HOUR_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """
        params_cost = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_cost += f" AND SUBSTR(STAT_HOUR, 9, 2) IN ({placeholders})"
            params_cost.extend(hours_filter_list)
        sql_cost += """
            GROUP BY DISTRICT_ID, SUBSTR(STAT_HOUR, 1, 10)
            ORDER BY DISTRICT_ID
        """

        cursor.execute(sql_cost, tuple(params_cost))
        rows_cost = cursor.fetchall()
        if not rows_cost:
            etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=0)}"'
            resp = Response(status=204)
            resp.headers["ETag"] = etag
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        # dict: (hh, district_id) -> travel_cost(float)
        cost_map = {}
        # dict: hh -> (mm, dd)  (라벨 생성용)
        mmdd_by_hh = {}

        for row in rows_cost:
            # 기대 형태: (DISTRICT_ID, STAT_HOUR, COST)
            d_id, stat_hour, cost = row[0], str(row[1]), row[2]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            hh = stat_hour[-2:]
            mm, dd = stat_hour[4:6], stat_hour[6:8]
            if hh not in mmdd_by_hh:
                mmdd_by_hh[hh] = (mm, dd)
            if d_id in (1, 2, 3, 4):
                try:
                    cost_map[(hh, d_id)] = int(cost) if cost is not None else None
                except:
                    cost_map[(hh, d_id)] = None

        # ============================================================
        # B) avg_delay — TFA_NODE_15MIN_RESULT × TFA_NODE_INFO (권역×시간대 평균)
        #    3컬럼 고정: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), DELAY(소수1자리)
        # ============================================================
        sql_delay = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(n.STAT_HOUR, 1, 10) AS STAT_HOUR,
                ROUND(AVG(n.DELAY), 1)     AS DELAY
            FROM TOMMS.TFA_NODE_15MIN_RESULT n
            JOIN TOMMS.TFA_NODE_INFO i
              ON i.NODE_ID = n.NODE_ID
            WHERE SUBSTR(n.STAT_HOUR, 1, 8) = ?
        """
        params_delay = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_delay += f" AND SUBSTR(n.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_delay.extend(hours_filter_list)
        sql_delay += """
            GROUP BY i.DISTRICT_ID, SUBSTR(n.STAT_HOUR, 1, 10)
            ORDER BY i.DISTRICT_ID
        """

        cursor.execute(sql_delay, tuple(params_delay))
        rows_delay = cursor.fetchall()

        # dict: (hh, district_id) -> avg_delay(float)
        delay_map = {}
        for row in rows_delay or []:
            d_id, stat_hour, delay = row[0], str(row[1]), row[2]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            hh = stat_hour[-2:]
            if d_id in (1, 2, 3, 4):
                try:
                    delay_map[(hh, d_id)] = float(delay) if delay is not None else None
                except:
                    delay_map[(hh, d_id)] = None

        # ============================================================
        # C) avg_speed — TDA_LINK_HOUR_RESULT × TDA_LINK_INFO (권역×시간대 평균)
        #    3컬럼 고정: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), SPEED(소수1자리)
        # ============================================================
        sql_speed = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(h.STAT_HOUR, 1, 10) AS STAT_HOUR,
                ROUND(AVG(h.SPEED), 1)     AS SPEED
            FROM TOMMS.TDA_LINK_HOUR_RESULT h
            JOIN TOMMS.TDA_LINK_INFO i
            ON i.LINK_ID = h.LINK_ID
            WHERE SUBSTR(h.STAT_HOUR, 1, 8) = ?
        """
        params_speed = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_speed += f" AND SUBSTR(h.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_speed.extend(hours_filter_list)
        sql_speed += """
            GROUP BY i.DISTRICT_ID, SUBSTR(h.STAT_HOUR, 1, 10)
            ORDER BY i.DISTRICT_ID
        """

        cursor.execute(sql_speed, tuple(params_speed))
        rows_speed = cursor.fetchall()

        # dict: (hh, district_id) -> avg_speed(float)
        speed_map = {}
        for row in rows_speed or []:
            d_id, stat_hour, speed = row[0], str(row[1]), row[2]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            hh = stat_hour[-2:]
            if d_id in (1, 2, 3, 4):
                try:
                    speed_map[(hh, d_id)] = float(speed) if speed is not None else None
                except:
                    speed_map[(hh, d_id)] = None

        # ============================================================
        # D) roads — TDA_LINK_HOUR_RESULT × TDA_LINK_INFO
        #    시간·권역별 + 도로명 단위 집계
        #    반환 컬럼: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), ROAD_NAME, VC, VEHS
        #    집계: 같은 ROAD_NAME 끼리 VC 평균(소수2자리), VEHS 합/3 (정수)
        # ============================================================
        sql_roads = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(h.STAT_HOUR, 1, 10) AS STAT_HOUR,
                i.ROAD_NAME,
                h.VC,
                h.VEHS
            FROM TOMMS.TDA_LINK_HOUR_RESULT h
            JOIN TOMMS.TDA_LINK_INFO i
            ON i.LINK_ID = h.LINK_ID
            WHERE SUBSTR(h.STAT_HOUR, 1, 8) = ?
        """
        params_roads = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_roads += f" AND SUBSTR(h.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_roads.extend(hours_filter_list)

        cursor.execute(sql_roads, tuple(params_roads))
        rows_roads = cursor.fetchall()

        # (hh, district_id, road_name) → {"vc_vals": [...], "vehs_sum": int}
        from collections import defaultdict
        roads_acc = defaultdict(lambda: {"vc_vals": [], "road_veh": 0})

        for row in rows_roads or []:
            d_id, stat_hour, road_name, vc, vehs = row[0], str(row[1]), row[2], row[3], row[4]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            if d_id not in (1, 2, 3, 4):
                continue
            hh = stat_hour[-2:]
            key = (hh, d_id, (road_name or "").strip())

            # VC 평균용 수집
            if vc is not None:
                try:
                    roads_acc[key]["vc_vals"].append(float(vc))
                except:
                    pass

            # 차량수 합산
            if vehs is not None:
                try:
                    roads_acc[key]["road_veh"] += int(vehs)
                except:
                    pass

        # (hh, district_id) → [ {road_name, vc_avg, vehs_sum} ... ]
        roads_map = defaultdict(list)
        for (hh, d_id, rname), agg in roads_acc.items():
            # VC 평균(소수 2자리)
            vc_avg = round(sum(agg["vc_vals"]) / len(agg["vc_vals"]), 2) if agg["vc_vals"] else None
            # VEHS 합/3
            vehs_div3 = int(round(agg["road_veh"] / 3.0)) if agg["road_veh"] else 0

            roads_map[(hh, d_id)].append({
                "road_name": rname,
                "road_vc": vc_avg,
                "road_veh": vehs_div3
            })

        # 보기 좋게 정렬(선택): 차량수 내림차순 → VC 평균 내림차순 → 도로명
        for k in roads_map:
            roads_map[k].sort(
                key=lambda x: (
                    x.get("road_veh", 0),
                    (x.get("road_vc") if x.get("road_vc") is not None else -1)
                ),
                reverse=True
            )
            
        # ============================================================
        # E) sa_result — TDA_LINK_HOUR_RESULT × TDA_LINK_INFO
        #    시간·권역·SA_NO 단위 집계
        #    반환: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), SA_NO, AVG_VC(2), AVG_SPEED(1)
        #    congestion 매핑: vc<=0.45 원활 / <=0.70 약간 지체 / <=0.85 지체 / >0.85 매우 지체
        # ============================================================
        sql_sa = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(h.STAT_HOUR, 1, 10) AS STAT_HOUR,
                i.SA_NO,
                ROUND(AVG(h.VC), 2)        AS VC,
                ROUND(AVG(h.SPEED), 1)     AS SPEED
            FROM TOMMS.TDA_LINK_HOUR_RESULT h
            JOIN TOMMS.TDA_LINK_INFO i
            ON i.LINK_ID = h.LINK_ID
            WHERE SUBSTR(h.STAT_HOUR, 1, 8) = ?
        """
        params_sa = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_sa += f" AND SUBSTR(h.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_sa.extend(hours_filter_list)
        sql_sa += """
            GROUP BY i.DISTRICT_ID, SUBSTR(h.STAT_HOUR, 1, 10), i.SA_NO
            ORDER BY i.DISTRICT_ID
        """

        cursor.execute(sql_sa, tuple(params_sa))
        rows_sa = cursor.fetchall()

        # (hh, district_id) -> [ {sa_no, vc, speed, congestion}, ... ]
        from collections import defaultdict
        sa_map = defaultdict(list)

        def map_congestion(vc_val: float) -> str:
            if vc_val is None:
                return "정보없음"
            if vc_val <= 0.45:
                return "원활"
            if vc_val <= 0.70:
                return "약간 지체"
            if vc_val <= 0.85:
                return "지체"
            return "매우 지체"

        for row in rows_sa or []:
            d_id, stat_hour, sa_no, vc, spd = row[0], str(row[1]), row[2], row[3], row[4]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            if d_id not in (1, 2, 3, 4):
                continue
            # SA_NO 없거나 공백이면 스킵(원하면 라벨링 후 포함도 가능)
            if sa_no is None or str(sa_no).strip() == "":
                continue

            hh = stat_hour[-2:]
            vc_val = None
            spd_val = None
            try:
                vc_val = float(vc) if vc is not None else None
            except:
                pass
            try:
                spd_val = float(spd) if spd is not None else None
            except:
                pass

            sa_map[(hh, d_id)].append({
                "sa_no": str(sa_no),
                "vc": vc_val,
                "speed": spd_val,
                "congestion": map_congestion(vc_val)
            })

        # 보기 좋게 정렬(선택): 혼잡 우선 vc 내림차순 → 속도 오름차순 → SA_NO
        for k in sa_map:
            sa_map[k].sort(
                key=lambda x: (
                    (x["vc"] if x["vc"] is not None else -1),
                    (x["speed"] if x["speed"] is not None else 1e9),
                    x["sa_no"]
                ),
                reverse=True
            )

        # ============================================================
        # F) node_result / node_vertex — TFA_NODE_15MIN_RESULT × TFA_NODE_INFO
        #    시간·권역·NODE 단위 집계
        #    반환: DISTRICT_ID, STAT_HOUR(YYYYMMDDHH), NODE_NAME, LAT, LON, DELAY(소수1), LOS(A~F)
        # ============================================================
        sql_node = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(n.STAT_HOUR, 1, 10) AS STAT_HOUR,
                i.CROSS_NAME               AS NODE_NAME,
                i.LAT,
                i.LON,
                ROUND(AVG(n.DELAY), 1)     AS DELAY
            FROM TOMMS.TFA_NODE_15MIN_RESULT n
            JOIN TOMMS.TFA_NODE_INFO i
            ON i.NODE_ID = n.NODE_ID
            WHERE SUBSTR(n.STAT_HOUR, 1, 8) = ?
        """
        params_node = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_node += f" AND SUBSTR(n.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_node.extend(hours_filter_list)
        sql_node += """
            GROUP BY i.DISTRICT_ID, SUBSTR(n.STAT_HOUR, 1, 10), i.CROSS_NAME, i.LAT, i.LON
            ORDER BY i.DISTRICT_ID
        """

        cursor.execute(sql_node, tuple(params_node))
        rows_node = cursor.fetchall()

        from collections import defaultdict

        # (hh, district_id) -> [ {node_name, delay, los}, ... ]
        node_map = defaultdict(list)
        # (hh, district_id) -> [ {los, coordinates:[lat, lon]}, ... ]
        node_vertex_map = defaultdict(list)

        def los_from_delay(d: float) -> str:
            if d is None: return "F"
            if d <= 15:  return "A"
            if d <= 30:  return "B"
            if d <= 50:  return "C"
            if d <= 70:  return "D"
            if d <= 100: return "E"
            return "F"

        for row in rows_node or []:
            d_id, stat_hour, node_name, lat, lon, delay = row
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            if d_id not in (1, 2, 3, 4):
                continue

            # 교차로 이름 없으면 스킵(원하면 라벨링 처리 가능)
            if node_name is None or str(node_name).strip() == "":
                continue

            hh = str(stat_hour)[-2:]

            # 숫자화
            try:
                delay_val = float(delay) if delay is not None else None
            except:
                delay_val = None
            try:
                lat_val = float(lat) if lat is not None else None
                lon_val = float(lon) if lon is not None else None
            except:
                lat_val, lon_val = None, None

            los_val = los_from_delay(delay_val)

            # node_result
            node_map[(hh, d_id)].append({
                "node_name": str(node_name),
                "delay": delay_val,
                "los": los_val
            })

            # node_vertex (좌표가 둘 다 존재할 때만 추가)
            if lat_val is not None and lon_val is not None:
                node_vertex_map[(hh, d_id)].append({
                    "los": los_val,
                    "coordinates": [lat_val, lon_val]   # [lat, lon]
                })

        # 선택: 정렬(지체 큰 순 → 이름)
        for k in node_map:
            node_map[k].sort(
                key=lambda x: (x["delay"] if x["delay"] is not None else -1, x["node_name"]),
                reverse=True
            )
        # 선택: node_vertex는 F→E→D→C→B→A 순으로 정렬하고 싶으면 아래 사용
        los_rank = {"F":6, "E":5, "D":4, "C":3, "B":2, "A":1}
        for k in node_vertex_map:
            node_vertex_map[k].sort(key=lambda x: los_rank.get(x["los"], 6), reverse=True)

        # ============================================================
        # G) link_vertex — 링크별 평균 속도 + 좌표(시퀀스 순)
        #    geometry 파라미터(0/1)에 따라 coordinates 포함 여부 제어
        #    결과: (hh, district) -> [{link_id, speed[, coordinates]] ...]
        # ============================================================
        # geometry 쿼리 파라미터 (기본: 0 → 좌표 제외)
        geometry_param = (request.args.get('geometry') or '0').strip()
        include_geometry = (geometry_param == '1')

        sql_link_speed = """
            SELECT
                i.DISTRICT_ID,
                SUBSTR(h.STAT_HOUR, 1, 10) AS STAT_HOUR,
                h.LINK_ID,
                ROUND(AVG(h.SPEED), 1)     AS SPEED
            FROM TOMMS.TDA_LINK_HOUR_RESULT h
            JOIN TOMMS.TDA_LINK_INFO i
            ON i.LINK_ID = h.LINK_ID
            WHERE SUBSTR(h.STAT_HOUR, 1, 8) = ?
        """
        params_link_speed = [rule_date]
        if hours_filter_list:
            placeholders = ",".join(["?"] * len(hours_filter_list))
            sql_link_speed += f" AND SUBSTR(h.STAT_HOUR, 9, 2) IN ({placeholders})"
            params_link_speed.extend(hours_filter_list)

        sql_link_speed += """
            GROUP BY i.DISTRICT_ID, SUBSTR(h.STAT_HOUR, 1, 10), h.LINK_ID
            ORDER BY i.DISTRICT_ID
        """

        cursor.execute(sql_link_speed, tuple(params_link_speed))
        rows_link_speed = cursor.fetchall()

        from collections import defaultdict

        # (hh, district, link_id) -> speed(float)
        link_speed_map = {}
        # 좌표 조회용 링크 집합 (geometry=1일 때만 사용)
        link_ids_set = set()

        for row in rows_link_speed or []:
            d_id, stat_hour, link_id, speed = row[0], str(row[1]), row[2], row[3]
            try:
                d_id = int(d_id) if d_id is not None else None
            except:
                d_id = None
            if d_id not in (1, 2, 3, 4):
                continue
            hh = stat_hour[-2:]
            lid_str = str(link_id)
            if include_geometry:
                link_ids_set.add(lid_str)
            try:
                spd_val = float(speed) if speed is not None else None
            except:
                spd_val = None
            link_speed_map[(hh, d_id, lid_str)] = spd_val

        # 링크 좌표 일괄 조회 (geometry=1인 경우에만)
        coords_by_link = defaultdict(list)
        if include_geometry and link_ids_set:
            link_ids_list = list(link_ids_set)
            placeholders = ",".join(["?"] * len(link_ids_list))
            sql_vertex = f"""
                SELECT LINK_ID, LINK_SEQ, WGS84_X, WGS84_Y
                FROM TOMMS.LINK_VERTEX
                WHERE LINK_ID IN ({placeholders})
                ORDER BY LINK_ID, LINK_SEQ
            """
            cursor.execute(sql_vertex, tuple(link_ids_list))
            vrows = cursor.fetchall()
            for lid, lseq, x, y in vrows or []:
                lid_str = str(lid)
                try:
                    fx = float(x) if x is not None else None
                    fy = float(y) if y is not None else None
                except:
                    fx, fy = None, None
                coords_by_link[lid_str].append([fx, fy])  # [lon, lat]

        # (hh, district) -> list of {link_id, speed[, coordinates]}
        link_vertex_map = defaultdict(list)
        for (hh, d_id, lid_str), spd in link_speed_map.items():
            item = {
                "link_id": lid_str,
                "speed": spd
            }
            if include_geometry:
                item["coordinates"] = coords_by_link.get(lid_str, [])
            link_vertex_map[(hh, d_id)].append(item)

        # 선택: 보기 좋게 정렬 (속도 느린 순 → 좌표 개수 많은 순)
        for k in link_vertex_map:
            link_vertex_map[k].sort(
                key=lambda x: (
                    (x["speed"] if x["speed"] is not None else 1e9),
                    -len(x.get("coordinates", [])) if include_geometry else 0
                )
            )

        # ------------------------------------------------------------
        # 시간대 순서/라벨
        # ------------------------------------------------------------
        def build_hours_order(present_hours, hours_filter_list):
            order = ["08", "11", "14", "17"]
            return [h for h in order if (h in hours_filter_list)] if hours_filter_list else [h for h in order if h in present_hours]

        hours_present = set(h for (h, d) in cost_map.keys())
        hours_order = build_hours_order(hours_present, hours_filter_list)

        def make_hour_label(mm: str, dd: str, hh: str) -> str:
            hs = int(hh); he = (hs + 1) % 24
            return f"{mm}월 {dd}일 {hs:02d}시 ~ {he:02d}시"

        # ------------------------------------------------------------
        # documents 조립 (판다스 사용 안 함)
        # ------------------------------------------------------------
        documents = []
        for hh in hours_order:
            if hh not in mmdd_by_hh:
                # 해당 시간대에 cost가 없으면 스킵
                continue
            mm, dd = mmdd_by_hh[hh]
            hour_label = make_hour_label(mm, dd, hh)

            districts_bucket = []
            for dno in [1, 2, 3, 4]:
                cost_val  = cost_map.get((hh, dno))
                delay_val = delay_map.get((hh, dno))
                speed_val = speed_map.get((hh, dno))  # ← ✅ 추가

                districts_bucket.append({
                    "district_no": dno,
                    "district_name": district_mapping.get(dno, str(dno)),
                    "avg_delay": delay_val,
                    "avg_speed": speed_val,
                    "travel_cost": int(cost_val) if cost_val is not None else None,  # 이전 결정 반영(정수화)
                    "road_result": roads_map.get((hh, dno), []),
                    "sa_result": sa_map.get((hh, dno), []),
                    "node_result": node_map.get((hh, dno), []),
                    "link_vertex": link_vertex_map.get((hh, dno), []),
                    "node_vertex": node_vertex_map.get((hh, dno), [])
                })

            documents.append({"hour_label": hour_label, "districts": districts_bucket})

        # ------------------------------------------------------------
        # 응답 + 헤더
        # ------------------------------------------------------------
        total_rows = len(rows_cost) + (len(rows_delay) if rows_delay else 0)
        etag = f'"{make_etag(rule_date, hours_key_for_etag, total_rows=total_rows)}"'

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

        payload = {"status": "success", "rule_date": rule_date, "documents": documents}
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

#  [ 신호운영 4 - 교차로별 효과지표 분석정보 ]  =========================================================

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

        # 추가: TIMEINT 출력 라벨 매핑
        TIMEINT_LABELS = {
            '00-15': '00분 ~ 15분',
            '15-30': '15분 ~ 30분',
            '30-45': '30분 ~ 45분',
            '45-00': '45분 ~ 00분'
        }

        def format_timeint(s: str) -> str:
            return TIMEINT_LABELS.get(s, s)
        
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

        # 숫자화 (DISTRICT 포함)
        for col in ["CROSS_ID","APPR_ID","DIRECTION","CROSS_TYPE","DISTRICT"]:
            if col in df_info.columns:
                df_info[col] = pd.to_numeric(df_info[col], errors="coerce")

        # ✅ CROSS_ID → DISTRICT 매핑(중복 발생 시 첫 값 사용)
        cid_dist = (
            df_info[['CROSS_ID','DISTRICT']]
            .dropna()
            .drop_duplicates(subset=['CROSS_ID'])
            .astype({'CROSS_ID':'int64','DISTRICT':'int64'})
        )
        district_by_cross_id: dict[int,int] = dict(zip(cid_dist['CROSS_ID'], cid_dist['DISTRICT']))

        # ✅ 여기서 merge하여 DISTRICT, CROSS_ID를 부여(기존 로직 유지)
        df = df.merge(df_info[['NODE_ID','CROSS_ID','DISTRICT']], on='NODE_ID', how='left')

        # 메타 프레임(기존 로직 유지)
        df_node_meta = (
            df_info
            .drop_duplicates(subset=['NODE_ID'])
            [['NODE_ID','NODE_NAME','CROSS_TYPE','INT_TYPE']]
            .set_index('NODE_ID')
        )
        df_appr_meta = df_info[['NODE_ID','APPR_ID','DIRECTION','APPR_NAME']].dropna()
        if 'APPR_ID' in df_appr_meta.columns:
            df_appr_meta['APPR_ID'] = pd.to_numeric(df_appr_meta['APPR_ID'], errors="coerce")

        # -------------------------------------------- 4) 일일 총 교통량 맵
        
        unique_cross_ids = sorted({int(c) for c in df['CROSS_ID'].dropna().astype(int).tolist()})
        daily_volume_map = {}
        if unique_cross_ids:
            placeholders = ",".join(["?"] * len(unique_cross_ids))
            query = f"""
                SELECT CROSS_ID, NVL(VOL, 0) AS VOL
                FROM TOMMS.STAT_DAY_CROSS
                WHERE STAT_DAY = ?
                AND INFRA_TYPE = 'SMT'
                AND CROSS_ID IN ({placeholders})
            """
            params = [latest_date] + unique_cross_ids
            cursor.execute(query, params)
            for cross_id_val, vol_val in cursor.fetchall():
                try:
                    c = int(cross_id_val)
                    v = int(float(vol_val or 0))  # 숫자/문자/NULL 어떤 형태여도 안전하게 0 또는 정수
                except Exception:
                    c, v = None, 0
                if c is not None:
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
        def _cross_type_label(ct) -> str | None:
            """
            CROSS_TYPE 값(3/4/5)을 한글 라벨로 변환.
            3 -> '3지 교차로', 4 -> '4지 교차로', 5 -> '5지 교차로'
            매핑 밖의 값/결측은 None 반환.
            """
            mapping = {3: "3지 교차로", 4: "4지 교차로", 5: "5지 교차로"}
            try:
                key = int(ct) if pd.notna(ct) else None
            except Exception:
                key = None
            return mapping.get(key)

        def _district_label(d) -> str | None:
            """
            DISTRICT 코드(1~4)를 권역 이름 문자열로 변환.
            1 -> '교동지구', 2 -> '송정지구', 3 -> '중심지구', 4 -> '경포지구'
            매핑 불가/결측은 None
            """
            mapping = {1: "교동지구", 2: "송정지구", 3: "중심지구", 4: "경포지구"}
            try:
                key = int(d) if pd.notna(d) else None
            except Exception:
                key = None
            return mapping.get(key)

        
        nodes = []

        for node_id, df_node in df.groupby('NODE_ID'):
            if node_id not in df_node_meta.index:
                continue

            node_meta = df_node_meta.loc[node_id]
            node_name = node_meta['NODE_NAME']
            cross_id  = df_node['CROSS_ID'].dropna().iloc[0] if not df_node['CROSS_ID'].dropna().empty else None
            sa_no     = df_node['SA_NO'].dropna().iloc[0] if 'SA_NO' in df_node.columns and not df_node['SA_NO'].dropna().empty else None

            # ✅ CROSS_ID로 DISTRICT 찾기 (sql_info 기반)
            district_val = None
            if pd.notna(cross_id):
                try:
                    district_val = district_by_cross_id.get(int(cross_id))
                    if district_val is not None:
                        district_val = int(district_val)
                except Exception:
                    district_val = None

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
                    daily_total_val = int(daily_volume_map.get(int(cross_id), 0))
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
                "district": _district_label(district_val),
                "sa_no": sa_no,
                "cross_type": _cross_type_label(node_meta['CROSS_TYPE']),
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
                    "timeint": format_timeint(slice_label),
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









#  [ 교통관리 1 - 예측교통량 패턴비교 분석정보 ]  =========================================================

@app.route('/management/compare-traffic-vol', methods=['GET'])
def compare_traffic_vol():
    pass

#  [ 교통관리 2 - 딥러닝 모델 성능 지표 ]  =========================================================

@app.route('/management/deep-learning-overview', methods=['GET'])
def deep_learning_overview():
    pass

#  [ 교통관리 3 - SA(Sub Area) 그룹별 교통혼잡 분석정보 ]  =========================================================

@app.route('/management/sa-group-info', methods=['GET'])
def congested_info():
    try:
        # =========================================================
        # ✅   기준 시간/헤더용 현재 시간 계산
        # =========================================================
        now_kst = datetime.now(KST)

        # ▶ 테스트용 (배포 시 resolve_dataset_date(now_kst)로 대체)
        rule_date = "20250701"
        # rule_date = resolve_dataset_date(now_kst)

        # ▶ 다음 업데이트 시각(매일 06:00 KST)
        next_update = now_kst.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_kst >= next_update:
            next_update += timedelta(days=1)
        x_next_update = next_update.isoformat()

        # =========================================================
        # ✅   DB 연결
        # =========================================================
        conn = get_connection()
        cursor = conn.cursor()

        # =========================================================
        # ✅   ETag 선판단(조기 304): 하루 고정 + 08/11/14/17 통합 뷰
        #     total_rows := SA_VERTEX + NODE_15MIN(4개 시간대) + LINK_HOUR(4개 시간대) + NODE_INFO(필터된 개수)
        # =========================================================
        HOURS = ["08", "11", "14", "17"]
        stat_hours = tuple(rule_date + h for h in HOURS)
        ph = ",".join(["?"] * len(stat_hours))

        # SA_VERTEX count
        cursor.execute("SELECT COUNT(*) FROM TOMMS.SIM_SA_VERTEX")
        sa_cnt = int(cursor.fetchone()[0] or 0)

        # NODE_15MIN count
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM TOMMS.TFA_NODE_15MIN_RESULT
            WHERE STAT_HOUR IN ({ph})
        """, stat_hours)
        node_cnt = int(cursor.fetchone()[0] or 0)

        # LINK_HOUR count (congested_road 근거)
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM TOMMS.TDA_LINK_HOUR_RESULT
            WHERE STAT_HOUR IN ({ph})
        """, stat_hours)
        link_cnt = int(cursor.fetchone()[0] or 0)

        # NODE_INFO count (INT_TYPE/LAT/LON 모두 존재)
        cursor.execute("""
            SELECT COUNT(*)
            FROM TOMMS.TFA_NODE_INFO
            WHERE INT_TYPE IS NOT NULL
              AND LAT IS NOT NULL
              AND LON IS NOT NULL
        """)
        nodeinfo_cnt = int(cursor.fetchone()[0] or 0)

        etag_val = make_etag(rule_date, {8, 11, 14, 17}, sa_cnt + node_cnt + link_cnt + nodeinfo_cnt)
        etag_hdr = f'"{etag_val}"'

        inm = _normalize_inm(request.headers.get("If-None-Match", ""))
        if inm == etag_val:
            resp = Response(status=304)
            resp.headers["ETag"] = etag_hdr
            resp.headers["X-Dataset-Date"] = rule_date
            resp.headers["X-Next-Update"] = x_next_update
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            return resp

        # =========================================================
        # ✅   SA Vertex → sa_vertex [{sa_no, coordinates:[[lon,lat],…]}]
        #      - LINK_SEQ 기준 정렬, 좌표는 [lon, lat]
        # =========================================================
        cursor.execute("""
            SELECT SA_NO, DISTRICT_ID, LINK_SEQ, LAT, LON
            FROM TOMMS.SIM_SA_VERTEX
            ORDER BY SA_NO, LINK_SEQ
        """)
        vrows = cursor.fetchall() or []

        from collections import defaultdict
        coords_by_sa = defaultdict(list)  # SA_NO -> [[lon,lat], ...]
        for sa_no, dist_id, link_seq, lat, lon in vrows:
            try:
                fx = float(lon) if lon is not None else None   # lon
                fy = float(lat) if lat is not None else None   # lat
            except:
                fx, fy = None, None
            coords_by_sa[str(sa_no)].append([fx, fy])

        sa_vertex = [{"sa_no": sa_no, "coordinates": coords}
                     for sa_no, coords in coords_by_sa.items()]

        # =========================================================
        # ✅   시간대별 혼잡 교차 TOP5 (DELAY 평균 기준)
        #      - ROW_NUMBER()로 시간대별 5개
        #      - traffic_status 추가 (delay 기준)
        # =========================================================
        cursor.execute(f"""
            SELECT hh, cross_name, district_name, avg_delay
            FROM (
                SELECT
                    SUBSTR(r.STAT_HOUR, 9, 2) AS hh,          -- 'HH'
                    n.CROSS_NAME           AS cross_name,
                    d.DISTRICT_NAME        AS district_name,
                    AVG(r.DELAY)           AS avg_delay,
                    ROW_NUMBER() OVER (
                        PARTITION BY SUBSTR(r.STAT_HOUR, 9, 2)
                        ORDER BY AVG(r.DELAY) DESC
                    ) AS rn
                FROM TOMMS.TFA_NODE_15MIN_RESULT r
                JOIN TOMMS.TFA_NODE_INFO n
                  ON n.NODE_ID = r.NODE_ID
                JOIN TOMMS.SIM_DISTRICT_INFO d
                  ON d.DISTRICT_ID = n.DISTRICT_ID
                WHERE r.STAT_HOUR IN ({ph})
                GROUP BY SUBSTR(r.STAT_HOUR, 9, 2),
                         n.CROSS_NAME, d.DISTRICT_NAME
            )
            WHERE rn <= 5
            ORDER BY hh, avg_delay DESC
        """, stat_hours)

        top_rows_all = cursor.fetchall() or []

        def make_label(hh: str) -> str:
            mm = int(rule_date[4:6]); dd = int(rule_date[6:8])
            start_h = int(hh); end_h = (start_h + 1) % 24
            return f"{mm}월 {dd}일 {start_h:02d}시 ~ {end_h:02d}시"

        def status_by_delay(d: float | None) -> str | None:
            if d is None:
                return None
            if d <= 30: return "원활"
            if d <= 50: return "약간 혼잡"
            if d <= 70: return "혼잡"
            return "매우 혼잡"

        by_hour_cross = {h: [] for h in HOURS}
        for hh, cross_name, district_name, avg_delay in top_rows_all:
            try:
                delay_val = None if avg_delay is None else round(float(avg_delay), 1)
            except:
                delay_val = None
            by_hour_cross[str(hh)].append({
                "cross_name": cross_name,
                "district_name": district_name,
                "delay": delay_val,
                "traffic_status": status_by_delay(delay_val)  # ← 추가
            })

        congested_cross = [
            {"hour_label": make_label(h), "data": by_hour_cross[h][:5]}
            for h in HOURS
        ]

        # =========================================================
        # ✅   시간대별 혼잡 도로 TOP5 (평균 SPEED 가장 낮은 도로)
        #      - SPEED > 5만 집계 포함
        #      - ROAD_NAME 단위 평균 속도(소수점 1자리)
        #      - DISTRICT_NAME은 해당 도로의 최빈(모드) 구역명
        #      - traffic_status 추가 (avg_speed 기준)
        # =========================================================
        cursor.execute(f"""
            WITH base AS (
                SELECT
                    SUBSTR(h.STAT_HOUR, 9, 2) AS hh,
                    li.ROAD_NAME               AS road_name,
                    di.DISTRICT_NAME           AS district_name,
                    h.SPEED                    AS speed
                FROM TOMMS.TDA_LINK_HOUR_RESULT h
                JOIN TOMMS.TDA_LINK_INFO li
                  ON li.LINK_ID = h.LINK_ID
                JOIN TOMMS.SIM_DISTRICT_INFO di
                  ON di.DISTRICT_ID = li.DISTRICT_ID
                WHERE h.STAT_HOUR IN ({ph})
                  AND h.SPEED > 5
            ),
            agg AS (
                SELECT hh, road_name, AVG(speed) AS avg_speed
                FROM base
                GROUP BY hh, road_name
            ),
            dist_mode AS (
                SELECT hh, road_name, district_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY hh, road_name
                           ORDER BY COUNT(*) DESC, district_name
                       ) AS rn
                FROM base
                GROUP BY hh, road_name, district_name
            )
            SELECT hh,
                   road_name,
                   ROUND(avg_speed, 1) AS avg_speed,
                   (SELECT district_name FROM dist_mode dm
                     WHERE dm.hh = a.hh AND dm.road_name = a.road_name AND dm.rn = 1) AS district_name
            FROM agg a
        """, stat_hours)

        def status_by_speed(s: float | None) -> str | None:
            if s is None:
                return None
            if s <= 10: return "매우 혼잡"
            if s <= 20: return "혼잡"
            if s <= 30: return "약간 혼잡"
            return "원활"

        road_rows = cursor.fetchall() or []
        by_hour_road = {h: [] for h in HOURS}
        for hh, road_name, avg_speed, district_name in road_rows:
            try:
                spd_val = None if avg_speed is None else round(float(avg_speed), 1)
            except:
                spd_val = None
            by_hour_road[str(hh)].append({
                "road_name": road_name,
                "avg_speed": spd_val,
                "district_name": district_name,
                "traffic_status": status_by_speed(spd_val)  # ← 추가
            })
        for h in HOURS:
            by_hour_road[h].sort(key=lambda x: (float('inf') if x["avg_speed"] is None else x["avg_speed"]))
        congested_road = [
            {"hour_label": make_label(h), "data": by_hour_road[h][:5]}
            for h in HOURS
        ]

        # =========================================================
        # ✅   node_vertex 추가 (TFA_NODE_INFO)
        #      - INT_TYPE/LAT/LON 모두 존재
        #      - 좌표는 [lat, lon]
        # =========================================================
        cursor.execute("""
            SELECT INT_TYPE, LAT, LON
            FROM TOMMS.TFA_NODE_INFO
            WHERE INT_TYPE IS NOT NULL
            AND LAT IS NOT NULL
            AND LON IS NOT NULL
            AND SA_NO IS NOT NULL
        """)
        nrows = cursor.fetchall() or []
        node_vertex = []
        for int_type, lat, lon in nrows:
            try:
                la = float(lat) if lat is not None else None
                lo = float(lon) if lon is not None else None
            except:
                la, lo = None, None
            if la is not None and lo is not None and int_type is not None:
                node_vertex.append({
                    "int_type": str(int_type),
                    "coordinates": [la, lo]  # [lat, lon]
                })

        # =========================================================
        # ✅   Payload + 헤더
        # =========================================================
        payload = {
            "status": "success",
            "sa_vertex": sa_vertex,                # [lon, lat]
            "congested_cross": congested_cross,    # delay 기반 traffic_status 포함
            "congested_road": congested_road,      # avg_speed 기반 traffic_status 포함
            "node_vertex": node_vertex             # [lat, lon]
        }

        body = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        resp = Response(body, content_type="application/json; charset=utf-8", status=200)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        resp.headers["ETag"] = etag_hdr
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

#  [ 교통관리 4 - 혼잡교차로 신호최적화 효과검증 ]  =========================================================

@app.route('/management/signal-optimize', methods=['GET'])
def cross_optimize():
    pass









# ========================================================= [ 서버실행 ]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101, debug=False)