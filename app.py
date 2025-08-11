import datetime, pyodbc, os, subprocess, json, pathlib
import pandas as pd
import numpy as np

from decimal import Decimal
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from windows import set_dpi_awareness


set_dpi_awareness()
app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# ========================================================= [ 현재시간 ]

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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

# Decimal 처리 함수
def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return obj

# 권역별 매핑
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

# 지체시간 기준 LOS
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
        conn = get_connection()
        cursor = conn.cursor()

        # [1] VISUM_ZONE_INFO → ZONE_ID, LAT, LON
        cursor.execute("SELECT ZONE_ID, LAT, LON FROM VISUM_ZONE_INFO")
        rows_info = cursor.fetchall()
        df_zone_info = pd.DataFrame(
            [[str(r[0]), float(r[1]), float(r[2])] for r in rows_info],
            columns=["ZONE_ID", "LAT", "LON"]
        )

        # [2] VISUM_ZONE_OD → OD Matrix
        cursor.execute("""
            SELECT FROM_ZONE_ID, FROM_ZONE_NAME,
            TO_ZONE_ID, TO_ZONE_NAME,
            AUTO_MATRIX_VALUE, BUS_MATRIX_VALUE, HGV_MATRIX_VALUE
            FROM VISUM_ZONE_OD
        """)
        rows_od = cursor.fetchall()
        df_od = pd.DataFrame([
            [
                str(r[0]), str(r[1]), str(r[2]), str(r[3]),
                float(r[4] or 0), float(r[5] or 0), float(r[6] or 0)
            ] for r in rows_od
        ], columns=[
            "FROM_ZONE_ID", "FROM_ZONE_NAME",
            "TO_ZONE_ID", "TO_ZONE_NAME",
            "AUTO_MATRIX_VALUE", "BUS_MATRIX_VALUE", "HGV_MATRIX_VALUE"
        ])

        # [3] OD_MATRIX_VALUE 계산
        df_od["OD_MATRIX_VALUE"] = (
            df_od["AUTO_MATRIX_VALUE"] +
            df_od["BUS_MATRIX_VALUE"] +
            df_od["HGV_MATRIX_VALUE"]
        ).round(6)

        # [4] 상위 5개 TO_ZONE 추출
        df_top5 = (
            df_od.sort_values(["FROM_ZONE_ID", "OD_MATRIX_VALUE"], ascending=[True, False])
            .groupby("FROM_ZONE_ID")
            .head(5)
            .reset_index(drop=True)
        )

        # [5] FROM ZONE 좌표 병합
        df_top5 = df_top5.merge(
            df_zone_info.rename(columns={
                "ZONE_ID": "FROM_ZONE_ID",
                "LAT": "FROM_LAT",
                "LON": "FROM_LON"
            }),
            on="FROM_ZONE_ID", how="left"
        )

        # [6] TO ZONE 좌표 병합
        df_top5 = df_top5.merge(
            df_zone_info.rename(columns={
                "ZONE_ID": "TO_ZONE_ID",
                "LAT": "TO_LAT",
                "LON": "TO_LON"
            }),
            on="TO_ZONE_ID", how="left"
        )

        # [7] 변환 → 중차 구조로 그룹
        result = []
        grouped = df_top5.groupby(["FROM_ZONE_ID", "FROM_ZONE_NAME", "FROM_LAT", "FROM_LON"])
        for (from_id, from_name, from_lat, from_lon), group_df in grouped:
            destinations = []
            for _, row in group_df.iterrows():
                destinations.append({
                    "to_zone_name": row["TO_ZONE_NAME"],
                    "coordinates": [row["TO_LAT"], row["TO_LON"]],
                    "value": round(row["OD_MATRIX_VALUE"], 2)
                })
            result.append({
                "coordinates": [from_lat, from_lon],
                "from_zone_name": from_name,
                "destination": destinations
            })

        conn.close()
        print(f"✅ [ {get_current_time()} ] OD Matrix 응답 {len(result)}개 그룹 완료")

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ [ {get_current_time()} ] OD Matrix 처리 에러: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "OD 분석 중 에러 발생",
            "error": str(e),
            "timestamp": get_current_time()
        }), 500

# ========================================================= [ 모니터링 3 - 분석지역별 교통흐름 통계정보 ] - 4k 

@app.route('/monitoring/statistics-traffic-flow/node-result', methods=['GET'])
def statistics_traffic_flow():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 📌 1. 최신 STAT_HOUR 추출
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

        # 📌 2. NODE_RESULT: delay만 간단히 조회
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, NODE_ID, DELAY, VEHS
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        result_rows = [tuple(row) for row in cursor.fetchall()]

        if not result_rows:
            return jsonify({"status": "fail", "message": "NODE_RESULT 데이터가 없습니다."}), 404

        df_result = pd.DataFrame(result_rows, columns=["DISTRICT", "STAT_HOUR", "NODE_ID", "DELAY", "VEHS"])
        df_result["DELAY"] = pd.to_numeric(df_result["DELAY"], errors='coerce')
        df_result["VEHS"] = pd.to_numeric(df_result["VEHS"], errors='coerce').fillna(0).astype(int)
        df_result["DISTRICT"] = df_result["DISTRICT"].apply(lambda x: int(x) if pd.notna(x) else None)

        # 📌 3. NODE_INFO: 위치 정보만 조회
        cursor.execute("""
            SELECT NODE_ID, LAT, LON
            FROM NODE_INFO
        """)
        info_rows = [tuple(row) for row in cursor.fetchall()]

        if not info_rows:
            return jsonify({"status": "fail", "message": "NODE_INFO 데이터가 없습니다."}), 404

        df_info = pd.DataFrame(info_rows, columns=["NODE_ID", "LAT", "LON"])

        # 📌 4. 병합
        df_merged = pd.merge(df_result, df_info, on="NODE_ID", how="left")
        df_filtered = df_merged.dropna(subset=["LAT", "LON", "DISTRICT"])

        # 📌 5. DISTRICT 명칭 적용
        df_filtered["DISTRICT_NAME"] = df_filtered["DISTRICT"].map(district_mapping)
        
        # 📌 6. stat_hour를 readable hourly label로 변환
        df_filtered["STAT_HOUR_LABEL"] = df_filtered["STAT_HOUR"].apply(
            lambda x: hourly_mapping.get(x[-2:], x)
        )
        df_filtered["TARGET_DATE"] = df_filtered["STAT_HOUR"].str[:8]

        # 📌 7. 그룹화: STAT_HOUR + DISTRICT_NAME + NODE_ID
        grouped = df_filtered.groupby(["TARGET_DATE", "STAT_HOUR_LABEL", "DISTRICT_NAME", "NODE_ID"]).agg({
            "DELAY": "mean",
            "LAT": "first",
            "LON": "first"
        }).reset_index()

        grouped["DELAY"] = grouped["DELAY"].round(2)
        grouped["LOS"] = grouped["DELAY"].apply(get_los)

        # 📌 🚀 추가: VEH 집계
        vehs_df = df_filtered.groupby(["TARGET_DATE", "STAT_HOUR_LABEL", "DISTRICT_NAME"])["VEHS"].sum().reset_index()
        vehs_df["VEHS"] = vehs_df["VEHS"].astype(int)

        # 📌 7. GeoJSON + VEHS 조합
        result_json = dict()

        for (hour_label, district), group_df in grouped.groupby(["STAT_HOUR_LABEL", "DISTRICT_NAME"]):
            features = []
            for idx, row in group_df.iterrows():
                feature = {
                    "type": "Feature",
                    "id": idx,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["LAT"]), float(row["LON"])]
                    },
                    "properties": {
                        "los": row["LOS"]
                    }
                }
                features.append(feature)

            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            veh_row = vehs_df[
                (vehs_df["STAT_HOUR_LABEL"] == hour_label) &
                (vehs_df["DISTRICT_NAME"] == district)
            ]
            vehs_value = int(veh_row["VEHS"].values[0]) if not veh_row.empty else 0
            
            result_json.setdefault(hour_label, {})[district] = {
                "VEHS": vehs_value,
                "GEOJSON": geojson
            }

        # 📌 8. 응답
        json_data = json.dumps({"status": "success", "target_date": latest_date, "data": result_json}, ensure_ascii=False, default=convert_decimal)
        return Response(json_data, content_type='application/json; charset=utf-8')

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

        # 📌 1. 가장 최신 날짜(YYYYMMDD) 추출
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

        # 📌 2. 해당 날짜 데이터 조회
        cursor.execute("""
            SELECT STAT_HOUR, TIMEINT, NODE_ID, QLEN, VEHS, DELAY, STOPS
            FROM NODE_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        rows = cursor.fetchall()
        rows = [tuple(row) for row in rows]

        if not rows:
            return jsonify({"status": "fail", "message": "해당 날짜에 대한 교차로 데이터가 없습니다."}), 404

        # 📌 3. DataFrame 생성
        df = pd.DataFrame(rows, columns=["STAT_HOUR", "TIMEINT", "NODE_ID", "QLEN", "VEHS", "DELAY", "STOPS"])
        df[["QLEN", "VEHS", "DELAY", "STOPS"]] = df[["QLEN", "VEHS", "DELAY", "STOPS"]].apply(pd.to_numeric, errors='coerce')

        # 📌 4. 날짜와 시간 분리
        df["DATE"] = df["STAT_HOUR"].str[:8]
        df["HOUR"] = df["STAT_HOUR"].str[8:10]

        # 📌 5. 평균값 계산
        df_avg = df.groupby(["DATE", "HOUR", "NODE_ID"], as_index=False).agg({
            "QLEN": "mean",
            "VEHS": "mean",
            "DELAY": "mean",
            "STOPS": "mean"
        }).round({"QLEN": 2, "DELAY": 2, "STOPS": 2})

        # 📌 6. 교차로 이름 병합
        cursor.execute("SELECT NODE_ID, CROSS_NAME FROM NODE_INFO")
        node_info = cursor.fetchall()
        node_info = [tuple(row) for row in node_info]
        df_node_info = pd.DataFrame(node_info, columns=["NODE_ID", "NODE_NAME"])
        df_node_info = df_node_info.drop_duplicates(subset="NODE_ID")

        df_merged = df_avg.merge(df_node_info, on="NODE_ID", how="left")

        # 📌 7. LOS 등급 계산
        def get_los(delay):
            if delay < 15: return "A"
            elif delay < 30: return "B"
            elif delay < 50: return "C"
            elif delay < 70: return "D"
            elif delay < 100: return "E"
            elif delay < 220: return "F"
            elif delay < 340: return "FF"
            else: return "FFF"

        df_merged["LOS"] = df_merged["DELAY"].apply(get_los)

        # 📌 8. 최종 컬럼 정리
        df_merged = df_merged[[
            "DATE", "HOUR", "NODE_NAME", "QLEN", "VEHS", "DELAY", "STOPS", "LOS"
        ]]

        # 📌 9. 유효한 교차로만 필터링 + VEHS 정수 변환
        df_merged = df_merged[df_merged["NODE_NAME"].notna()]
        df_merged["VEHS"] = df_merged["VEHS"].round(0).astype("Int64")

        # 📌 10. 중복 제거
        df_merged = df_merged.drop_duplicates()

        # 📌 11. JSON 변환 → { "target_date": "YYYYMMDD", "data": { "한글시간라벨": [ ... ] } }
        result_dict = {}
        target_date = None
        mapped_data = {}

        for (date, hour), group in df_merged.groupby(["DATE", "HOUR"]):
            records = (
                group.drop(columns=["DATE", "HOUR"])
                    .replace({np.nan: None})
                    .to_dict(orient="records")
            )

            # 첫 번째 date 값을 target_date로 설정
            if not target_date:
                target_date = date

            # hourly label로 변환
            hour_label = hourly_mapping.get(hour, hour)  # 매핑 안되면 그대로 hour 사용
            mapped_data[hour_label] = records

        # ✅ 응답 반환
        return app.response_class(
            response=json.dumps({
                "target_date": target_date,
                "data": mapped_data
            }, ensure_ascii=False, allow_nan=False),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "교차로 결과 조회 중 오류 발생",
            "error": str(e),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500






# ========================================================= [ 신호운영 1 - 도로축별 통계정보 ]

@app.route('/signal/vttm-result', methods=['GET'])
def vttm_result_summary():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 📌 1. 가장 최신 날짜(YYYYMMDD) 추출
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

        # 📌 2. 해당 날짜의 교통 결과 데이터 조회
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, FROM_NODE_NAME, TO_NODE_NAME, UPDOWN, DISTANCE, TRAVEL_TIME
            FROM VTTM_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        rows = cursor.fetchall()

        columns = ['DISTRICT', 'STAT_HOUR', 'FROM_NODE_NAME', 'TO_NODE_NAME', 'UPDOWN', 'DISTANCE', 'TRAVEL_TIME']

        # 📌 데이터 전처리 구조 초기화
        grouped_data = defaultdict(lambda: defaultdict(list))
        pair_buffer = defaultdict(lambda: defaultdict(dict))  # (district, hour_label) => segment_key => {updown}

        for row in rows:
            record = dict(zip(columns, row))
            district_id = record['DISTRICT']
            stat_hour = record['STAT_HOUR']
            hour_code = stat_hour[-2:]
            hour_label = hourly_mapping.get(hour_code, hour_code)
            district_name = district_mapping.get(district_id, f"기타지역-{district_id}")

            from_node = record['FROM_NODE_NAME']
            to_node = record['TO_NODE_NAME']
            updown = str(record['UPDOWN'])
            distance = float(record['DISTANCE'] or 0)
            travel_time_val = float(record['TRAVEL_TIME'] or 0)

            travel_time = round(travel_time_val, 1) if travel_time_val > 0 else 0.0
            travel_speed = round((distance / travel_time_val) * 3.6, 1) if travel_time_val > 0 else 0.0
            travel_cost = 99.9

            segment_key = tuple(sorted([from_node, to_node]))

            pair_buffer[(district_name, hour_label)][segment_key][updown] = {
                "from_node": from_node,
                "to_node": to_node,
                "travel_time": travel_time,
                "travel_speed": travel_speed,
                "travel_cost": travel_cost
            }

        # 📌 완성된 쌍만 정리
        for (district_name, hour_label), segment_dict in pair_buffer.items():
            for segment_key, directions in segment_dict.items():
                if '0' in directions and '1' in directions:
                    from_node_data = directions['0']
                    to_node_data = directions['1']

                    record = {
                        from_node_data['from_node']: {
                            "travel_cost": from_node_data['travel_cost'],
                            "travel_speed": from_node_data['travel_speed'],
                            "travel_time": from_node_data['travel_time']
                        },
                        to_node_data['from_node']: {
                            "travel_cost": to_node_data['travel_cost'],
                            "travel_speed": to_node_data['travel_speed'],
                            "travel_time": to_node_data['travel_time']
                        }
                    }
                    grouped_data[district_name][hour_label].append(record)

        return jsonify({
            "status": "success",
            "target_date": latest_date,
            "data": grouped_data
        }), 200

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "교차로 결과 조회 중 오류 발생",
            "error": str(e),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    hour_filter = request.args.get('hour')  # '08', '11', '14', '17' 중 하나
    label = hourly_mapping.get(hour_filter)
    if not label:
        return jsonify({
            "status": "fail",
            "message": f"유효하지 않은 hour 파라미터입니다: {hour_filter}",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 400
            
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 📌 1. 최신 일자(STAT_HOUR → YYYYMMDD) 조회
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
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }), 404

        latest_date = latest_date_row[0]

        # 📌 2. NODE_DIR_RESULT 조회
        cursor.execute("""
            SELECT DISTRICT, STAT_HOUR, TIMEINT, NODE_ID, CROSS_ID, SA_NO,
                APPR_ID, DIRECTION, QLEN, VEHS, DELAY, STOPS
            FROM NODE_DIR_RESULT
            WHERE SUBSTR(STAT_HOUR, 1, 8) = ?
        """, [latest_date])
        node_dir_rows = [tuple(row) for row in cursor.fetchall()]
        node_dir_columns = ['DISTRICT', 'STAT_HOUR', 'TIMEINT', 'NODE_ID', 'CROSS_ID', 'SA_NO',
                            'APPR_ID', 'DIRECTION', 'QLEN', 'VEHS', 'DELAY', 'STOPS']
        df_node_dir = pd.DataFrame(node_dir_rows, columns=node_dir_columns)
        for col in df_node_dir.columns:
            df_node_dir[col] = df_node_dir[col].map(lambda x: float(x) if isinstance(x, Decimal) else x)

        # 📌 3. NODE_DIR_INFO 조회
        cursor.execute("""
            SELECT CROSS_ID, DISTRICT, NODE_ID, NODE_NAME, CROSS_TYPE, INT_TYPE,
                APPR_ID, DIRECTION, APPR_NAME
            FROM NODE_DIR_INFO
        """)
        node_dir_info_rows = [tuple(row) for row in cursor.fetchall()]
        node_dir_info_columns = ['CROSS_ID', 'DISTRICT', 'NODE_ID', 'NODE_NAME',
                                'CROSS_TYPE', 'INT_TYPE', 'APPR_ID', 'DIRECTION', 'APPR_NAME']
        df_node_info = pd.DataFrame(node_dir_info_rows, columns=node_dir_info_columns)
        for col in df_node_info.columns:
            df_node_info[col] = df_node_info[col].map(lambda x: float(x) if isinstance(x, Decimal) else x)

        # 📌 메타 정보 추출
        df_node_meta = df_node_info.drop_duplicates(subset=['NODE_ID'])[
            ['NODE_ID', 'NODE_NAME', 'CROSS_TYPE', 'INT_TYPE']
        ].set_index('NODE_ID')

        df_appr_meta = df_node_info[[
            'NODE_ID', 'APPR_ID', 'DIRECTION', 'APPR_NAME'
        ]].dropna()

        # 📌 4. 결과 가공
        grouped_result = {}

        for stat_hour, df_hour in df_node_dir.groupby('STAT_HOUR'):
            hh = stat_hour[-2:]
            if hh != hour_filter:
                continue

            grouped_result[label] = {}

            for node_id, df_node_alltime in df_hour.groupby('NODE_ID'):
                if node_id not in df_node_meta.index:
                    continue

                node_meta = df_node_meta.loc[node_id]
                node_name = node_meta['NODE_NAME']
                cross_id = df_node_alltime['CROSS_ID'].iloc[0]
                sa_no = df_node_alltime['SA_NO'].iloc[0]

                result_dict = {
                    "CROSS_ID": int(float(cross_id)),
                    "SA_NO": sa_no,
                    "CROSS_TYPE": int(float(node_meta['CROSS_TYPE'])),
                    "INT_TYPE": node_meta['INT_TYPE']
                }

                hourly_summary = {}
                all_vehs_total, all_delay_sum, all_delay_count = 0, 0.0, 0

                for appr_id, df_appr in df_node_alltime.groupby('APPR_ID'):
                    vehs = int(df_appr['VEHS'].sum(skipna=True) or 0)
                    delay_vals = df_appr['DELAY'].dropna().astype(float).tolist()
                    delay_avg = round(sum(delay_vals) / len(delay_vals), 1) if delay_vals else 0.0
                    los = get_los(delay_avg)

                    all_vehs_total += vehs
                    all_delay_sum += sum(delay_vals)
                    all_delay_count += len(delay_vals)

                    match = df_appr_meta[(df_appr_meta['NODE_ID'] == node_id) & (df_appr_meta['APPR_ID'] == appr_id)]
                    appr_name = match.iloc[0]['APPR_NAME'] if not match.empty else "미지정"

                    hourly_summary[appr_name] = {
                        "VEHS": int(vehs),
                        "DELAY": delay_avg,
                        "LOS": los
                    }

                result_dict["TOTAL_VEHS"] = all_vehs_total
                result_dict["TOTAL_DELAY"] = round(all_delay_sum / all_delay_count, 1) if all_delay_count > 0 else 0.0
                result_dict["TOTAL_LOS"] = get_los(result_dict["TOTAL_DELAY"])
                result_dict["hourly"] = hourly_summary

                for timeint, df_time in df_node_alltime.groupby('TIMEINT'):
                    timeint_str = str(timeint).zfill(2)
                    timeint_result = {}

                    for appr_id, df_appr_name in df_time.groupby('APPR_ID'):
                        match = df_appr_meta[
                            (df_appr_meta['NODE_ID'] == node_id) & 
                            (df_appr_meta['APPR_ID'] == appr_id)
                        ]
                        appr_name = match.iloc[0]['APPR_NAME'] if not match.empty else "미지정"

                        if appr_name not in timeint_result:
                            timeint_result[appr_name] = {}

                        for direction, df_dir in df_appr_name.groupby('DIRECTION'):
                            vehs = int(df_dir['VEHS'].sum(skipna=True) or 0)
                            delay_vals = df_dir['DELAY'].dropna().astype(float).tolist()
                            delay_avg = round(sum(delay_vals) / len(delay_vals), 1) if delay_vals else 0.0
                            los = get_los(delay_avg)

                            timeint_result[appr_name][str(int(direction))] = {
                                "VEHS": vehs,
                                "DELAY": delay_avg,
                                "LOS": los
                            }

                    result_dict[timeint_str] = timeint_result

                grouped_result[label][node_name] = result_dict
            
            if label not in grouped_result or not grouped_result[label]:
                return jsonify({
                    "status": "fail",
                    "message": f"{label}에 해당하는 데이터가 없습니다.",
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }), 404

        return app.response_class(
            response=json.dumps({
                "status": "success",
                "label": label,
                "latest_date": latest_date,
                "data": grouped_result[label],
            }, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )

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