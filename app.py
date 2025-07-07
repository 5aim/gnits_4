from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import datetime, pyodbc, os, subprocess, json
from windows import set_dpi_awareness


set_dpi_awareness()
app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# ========================================================= [ 현재시간 ]

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ========================================================= [ VISUM 자동화 코드 실행 ]

def run_first_script():
    script_path = os.path.join(os.path.dirname(__file__), 'auto_simulation', 'auto_visum.py')
    print(f"[{get_current_time()}] 실행: first.py")
    subprocess.Popen(['python', script_path])
    
# ========================================================= [ VISSIM 자동화 코드 실행 ]

def run_second_script():
    script_path = os.path.join(os.path.dirname(__file__), 'auto_simulation', 'auto_vissim.py')
    print(f"[{get_current_time()}] 실행: second.py")
    subprocess.Popen(['python', script_path])

# ========================================================= [ 자동화 시뮬레이션 스케쥴러 설정 ]

# nohup python app.py > server.log 2>&1 &
scheduler = BackgroundScheduler()
scheduler.add_job(run_first_script, 'cron', hour=2, minute=0)
scheduler.add_job(run_second_script, 'cron', hour=4, minute=0)
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

# ========================================================= [ SC-TWIN-0001 ]

@app.route('/sctwin0001', methods=['GET', 'POST'])
def sctwin0001():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            response_data = {
                'route': '/sctwin0001',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data': [
                    {
                        'fnr-twin-0005': '디지털 트윈 기반 AI 신호 분석 및 관리 시스템 기능 - 교통데이터 (교통 및 신호) 분석 가공 기능 교통분석: 전일/시간대별 교통',
                    },
                    {
                        'fnr-twin-0006': '오전 첨두/비첨두 시간 교통 표출'
                    },
                    {
                        'fnr-twin-0007': '오후 첨두/비첨두 시간 교통 표출'
                    }
                ]
            }
            
            return Response(
                json.dumps(response_data, ensure_ascii=False),
                content_type='application/json; charset=utf-8'
            )

        except Exception as e:
            return jsonify({
                'route': '/sctwin0001',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0001] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0001',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0001',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0002 ]

@app.route('/sctwin0002', methods=['GET', 'POST'])
def sctwin0002():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0002',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0005': '디지털 트윈 기반 AI 신호 분석 및 관리 시스템 기능 - 교통데이터 (교통 및 신호) 분석 가공 기능 교통분석: 전일/시간대별 교통',
                    },
                    {
                        'fnr-twin-0008': '도로 구간별 진입/진출 교통 분석'
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0002',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0002] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0002',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0002',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0010 ~ 11 ]

@app.route('/sctwin0010_11', methods=['GET', 'POST'])
def sctwin0010_11():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0010_11',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0016': '도로망 전체 평균 지체시간/평균통행속도',
                    },
                    {
                        'fnr-twin-0017': '도로망 전체 속도/밀도'
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0010_11',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0010_11] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0010_11',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0010_11',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0016 ~ 18 ]

@app.route('/sctwin0016_18', methods=['GET', 'POST'])
def sctwin0016_18():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0016_18',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0026': '도로구간 V/C',
                    },
                    {
                        'fnr-twin-0027': '주요 교통축 V/C, 통행속도/심각도'
                    },
                    {
                        'fnr-twin-0028': '주요 교차로 지체시간/서비스수준'
                    },
                    {
                        'fnr-twin-0028': '구간별 통행속도/구간별 통행시간(?)'
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0016_18',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0016_18] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0016_18',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0016_18',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0027_44_45 ]

@app.route('/sctwin0027_44_45', methods=['GET', 'POST'])
def sctwin0027_44_45():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0027_44_45',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0042': '도로구간 통행시간/통행속도/통행비용',
                    },
                    {
                        'fnr-twin-0106': '시간단위 교차로 교통량 집계: 교통량/지체시간/서비스수준',
                    },
                    {
                        'fnr-twin-0107': '일단위 교차로 교통량 집계: 교통량/지체시간/서비스수준',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0027_44_45',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0027_44_45] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0027_44_45',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0027_44_45',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0028_30_40_41_42_43 ]

@app.route('/sctwin0028_30_40_41_42_43', methods=['GET', 'POST'])
def sctwin0028_30_40_41_42_43():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0028_30_40_41_42_43',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0043': '교차로 방향별(직진, 좌회전, 우회전) 교통흐름',
                    },
                    {
                        'fnr-twin-0077': '기초데이터 조회/관리 기능 구간 설정 및 파라미터 관리: 방향별 구간 정보',
                    },
                    {
                        'fnr-twin-0078': '구간설정 및 파라미터 관리: 링크 정보 조회',
                    },
                    {
                        'fnr-twin-0099': '교차로 15분/1시간/일별 교통량 통계 정보 조회',
                    },
                    {
                        'fnr-twin-0101': '통계조회: 직진/우회전/좌회전 15분 평균 지체시간 통계 정보',
                    },
                    {
                        'fnr-twin-0103': '통계조회: 접근로별 15분 평균 지체시간 통계 정보',
                    },
                    {
                        'fnr-twin-0105': '시각화 교통데이터 집계: 평균 접근로 평균 교통량/지체시간/서비스수준 집계 정보',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0028_30_40_41_42_43',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0028_30_40_41_42_43] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0028_30_40_41_42_43',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0028_30_40_41_42_43',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0031_34 ]

@app.route('/sctwin0031_34', methods=['GET', 'POST'])
def sctwin0031_34():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0031_34',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0082': '지점환경 관리： 도로조건 조회 (차로수)',
                    },
                    {
                        'fnr-twin-0084': '지점환경 관리: 교통조건 조회 (교차로 유형)',
                    },
                    {
                        'fnr-twin-0086': '지점환경 관리: 신호운영 조회 (주기)',
                    },
                    {
                        'fnr-twin-0088': '지점환경 관리: 교차로정보 조회 (교차로 중요도)',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0031_34',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0031_34] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0031_34',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0031_34',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0035_38 ]

@app.route('/sctwin0035_38', methods=['GET', 'POST'])
def sctwin0035_38():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0035_38',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0090': '메인지도: API 지도 조회',
                    },
                    {
                        'fnr-twin-0091': '메인지도: 지점 정보 조회',
                    },
                    {
                        'fnr-twin-0094': '메인지도: 정보생성 위치 및 운영정보',
                    },
                    {
                        'fnr-twin-0095': '메인지도: 정보생성지점 실시간 서비스 수준',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0035_38',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0035_38] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0035_38',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0035_38',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0039_46_47 ]

@app.route('/sctwin0039_46_47', methods=['GET', 'POST'])
def sctwin0039_46_47():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0039_46_47',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0097': '15분/1시간/일별 4개 지구 교통량 조회',
                    },
                    {
                        'fnr-twin-0109': '표준 노드/링크 일/월/년 평균 속도 생성',
                    },
                    {
                        'fnr-twin-0111': '도로 축별 시간/일/월/년 평균 속도 생성',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0039_46_47',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0039_46_47] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0039_46_47',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0039_46_47',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0048 ]

@app.route('/sctwin0048', methods=['GET', 'POST'])
def sctwin0048():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0048',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0112': '관리자 로그인 페이지 렌딩',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0048',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0048] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0048',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0048',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0051 ]

@app.route('/sctwin0051', methods=['GET', 'POST'])
def sctwin0051():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0051',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0123': 'visum 기반 기종점 이동 통행량 분석정보',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0051',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0051] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0051',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0051',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0052 ]

@app.route('/sctwin0052', methods=['GET', 'POST'])
def sctwin0052():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0052',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0124': 'visum 기반 시간대별 통행량 및 도로용량 정보',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0052',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0052] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0052',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0052',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0053 ]

@app.route('/sctwin0053', methods=['GET', 'POST'])
def sctwin0053():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0053',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0125': 'visum 기반 교통존 단위 통행정보',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0053',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0053] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0053',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0053',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0054 ]

@app.route('/sctwin0054', methods=['GET', 'POST'])
def sctwin0054():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0054',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0126': 'vissim 기반 시간간격 단위 지점(?) 통과교통량/통행속도/통행시간 정보. 그림엔 vms 트윈사진임',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0054',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0054] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0054',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0054',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0055 ]

@app.route('/sctwin0055', methods=['GET', 'POST'])
def sctwin0055():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0055',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0127': 'vissim 기반 시간간격 단위 교차로 진출입교통량/지체시간/대기행렬/정지횟수/연료소모량 정보',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0055',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0055] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0055',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0055',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0056 ]

@app.route('/sctwin0056', methods=['GET', 'POST'])
def sctwin0056():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0056',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0128': 'vissim 기반 분석지역 단위 상황별 평균 교통량 통계',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0056',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0056] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0056',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0056',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0057 ]

@app.route('/sctwin0057', methods=['GET', 'POST'])
def sctwin0057():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0057',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0135': '시간대별 혼잡 구간 및 교차로 분석',
                    },
                    {
                        '혼잡 구간 리스트': [
                                '구간명',
                                '지역',
                                '혼잡도 판단(km/h)',
                                '혼잡 지속성',
                                '교통상황 판단'
                            ],
                    },
                    {
                        '혼잡 교차로 리스트': [
                                '교차로명',
                                '지역',
                                '혼잡도 판단(지체시간)',
                                '혼잡 지속성',
                                '교통상황 판단'
                            ],
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0057',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0057] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0057',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0057',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0058 ]

@app.route('/sctwin0058', methods=['GET', 'POST'])
def sctwin0058():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0058',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0140': '신호 최적화 정보',
                    },
                    {
                        '혼잡 교차로 신호최적화 효과 검증': [
                                '기준일자',
                                '기준요일',
                                '교차로명',
                                '행정구역',
                                '교통축',
                                '교차로유형',
                                '지체시간',
                                '서비스수준',
                                '혼잡도'
                            ],
                    },
                    {
                        '현황': [
                                '지체시간',
                                '통행속도',
                            ],
                        '개선': [
                                '지체시간',
                                '통행속도',
                            ],
                    },
                    '교통축 GreenBand 비교 이미지',
                    '교차로 녹색시간 비교 이미지'
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0058',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0058] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0058',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0058',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ SC-TWIN-0059 ]

@app.route('/sctwin0059', methods=['GET', 'POST'])
def sctwin0059():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0059',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        'fnr-twin-0142': '교차로 통행 상황 가늠 목적. 시내 전체 혹은 주요 축별 교차로 소통상황(링크)',
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0059',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/sctwin0059] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/sctwin0059',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/sctwin0059',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ 딥러닝 모델 ]

@app.route('/analy_traffic_vol', methods=['GET', 'POST'])
def analy_traffic_vol():
    if request.method == 'GET':
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # 접속 성공 확인용 메시지
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/analy_traffic_vol',
                'method': 'GET',
                'status': 'DB 연결 성공',
                'result': result[0],
                'time': get_current_time(),
                'data':[
                    {
                        '딥러닝': [
                                '학습횟수',
                                '정확도',
                                '오차',
                                '가공데이터',
                                '추정교통량',
                                '예측데이터 정확성 비교(가로교통량)',
                                '예측데이터 정확성 비교(교차로 총 진입교통량) > 방향별로 해야 할 듯'
                            ],
                    },
                    {
                        '추가 필요': [
                            '관광데이터',
                            '날씨데이터',
                            '단기 예측',
                            '장기 예측(불가능 예상)',
                            '예측 정보의 신뢰성 검증을 위한 디지털 트윈 시뮬레이션 프로그램 구동(?)'
                        ]
                    }
                ]
            })

        except Exception as e:
            return jsonify({
                'route': '/analy_traffic_vol',
                'method': 'GET',
                'status': 'DB 연결 실패',
                'error': str(e),
                'time': get_current_time()
            }), 500

    if request.method == 'POST':
        try:
            data = request.get_json()

            conn = get_connection()
            cursor = conn.cursor()

            print("[/analy_traffic_vol] 받은 데이터:", data)

            cursor.close()
            conn.close()

            return jsonify({
                'route': '/analy_traffic_vol',
                'method': 'POST',
                'message': 'POST 요청 처리 성공',
                'received_data': data
            })

        except Exception as e:
            return jsonify({
                'route': '/analy_traffic_vol',
                'method': 'POST',
                'message': 'DB 연결 또는 처리 중 오류 발생',
                'error': str(e)
            }), 500

# ========================================================= [ 서버실행 ]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5101, debug=False)