import os
import subprocess

# ==================================== [ 딥러닝 학습 시작 ] ====================================

def run_deep_learning_pipeline():
    logs = []

    VENV_PYTHON = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    
    base_dir = os.path.join(os.getcwd(), "gndl")
    gnn_data_dir = os.path.join(base_dir, "gnn_data")

    pkl_files = ["node_features.pkl", "edge_list.pkl", "node_index.pkl"]
    pkl_paths = [os.path.join(gnn_data_dir, f) for f in pkl_files]

    # STEP 1: 전처리 필요 여부 확인
    if not all(os.path.exists(p) for p in pkl_paths):
        logs.append("🟡 전처리 데이터 없음 → 1.preprocess.py 실행")
        result = subprocess.run([VENV_PYTHON, "gndl/1.preprocess.py"], capture_output=True, text=True)
        logs.append(result.stdout or result.stderr)
    else:
        logs.append("✅ 전처리된 데이터가 존재하여 건너뜀")

    # STEP 2: 학습 모델 파일 체크
    model_path = os.path.join(gnn_data_dir, "best_model.pt")
    if not os.path.exists(model_path):
        logs.append("🟡 모델 없음 → 2.gnn.py 실행")
        result = subprocess.run([VENV_PYTHON, "gndl/2.gnn.py"], capture_output=True, text=True)
        logs.append(result.stdout or result.stderr)
    else:
        logs.append("✅ 학습된 모델이 존재하여 건너뜀")

    # STEP 3: 시각화 결과 확인
    visual_path = os.path.join(gnn_data_dir, "visual_result.png")
    if not os.path.exists(visual_path):
        logs.append("🟡 시각화 없음 → 3.gnn_visual.py 실행")
        result = subprocess.run([VENV_PYTHON, "gndl/3.gnn_visual.py"], capture_output=True, text=True)
        logs.append(result.stdout or result.stderr)
    else:
        logs.append("✅ 시각화 결과가 존재하여 건너뜀")

    # STEP 4: 미래 예측 결과 확인
    pred_path = os.path.join(gnn_data_dir, "future_prediction.json")
    if not os.path.exists(pred_path):
        logs.append("🟡 미래 예측 없음 → 4.future_prediction.py 실행")
        result = subprocess.run([VENV_PYTHON, "gndl/4.future_prediction.py"], capture_output=True, text=True)
        logs.append(result.stdout or result.stderr)
    else:
        logs.append("✅ 미래 예측 결과가 존재하여 건너뜀")

    # 최종 로그 출력
    for log in logs:
        print(log)

    return logs

if __name__ == "__main__":
    logs = run_deep_learning_pipeline()
    print("전체 파이프라인 실행 완료.")
