import os
import shutil
import yaml
import torch
import logging
from datetime import datetime
from ultralytics import YOLO
from roboflow import Roboflow

# ==============================================================
# [설정] 기본 경로 및 API 키
# ==============================================================
ROBOFLOW_API_KEY = "7QLEWkuCXDEO1IV0ccst"

# 현재 스크립트 위치(scripts/)의 상위 폴더를 프로젝트 루트(~/yolo/)로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)  # ~/yolo/

# 폴더 경로 정의
DIRS = {
    "data": os.path.join(ROOT_DIR, "data"),
    "configs": os.path.join(ROOT_DIR, "configs"),
    "runs": os.path.join(ROOT_DIR, "runs"),
    "models": os.path.join(ROOT_DIR, "models"),
    "logs": os.path.join(ROOT_DIR, "logs"),
}

# 폴더 생성 함수
for key, path in DIRS.items():
    os.makedirs(path, exist_ok=True)

# 로거 설정 (logs/train.log에 기록)
log_file_path = os.path.join(DIRS["logs"], "train.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_dataset_and_config(project_id, version_num):
    """
    데이터셋을 다운로드하고 configs/data.yaml을 생성하여 경로를 연결합니다.
    """
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("hotba318").project(project_id)
        version = project.version(version_num)
        
        # data 폴더에 다운로드
        dataset = version.download("yolov8", location=DIRS["data"])
        logger.info(f"데이터셋 다운로드 완료: {dataset.location}")

        # 원본 yaml 읽기
        origin_yaml_path = os.path.join(dataset.location, "data.yaml")
        with open(origin_yaml_path, 'r') as f:
            data_conf = yaml.safe_load(f)

        # 경로를 절대 경로로 수정 (configs 폴더로 yaml을 옮기기 때문)
        # Roboflow 데이터셋 구조에 맞춰 경로 지정
        data_conf['path'] = dataset.location # 데이터셋 루트
        data_conf['train'] = "train/images"
        data_conf['val'] = "val/images"
        data_conf['test'] = "test/images"

        # configs/data.yaml로 저장
        new_yaml_path = os.path.join(DIRS["configs"], "data.yaml")
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data_conf, f)
        
        logger.info(f"설정 파일 생성 완료: {new_yaml_path}")
        return new_yaml_path

    except Exception as e:
        logger.error(f"데이터셋 설정 실패: {e}")
        raise e

def train_process(project_id, version_num, run_name):
    logger.info(f"{'='*30}")
    logger.info(f"학습 프로세스 시작: {project_id} (Small Model)")
    logger.info(f"{'='*30}")

    # 0. GPU 확인
    device = '0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"학습 장치: {device}")

    # 1. 데이터셋 및 Config 준비
    data_yaml_path = setup_dataset_and_config(project_id, version_num)

    # 2. 모델 로딩 (Small 모델)
    logger.info("모델 로딩 중 (yolov8s.pt - Small)...")
    model = YOLO('yolov8s.pt') 

    # 3. 학습 시작
    # runs/detect/{run_name} 에 저장됨
    logger.info("학습 시작...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        project=os.path.join(DIRS["runs"], "detect"), # ~/yolo/runs/detect
        name=run_name,
        exist_ok=True,
        device=device,
        workers=8,        # 서버 환경 권장
        batch=16          # Small 모델은 메모리를 더 쓰므로 OOM 발생 시 8로 줄이세요
    )
    
    # 4. 결과 정리 및 이동
    src_best_model = os.path.join(DIRS["runs"], "detect", run_name, "weights", "best.pt")
    dst_best_model = os.path.join(DIRS["models"], "best.pt")

    if os.path.exists(src_best_model):
        shutil.copy(src_best_model, dst_best_model)
        logger.info(f"Best 모델 복사 완료: {dst_best_model}")
    else:
        logger.warning(f"모델 파일을 찾을 수 없음: {src_best_model}")

    # 5. 성능 평가 로그 기록
    logger.info("\n모델 성능 평가 중...")
    metrics = model.val(data=data_yaml_path)
    
    performance_msg = f"""
    [{run_name} - Small Model] 결과 요약
    -----------------------------------
    mAP@0.5-0.95 : {metrics.box.map:.4f}
    mAP@0.5      : {metrics.box.map50:.4f}
    """
    logger.info(performance_msg)
    print(performance_msg) # 콘솔에도 출력

# ==============================================================
# [메인 실행부]
# ==============================================================
if __name__ == "__main__":
    # 실행 작업
    try:
        train_process("test-2k1ir", 1, "exp1")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")