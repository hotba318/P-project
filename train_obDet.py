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

# 프로젝트 루트 경로 설정
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

# 폴더 생성
for key, path in DIRS.items():
    os.makedirs(path, exist_ok=True)

# 로거 설정
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
    Roboflow에서 데이터셋을 다운로드하고, 경로를 수정한 yaml 파일을 생성합니다.
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
        if not os.path.exists(origin_yaml_path):
             # 가끔 폴더 구조가 다를 경우를 대비한 체크
             logger.error(f"data.yaml 파일을 찾을 수 없습니다: {origin_yaml_path}")
             raise FileNotFoundError("data.yaml not found")

        with open(origin_yaml_path, 'r') as f:
            data_conf = yaml.safe_load(f)

        # 경로 수정 (절대 경로)
        data_conf['path'] = dataset.location
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
    logger.info(f"학습 프로세스 시작: {project_id}")
    logger.info(f"{'='*30}")

    # 1. GPU 확인
    # 코드에서는 '0'으로 설정하지만, 실제 물리적 GPU는 터미널 실행 시 CUDA_VISIBLE_DEVICES로 제어 권장
    device = '0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"학습 장치 설정: {device} (torch version: {torch.__version__})")

    # 2. 데이터셋 준비
    data_yaml_path = setup_dataset_and_config(project_id, version_num)

    # 3. 모델 로딩
    model_path = os.path.join(ROOT_DIR, 'yolov8s.pt') # 로컬에 있으면 그거 사용
    if os.path.exists(model_path):
        logger.info(f"로컬 모델 파일 로드: {model_path}")
        model = YOLO(model_path)
    else:
        logger.info("모델 다운로드 및 로드 (yolov8s.pt)...")
        model = YOLO('yolov8s.pt') 

    # 4. 학습 시작
    logger.info("학습 시작...")
    
    # model.train()은 학습 결과를 객체로 반환하지 않음 (Ultralytics 버전에 따라 다름),
    # 하지만 model 객체 내부 상태가 업데이트됨.
    model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        project=os.path.join(DIRS["runs"], "detect"),
        name=run_name,
        exist_ok=True,    # 주의: True면 exp1 폴더에 덮어씁니다. 기록 보존이 필요하면 False로 하세요.
        device=device,
        workers=4,        # [수정] 학교 서버 공유메모리 부족 방지 (8 -> 4)
        batch=16          # [참고] 메모리 부족 시 8로 줄이세요
    )
    
    # 5. 결과 파일 찾기 및 이동 (동적 경로 처리)
    # model.trainer.save_dir : 실제 학습 결과가 저장된 경로 (예: runs/detect/exp2)
    save_dir = str(model.trainer.save_dir)
    logger.info(f"실제 저장된 경로: {save_dir}")

    src_best_model = os.path.join(save_dir, "weights", "best.pt")
    dst_best_model = os.path.join(DIRS["models"], "best.pt")

    if os.path.exists(src_best_model):
        shutil.copy(src_best_model, dst_best_model)
        logger.info(f"Best 모델 복사 완료: {dst_best_model}")
    else:
        logger.warning(f"모델 파일을 찾을 수 없습니다: {src_best_model}")

    # 6. 성능 평가 로그
    logger.info("\n모델 성능 평가 중...")
    try:
        metrics = model.val(data=data_yaml_path)
        performance_msg = f"""
        [학습 완료] 결과 요약
        -----------------------------------
        저장 경로    : {save_dir}
        mAP@0.5-0.95 : {metrics.box.map:.4f}
        mAP@0.5      : {metrics.box.map50:.4f}
        """
        logger.info(performance_msg)
        print(performance_msg)
    except Exception as e:
        logger.error(f"성능 평가 중 오류 발생: {e}")

# ==============================================================
# [메인 실행부]
# ==============================================================
if __name__ == "__main__":
    try:
        # 프로젝트 ID와 버전은 본인 상황에 맞게 수정
        train_process("test-2k1ir", 1, "exp1")
    except Exception as e:
        logger.error(f"Critical Error: {e}")
