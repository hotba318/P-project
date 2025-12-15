import os
import shutil
import yaml
import torch
import logging
from datetime import datetime
from ultralytics import YOLO
from roboflow import Roboflow

# ==============================================================
# [설정] 1. 전역 설정 및 경로 변수 (Global Variables)
# ==============================================================
ROBOFLOW_API_KEY = "7QLEWkuCXDEO1IV0ccst"

# 프로젝트 루트 및 현재 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)  # 예: ~/yolo/

# 폴더 경로 정의 (DIRS)
DIRS = {
    "data": os.path.join(ROOT_DIR, "data"),
    "configs": os.path.join(ROOT_DIR, "configs"),
    "runs": os.path.join(ROOT_DIR, "runs"),
    "models": os.path.join(ROOT_DIR, "models"),
    "logs": os.path.join(ROOT_DIR, "logs"),
}

# 폴더 자동 생성
for key, path in DIRS.items():
    os.makedirs(path, exist_ok=True)

# 로거(Logger) 설정
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


# ==============================================================
# [함수] 2. 데이터셋 및 설정 함수 (setup_dataset_and_config)
# ==============================================================
def setup_dataset_and_config(project_id, version_num):
    """
    Roboflow에서 데이터셋을 다운로드하고, 경로를 수정한 yaml 파일을 생성합니다.
    설계서 변수: rf, project, version, dataset, origin_yaml_path, new_yaml_path, data_conf
    """
    try:
        # Roboflow 객체 초기화
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("hotba318").project(project_id)
        version = project.version(version_num)
        
        # 데이터셋 다운로드 (data 폴더)
        dataset = version.download("yolov8", location=DIRS["data"])
        logger.info(f"데이터셋 다운로드 완료: {dataset.location}")

        # 원본 yaml 파일 경로 (origin_yaml_path)
        origin_yaml_path = os.path.join(dataset.location, "data.yaml")
        
        if not os.path.exists(origin_yaml_path):
             logger.error(f"data.yaml 파일을 찾을 수 없습니다: {origin_yaml_path}")
             raise FileNotFoundError("data.yaml not found")

        # yaml 파일 로드 (data_conf)
        with open(origin_yaml_path, 'r') as f:
            data_conf = yaml.safe_load(f)

        # 경로 수정 (절대 경로로 변경)
        data_conf['path'] = dataset.location
        data_conf['train'] = "train/images"
        data_conf['val'] = "val/images"
        data_conf['test'] = "test/images"

        # 새로운 yaml 저장 (new_yaml_path)
        new_yaml_path = os.path.join(DIRS["configs"], "data.yaml")
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data_conf, f)
        
        logger.info(f"설정 파일 생성 완료: {new_yaml_path}")
        return new_yaml_path

    except Exception as e:
        logger.error(f"데이터셋 설정 실패: {e}")
        raise e


# ==============================================================
# [함수] 3. 학습 프로세스 함수 (train_process)
# ==============================================================
def train_process(project_id, version_num, run_name):
    """
    모델 학습 전체 프로세스를 수행합니다.
    설계서 변수: device, data_yaml_path, model, epochs, imgsz, batch, workers, save_dir 등
    """
    logger.info(f"{'='*30}")
    logger.info(f"학습 프로세스 시작: {project_id} (Version: {version_num})")
    logger.info(f"{'='*30}")

    # 1. 학습 장치 설정 (device)
    device = '0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"학습 장치 설정: {device} (torch version: {torch.__version__})")

    # 2. 데이터셋 준비 (data_yaml_path)
    data_yaml_path = setup_dataset_and_config(project_id, version_num)

    # 3. 모델 로딩 (model, model_path)
    model_path = os.path.join(ROOT_DIR, 'yolov8s.pt')
    
    if os.path.exists(model_path):
        logger.info(f"로컬 모델 파일 로드: {model_path}")
        model = YOLO(model_path)
    else:
        logger.info("모델 다운로드 및 로드 (yolov8s.pt)...")
        model = YOLO('yolov8s.pt') 

    # 4. 학습 파라미터 설정 (설계서 명칭 표준안 반영)
    epochs = 30
    imgsz = 640
    batch = 16
    workers = 4  # 학교 서버 메모리 이슈 방지용

    # 5. 학습 시작
    logger.info(f"학습 시작 (Epochs: {epochs}, Batch: {batch})...")
    
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project=os.path.join(DIRS["runs"], "detect"),
        name=run_name,
        exist_ok=True,    # 덮어쓰기 허용 (실험 관리에 따라 False 권장)
        device=device,
        workers=workers,
        batch=batch
    )
    
    # 6. 결과 파일 처리 (save_dir, src_best_model, dst_best_model)
    # model.trainer.save_dir는 PosixPath 객체일 수 있으므로 문자열 변환
    save_dir = str(model.trainer.save_dir)
    logger.info(f"실제 저장된 경로: {save_dir}")

    src_best_model = os.path.join(save_dir, "weights", "best.pt")
    dst_best_model = os.path.join(DIRS["models"], "best.pt")

    if os.path.exists(src_best_model):
        shutil.copy(src_best_model, dst_best_model)
        logger.info(f"Best 모델 복사 완료: {dst_best_model}")
    else:
        logger.warning(f"모델 파일을 찾을 수 없습니다: {src_best_model}")

    # 7. 성능 평가 및 로그 (metrics)
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
        # 설계서 기준 파라미터 입력
        train_process("test-2k1ir", 1, "exp1")
    except Exception as e:
        logger.error(f"Critical Error: {e}")
