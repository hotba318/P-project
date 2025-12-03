import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# 모델 경로
MODEL_FILENAME = "best.pt"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# 모델 로드
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        model = YOLO(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

# YOLO 예측 결과를 분석, JSON 데이터 생성
def process_results(results, filename):
    if not results:
        return {"message": "No objects detected", "total_count": 0}

    result = results[0]
    
    # 탐지된 정보 추출
    if result.boxes:
        detected_cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names_dict = result.names
        detected_names = [class_names_dict[cls_id] for cls_id in detected_cls_ids]
        object_counts = dict(Counter(detected_names))
        total_count = len(detected_names)
    else:
        object_counts = {}
        total_count = 0
    
    # JSON 구조 생성
    return {
        "filename": filename,
        "total_count": total_count,
        "detections": object_counts
    }

# 이미지 파일 경로를 받아 객체 탐지 후 JSON 반환
def detect_objects_from_bytes(image_bytes, filename_hint="uploaded_image.jpg"):
    model, error = load_model()
    if error: return json.dumps({"error": error}, ensure_ascii=False)

    # 1. 바이트 데이터를 OpenCV 이미지 포맷으로 변환 (Decoding)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_data is None:
            return json.dumps({"error": "Failed to decode image bytes"}, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({"error": f"Image processing error: {str(e)}"}, ensure_ascii=False)

    # 2. 예측 실행 (이미지 객체 전달)
    results = model.predict(source=img_data, conf=0.5, save=False, show=False)
    
    # 3. 결과 처리 및 반환
    output_data = process_results(results, filename_hint)
    return json.dumps(output_data, indent=4, ensure_ascii=False)