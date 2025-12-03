import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# 모델 경로 설정
MODEL_FILENAME = "best.pt"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# 모델 로드 (전역 변수로 한 번만 로드하는 것을 권장하나, 기존 구조 유지)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        model = YOLO(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

# YOLO 예측 결과를 API 명세에 맞춰 가공
def process_results(results):
    detections_list = []

    # 결과가 없거나 박스가 없는 경우 처리
    if not results or not results[0].boxes:
        return []

    result = results[0]
    
    # 탐지된 객체 반복 처리
    for box in result.boxes:
        # 1. Class ID [cite: 15]
        cls_id = int(box.cls[0].item())
        
        # 2. Label (이름) [cite: 16]
        label_name = result.names[cls_id]
        
        # 3. BBox 좌표 (x1, y1, x2, y2) 
        # 소수점 좌표를 정수형으로 변환 (필요 시 float 유지 가능)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        
        # 4. Confidence (신뢰도) [cite: 18]
        conf = float(box.conf[0].item())

        # 개별 객체 정보 딕셔너리 생성
        detection_info = {
            "classId": cls_id,
            "label": label_name,
            "bbox": bbox,
            "confidence": round(conf, 2)  # 소수점 2자리 반올림
        }
        detections_list.append(detection_info)

    return detections_list

# 이미지 바이트를 받아 객체 탐지 후 지정된 JSON 반환
def detect_objects_from_bytes(image_bytes):
    # 1. 모델 로드
    model, error = load_model()
    if error:
        # 모델 로드 실패 -> 서버 내부 오류로 처리 
        response = {
            "success": False,
            "message": "이미지 분석 중 오류가 발생했습니다. (Model Error)",
            "statusCode": 500,
            "detections": None
        }
        return json.dumps(response, ensure_ascii=False)

    try:
        # 2. 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_data is None:
            raise ValueError("Failed to decode image bytes")

        # 3. 예측 실행
        results = model.predict(source=img_data, conf=0.5, save=False, show=False)
        
        # 4. 결과 가공
        detections = process_results(results)

        # 5. 최종 응답 생성 (API 명세 준수)
        if len(detections) > 0:
            # 성공 케이스 [cite: 8]
            response = {
                "success": True,
                "message": "식재료가 검출되었습니다.",
                "statusCode": 200,
                "detections": detections
            }
        else:
            # 실패 케이스 - 객체 미검출 [cite: 28]
            response = {
                "success": False,
                "message": "이미지에서 식재료를 찾을 수 없습니다.",
                "statusCode": 400,
                "detections": []
            }

    except Exception as e:
        # 실패 케이스 - 서버 내부 오류 
        # 로그 출력 등 추가 조치 가능
        print(f"Error: {e}") 
        response = {
            "success": False,
            "message": "이미지 분석 중 오류가 발생했습니다.",
            "statusCode": 500,  # 문서에는 00으로 되어있으나 HTTP 표준인 500 사용 권장
            "detections": None
        }

    return json.dumps(response, indent=4, ensure_ascii=False)