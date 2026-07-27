import cv2
import time
import numpy as np
import torch
import tensorrt as trt
import serial # 추가된 부분

# --- [통신 세팅] 피코 보드 연결 ---
# (젯슨에 USB로 연결된 기기는 보통 ttyACM0이나 ttyUSB0로 잡힙니다)
try:
    pico = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    print("✅ 피코 보드 연결 성공!")
except:
    print("❌ 피코 보드를 찾을 수 없습니다. (포트 번호를 확인하세요)")
    pico = None

# ... (중간 텐서RT 세팅 및 카메라 초기화 코드는 아까와 동일) ...

while True:
    # ... (카메라 읽고 텐서RT로 추론하는 과정 동일) ...
    
    depth = d_output.cpu().numpy()[0]
    depth_resized = cv2.resize(depth, (640, 480))
    
    # -----------------------------------------------------
    # [데이터 추출 및 전송] 화면 정중앙의 깊이 값 뽑아내기
    # 화면(640x480)의 정중앙 좌표인 (y=240, x=320)의 값을 가져옵니다.
    center_depth_value = depth_resized[240, 320]
    
    # 디버깅용: 정규화된 값(0~255)으로 변환해서 확인
    norm_val = int(np.clip((center_depth_value / np.max(depth_resized)) * 255, 0, 255))
    
    # 피코로 데이터 전송 (예: "D:150\n" 형태로 전송)
    if pico is not None:
        message = f"D:{norm_val}\n"
        pico.write(message.encode('utf-8'))
    # -----------------------------------------------------
    
    # 보기 좋게 열화상 색상 입히기
    depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # 화면 중앙에 십자선과 깊이 값 표시하기
    cv2.drawMarker(depth_color, (320, 240), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(depth_color, f"Depth: {norm_val}", (330, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Depth to Pico', depth_color)
    
    if cv2.waitKey(1) == ord('q'): break

if pico is not None:
    pico.close()
cap.release()
cv2.destroyAllWindows()
