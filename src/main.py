import cv2
import time
import numpy as np
import torch
import tensorrt as trt
from ultralytics import YOLO
import serial
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DepthEstimatorTRT:
    def __init__(self, engine_path: str, input_shape: tuple = (518, 518)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        
        self.d_input = torch.empty((1, 3, *self.input_shape), dtype=torch.float32, device='cuda')
        self.d_output = torch.empty((1, *self.input_shape), dtype=torch.float32, device='cuda')
        
        self.context.set_tensor_address("input", int(self.d_input.data_ptr()))
        self.context.set_tensor_address("output", int(self.d_output.data_ptr()))
        
        self.stream = torch.cuda.Stream()
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def infer(self, img_518: np.ndarray) -> tuple:
        img = cv2.cvtColor(img_518, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        
        with torch.cuda.stream(self.stream):
            self.d_input.copy_(torch.from_numpy(np.expand_dims(img, 0)), non_blocking=True)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        depth_raw = self.d_output.cpu().numpy()[0]
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        return depth_color, depth_norm


class HardwareCommunicator:
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200, timeout: int = 1):
        self.port = port
        self.conn = None
        try:
            self.conn = serial.Serial(port, baudrate, timeout=timeout)
            logging.info(f"시리얼 통신 연결 성공: {port}")
        except serial.SerialException:
            logging.warning(f"하드웨어를 찾을 수 없습니다 ({port}). 오프라인 모드로 동작합니다.")

    def transmit(self, data: str):
        if self.conn and self.conn.is_open:
            self.conn.write(data.encode('utf-8'))

    def close(self):
        if self.conn and self.conn.is_open:
            self.conn.close()


def main():
    YOLO_ENGINE_PATH = "checkpoints/yolo26n.engine"
    DEPTH_ENGINE_PATH = "checkpoints/depth_anything_v2_vits.engine"
    PROC_SIZE = (518, 518)

    logging.info("3분할 파이프라인 초기화 중...")
    yolo_detector = YOLO(YOLO_ENGINE_PATH, task='detect')
    depth_estimator = DepthEstimatorTRT(DEPTH_ENGINE_PATH, input_shape=PROC_SIZE)
    hardware = HardwareCommunicator()

    # 웹캠 최적화 설정 (V4L2 + MJPEG, 720p)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow('Sensory Substitution: Triple View', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: 
                break

            # 1. 왼쪽 화면용: 원본 프레임을 높이 518에 맞춰 비율 유지 리사이즈
            h_orig, w_orig = frame.shape[:2]
            target_h = PROC_SIZE[1]
            target_w = int(w_orig * (target_h / h_orig))
            left_original = cv2.resize(frame, (target_w, target_h))

            # 2. 연산용 518x518 축소 이미지
            frame_518 = cv2.resize(frame, PROC_SIZE, interpolation=cv2.INTER_NEAREST)

            # 3. 중앙 화면용: YOLO 추론 (COCO 라벨 및 박스 오버레이)
            results = yolo_detector(frame_518, verbose=False, conf=0.4, device=0)[0]
            center_yolo = results.plot()  # 518x518

            # 4. 오른쪽 화면용: Depth 추론
            right_depth, depth_norm = depth_estimator.infer(frame_518)

            # 하드웨어 신호 전송 (정중앙 픽셀 기준 깊이값)
            center_col, center_row = PROC_SIZE[0] // 2, PROC_SIZE[1] // 2
            center_depth_value = depth_norm[center_row, center_col]
            hardware.transmit(f"D:{center_depth_value}\n")

            # Depth 화면에 정중앙 조준선 표시
            cv2.drawMarker(right_depth, (center_col, center_row), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # FPS 계산 및 원본 화면에 오버레이
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(left_original, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # [최종 레이아웃] [원본 화면] | [YOLO 화면] | [Depth 화면] 3분할 결합
            final_display = np.hstack((left_original, center_yolo, right_depth))
            cv2.imshow('Sensory Substitution: Triple View', final_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        hardware.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
