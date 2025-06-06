import torch
import numpy as np
import time
import cv2
import os
import sys

from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera
from utils import preprocess_image
from autopilot_model import E2E_CRNN
from collections import deque


# ─── 설정 ─────────────────────────────────────────────────────
MODEL_PATH     = "/home/ircv7/Embedded/Project_2/checkpoint/last_model.pt"

CAMERA_WIDTH   = 960
CAMERA_HEIGHT  = 540

FRAME_SIZE     = 224
FRAME_CHANNELS = 3

SHOW_LOGS = True

# ─── 1) 모델 인스턴스 생성 ─────────────────────────────────────
model = E2E_CRNN().cuda()
model.eval()


# ─── 2) checkpoint 불러오기 ────────────────────────────────────
ckpt = torch.load(MODEL_PATH, weights_only=True)
model.load_state_dict(ckpt, strict=False)

# ─── 3) BaseController 및 카메라 초기화 ─────────────────────────
car = BaseController("/dev/ttyUSB0", 115200)
camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

# 최초 프레임을 받아서 실제 크기 확인
frame = camera.read()
h, w = frame.shape[:2]
print(f"Camera frame size: {w}×{h}")

# 디버깅용 비디오 저장 세팅 (프레임 크기가 실제 크기와 일치하게 설정)
video_filename = "debug_output.avi"  # avi로 바꿔 봅니다.
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_fps = 20
writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (w, h))

if not writer.isOpened():
    print("Error: VideoWriter가 열리지 않았습니다. 코덱이나 경로를 확인하세요.")
    sys.exit()

speed_buffer = deque([[0.0, 0.0]] * 10, maxlen=10)

try:
    while True:
        if SHOW_LOGS:
            start_time = time.time()

        frame = camera.read()            # (h, w, 3)
        resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        tensor_img = preprocess_image(resized).to('cuda')  # [1,3,224,224] on GPU
        speed_seq = torch.tensor([list(speed_buffer)], dtype=torch.float32).to(device)  # (1,10,2)
        
        with torch.no_grad():
            out = model(tensor_img,speed_seq)
            pred = out.detach().clamp(-0.5, 0.5).cpu().numpy().flatten()

        left, right = float(pred[0]), float(pred[1])
        L = np.clip(left, -0.5, 0.5)
        R = np.clip(right, -0.5, 0.5)
        car.base_speed_ctrl(L, R)
        speed_buffer.append([L, R])


        # 시각화: 숫자 텍스트
        cv2.putText(frame, f"Left: {left:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Right: {right:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 시각화: 방향 화살표
        start_pt = (w // 2, h - 30)
        arrow_length = 60
        max_angle = np.pi / 4
        diff = (left - right)
        angle = diff * max_angle
        dx = int(arrow_length * np.sin(angle))
        dy = int(-arrow_length * np.cos(angle))
        end_pt = (start_pt[0] + dx, start_pt[1] + dy)
        cv2.arrowedLine(frame, start_pt, end_pt, (0,0,255), 3, tipLength=0.2)

        # 디버깅용 영상에 기록
        writer.write(frame)

        # 화면 출력
        cv2.imshow("Autopilot Visualization", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        prev_left, prev_right = left, right

        if SHOW_LOGS:
            fps = int(1.0 / (time.time() - start_time))
            print(f"fps: {fps}, left: {left:.3f}, right: {right:.3f}", end="\r")

except KeyboardInterrupt:
    pass

finally:
    # 정상 종료 / 예외 처리 모두 처리되도록
    writer.release()
    car.current_left  = 0.0
    car.current_right = 0.0
    cv2.destroyAllWindows()
    print(f"\nSaved video to '{video_filename}'")
