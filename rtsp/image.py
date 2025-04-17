import cv2
import time
import os

# RTSP流地址（根据你的摄像头地址修改）
rtsp_url = "rtsp://rtspstream:jG2zm1ADvRayJ7XLoiBT5@zephyr.rtsp.stream/people"

# 创建输出目录
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# 打开RTSP流
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法打开RTSP流")
    exit()

# 记录时间
last_time = time.time()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，可能断流")
        break

    current_time = time.time()

    # 每秒保存一帧
    if current_time - last_time >= 1:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"保存截图: {filename}")
        last_time = current_time

    # 可选：限制帧率，防止CPU过载
    time.sleep(0.01)

cap.release()
