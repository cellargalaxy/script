import cv2
import time
import os

# RTSP 流地址
rtsp_url = "rtsp://rtspstream:jG2zm1ADvRayJ7XLoiBT5@zephyr.rtsp.stream/people"

# 视频保存路径
output_dir = "videos"
os.makedirs(output_dir, exist_ok=True)

# 打开 RTSP 视频流
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法打开 RTSP 流")
    exit()

# 获取视频帧率和大小（某些设备不准确，必要时强制指定）
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 60:
    fps = 15  # 默认帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, 分辨率: {width}x{height}")

# 初始化视频写入器
def create_writer(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    return cv2.VideoWriter(filename, fourcc, fps, (width, height))

# 每段视频的时长（秒）
segment_duration = 60
start_time = time.time()
timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"{timestamp}.mp4")
writer = create_writer(filename)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("断流或读取失败，等待重试...")
            time.sleep(1)
            continue

        now = time.time()

        # 每到设定时长就重新生成新视频
        if now - start_time >= segment_duration:
            writer.release()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"{timestamp}.mp4")
            writer = create_writer(filename)
            start_time = now
            print(f"开始新文件：{filename}")

        writer.write(frame)

        # 控制帧率（可选，缓解树莓派负载）
        time.sleep(1 / fps)

except KeyboardInterrupt:
    print("手动中断，清理资源")

finally:
    cap.release()
    writer.release()
