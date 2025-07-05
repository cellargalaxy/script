import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from speechbrain.inference.speaker import SpeakerRecognition
import util

audio_dir = '../aaa/output/long/segment_split'
reference_file = '../aaa/output/long/segment_split/00001_speech.wav'

# 只初始化一次模型
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)

def process_file(file):
    if not file.endswith('speech.wav'):
        return None
    audio_path = os.path.join(audio_dir, file)
    score, prediction = verification.verify_files(reference_file, audio_path)
    return (audio_path, score, prediction)

def main():
    files = util.listdir(audio_dir)
    # 过滤不需要处理的文件
    files = [file for file in files if file.endswith('speech.wav')]
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                audio_path, score, prediction = result
                print(audio_path, score, prediction)

if __name__ == "__main__":
    main()