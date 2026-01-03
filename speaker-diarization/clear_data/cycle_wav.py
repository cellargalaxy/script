"""
写一个python脚本，使用纯英文编写，只用标准库，无外部依赖
1. 脚本调用方式是`file.py 5`，其中「5」是参数，单位为秒
2. 遍历脚本所在的文件夹，查找除了「mix.wav」以外的全部wav文件，得到wav文件列表，不需要递归查找
3. 如果「mix.wav」与「mix.srt」存在，则删除这两个文件
4. 对wav文件列表按文件名称排序，按顺序将这多个的wav文件，剪辑在一个新的wav中
4.1 按顺序每个文件播放5秒（脚本启动时输入的参数），遍历完一边全部文件之后，继续再遍历一遍，直到整段音频播放完成
4.2 例如有两个文件a.wav与b.wav，音频长度为23秒，新wav中则是：0-5秒对应a的0-5秒，6-10秒对应b的6-10秒，11-15秒对应a的11-15秒，16-20秒对应b的16-20秒，21-23秒对应ba的21-23秒
5. 将新的wav保存为「mix.wav」
6. 根据时间所播放的文件，生成一个字幕文件「mix.srt」，字幕内容为当前正在播放的原wav文件的文件名
7. 最后sleep一分钟再退出脚本
"""

import os
import sys
import wave
import time
from datetime import timedelta


def format_srt_time(seconds):
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def main():
    # 1. Parse argument
    if len(sys.argv) < 2:
        print("Usage: python mix_audio.py <seconds>")
        return

    try:
        interval = float(sys.argv[1])
    except ValueError:
        print("Error: Argument must be a number.")
        return

    # 2. Setup paths and cleanup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_wav = os.path.join(current_dir, "mix.wav")
    output_srt = os.path.join(current_dir, "mix.srt")

    # 3. Delete existing mix files
    for f in [output_wav, output_srt]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted existing: {os.path.basename(f)}")

    # 4. Get and sort wav files
    wav_files = [f for f in os.listdir(current_dir)
                 if f.lower().endswith(".wav") and f.lower() != "mix.wav"]
    wav_files.sort()

    if not wav_files:
        print("No source WAV files found.")
        return

    print(f"Found {len(wav_files)} files to process.")

    # Open all source files and validate parameters
    sources = []
    max_frames = 0
    params = None

    try:
        for f in wav_files:
            w = wave.open(os.path.join(current_dir, f), 'rb')
            if params is None:
                params = w.getparams()
            sources.append({"name": f, "handle": w, "frames": w.getnframes()})
            max_frames = max(max_frames, w.getnframes())

        sample_rate = params.framerate
        total_duration = max_frames / sample_rate
        interval_frames = int(interval * sample_rate)

        # 5. Create the mix.wav
        with wave.open(output_wav, 'wb') as output:
            output.setparams(params)

            srt_entries = []
            current_frame = 0
            srt_counter = 1

            while current_frame < max_frames:
                # Calculate which file to use based on the interval
                file_idx = int((current_frame / interval_frames)) % len(wav_files)
                source = sources[file_idx]

                # Determine how many frames to read
                remaining_frames = max_frames - current_frame
                chunk_size = min(interval_frames, remaining_frames)

                # Seek to the current position in the source file
                source['handle'].setpos(min(current_frame, source['frames']))

                # Read frames (if file is shorter than current_frame, readframes returns empty)
                frames_to_read = min(chunk_size, max(0, source['frames'] - current_frame))
                data = source['handle'].readframes(frames_to_read)

                # If source is shorter than the global progress, fill with silence
                if len(data) < chunk_size * params.sampwidth * params.nchannels:
                    silence_frames = chunk_size - (len(data) // (params.sampwidth * params.nchannels))
                    data += b'\x00' * (silence_frames * params.sampwidth * params.nchannels)

                output.writeframes(data)

                # 6. Prepare SRT entry
                start_time = current_frame / sample_rate
                end_time = (current_frame + chunk_size) / sample_rate
                srt_entries.append(
                    f"{srt_counter}\n{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n{source['name']}\n"
                )

                current_frame += chunk_size
                srt_counter += 1

        # Save the SRT file
        with open(output_srt, "w", encoding="utf-8") as f_srt:
            f_srt.write("\n".join(srt_entries))

        print(f"Success: 'mix.wav' and 'mix.srt' generated. Total duration: {total_duration:.2f}s")

    finally:
        # Close all file handles
        for s in sources:
            s['handle'].close()

    # 7. Sleep for 1 minute before exiting
    print("Script finished. Sleeping for 60 seconds...")
    time.sleep(5)


if __name__ == "__main__":
    main()
