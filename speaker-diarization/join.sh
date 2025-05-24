ffmpeg -i aaaaa/demo.mkv -i aaaaa/output/demo/demucs/htdemucs/wav/vocals.wav
-map 0:v
-map 1:a
-c:v copy
-c:a aac
-b:a 192k
-shortest
output_video.mkv


ffmpeg -i 'S01E01.mkv' -t 00:10:00 -map 0 -c copy demo.mkv
