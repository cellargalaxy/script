ffmpeg -i aaaaa/demo.mkv -i aaaaa/output/demo/demucs/htdemucs/wav/vocals.wav
-map 0:v
-map 1:a
-c:v copy
-c:a aac
-b:a 192k
-shortest
output_video.mkv





bin/ffmpeg -i '../S01E01.mkv' -y -vf \
"drawtext=text='%{pts\:hms}':fontsize=96:fontcolor=white:x=10:y=10, \
drawtext=text='Seconds\: %{pts}':fontsize=96:fontcolor=white:x=10:y=100, \
drawtext=text='Frame\: %{n}':fontsize=96:fontcolor=white:x=10:y=200" \
-t 00:04:40 -map 0 -c:a copy demo.mkv

ffmpeg -i long.mkv -y -vf \
"drawtext=text='%{pts\:hms}':fontsize=96:fontcolor=white:x=10:y=10, \
drawtext=text='Seconds\: %{pts}':fontsize=96:fontcolor=white:x=10:y=100, \
drawtext=text='Frame\: %{n}':fontsize=96:fontcolor=white:x=10:y=200" \
-t 00:06:45 -map 0 -c:a copy test.mkv

bin/ffmpeg -ss 0 -to 4.785343750000001 -i output/demo/demucs/mkv.mkv -c copy -y 00000_silene.mkv
bin/ffmpeg -ss 4.78534375000000 -to 112.59159375 -i output/demo/demucs/mkv.mkv -c copy -y 00001_speech.mkv

ffmpeg -ss 0 -to 4 -i output/demo/demucs/mkv.mkv -c:v libx264 -preset veryfast -crf 28 -vf "scale=640:-1" -c:a aac -b:a 128k -y 00000_speech.mkv
ffmpeg -ss 4.78534375000000 -to 112.59159375 -i output/demo/demucs/mkv.mkv -c:v libx264 -preset veryfast -crf 28 -vf "scale=640:-1" -c:a aac -b:a 128k -y 00001_speech.mkv
