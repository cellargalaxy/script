# pip install -U demucs voicefixer torchaudio soundfile
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
from voicefixer import VoiceFixer


# 1. Demucs 分离人声
def separate_vocals(input_path, output_vocals_path="vocals.wav"):
    model = pretrained.get_model('htdemucs')
    model.eval()
    device ="cuda"
    model.to(device)

    wav, sr = torchaudio.load(input_path)
    if sr != 44100:
        wav = torchaudio.transforms.Resample(sr, 44100)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.unsqueeze(0).to(device)

    print('qqqqqqqqqqqqq')
    sources = apply_model(model, wav, device=device, overlap=0.5)[0]
    print('aaaaaaaaaaaaaaa')
    vocals = sources[3].mean(0, keepdim=True).cpu()  # mono
    print('zzzzzzzzzzz')

    torchaudio.save(output_vocals_path, vocals, 44100)
    print('wwwwwwwwwww')
    return output_vocals_path


# 2. VoiceFixer 修复
def voicefixer_restore(input_path, output_path="final_clean.wav"):
    print('dddddddddddd')
    vf = VoiceFixer()
    print('ccccccccccc')
    vf.restore(
        input=input_path,
        output=output_path,
        # mode=0 , # 0=最强降噪模式，对呼吸声最狠
        cuda=True,
    )
    print('rrrrrrrrrrrr')


# 一键运行
# separate_vocals("/workspace/script/speaker-diarization/gen_subt_v7/output/xiang/extract_audio/wav.wav", "vocals.wav")
voicefixer_restore("/workspace/script/speaker-diarization/gen_subt_v7/output/xiang/extract_stem/noreverb.wav", "final_clean.wav")
print("完成！输出文件：final_clean.wav")
